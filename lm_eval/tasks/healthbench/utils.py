import os
import asyncio
import json
import logging
import re
import copy
from collections import defaultdict
from typing import Dict, List
import numpy as np
from datasets import Dataset
from openai import AsyncOpenAI
from pydantic import BaseModel


# Initialize client lazily to avoid import-time API key requirement
_client = None


def get_client():
    """Get or create AsyncOpenAI client."""
    global _client
    if _client is None:
        _client = AsyncOpenAI(timeout=300.0, max_retries=1)
    return _client


eval_logger = logging.getLogger(__name__)

GRADER_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

# Example 1
For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", you should return a json like this:

```json
{
  "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
  "criteria_met": false
}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:

```json
{
  "explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",
  "criteria_met": false
}
```

As another example, if the critera says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteria_met": false
}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()


def parse_json_to_dict(json_string: str) -> dict:
    """Parse JSON from markdown-style code blocks."""
    json_cleaned = re.sub(r"^```json\s*|\s*```$", "", json_string.strip())
    try:
        return json.loads(json_cleaned)
    except json.JSONDecodeError as e:
        eval_logger.warning(f"JSON decoding failed: {e}")
        return {}


class RubricItem:
    """Represents a single rubric criterion."""

    def __init__(self, criterion: str, points: float, tags: list[str]):
        self.criterion = criterion
        self.points = points
        self.tags = tags

    def __str__(self):
        return f"[{self.points}] {self.criterion}"

    def to_dict(self):
        return {
            "criterion": self.criterion,
            "points": self.points,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            criterion=d["criterion"],
            points=d["points"],
            tags=d["tags"],
        )


class GradingResponse(BaseModel):
    """Pydantic model for grader response."""

    explanation: str
    criteria_met: bool


def doc_to_text(doc: dict) -> str:
    """Convert document to text prompt."""
    # The prompt field contains a list of message dicts
    prompt_messages = doc.get("prompt", [])
    if not prompt_messages:
        return ""

    # Format as conversation
    conversation = []
    for msg in prompt_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        conversation.append(f"{role}: {content}")

    return "\n\n".join(conversation)


def process_docs(dataset: Dataset) -> Dataset:
    """Process documents from the dataset."""

    def _process_doc(doc: dict) -> dict:
        # Keep rubrics as dicts for Arrow serialization
        # They'll be converted to RubricItem objects in process_results
        out_doc = {
            "prompt": doc.get("prompt", []),
            "rubrics": doc.get("rubrics", []),  # Keep as list of dicts
            "example_tags": doc.get("example_tags", []),
            "prompt_id": doc.get("prompt_id", ""),
        }
        return out_doc

    return dataset.map(_process_doc)


async def grade_rubric_item(
    conversation: str, rubric_item: RubricItem, model: str = "gpt-4.1-2025-04-14"
) -> dict:
    """Grade a single rubric item using LLM-as-judge."""
    grader_prompt = GRADER_TEMPLATE.replace("<<conversation>>", conversation).replace(
        "<<rubric_item>>", str(rubric_item)
    )

    client = get_client()
    max_retries = 3
    for attempt in range(max_retries):
        try:
            completion = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": grader_prompt}],
                temperature=0,
                max_tokens=2048,
            )
            grading_response = completion.choices[0].message.content
            grading_response_dict = parse_json_to_dict(grading_response)

            if "criteria_met" in grading_response_dict:
                label = grading_response_dict["criteria_met"]
                if label is True or label is False:
                    return grading_response_dict

            eval_logger.warning(
                f"Grading failed due to bad JSON output (attempt {attempt + 1}/{max_retries}), retrying..."
            )
        except Exception as e:
            eval_logger.warning(
                f"Grading failed with error: {e} (attempt {attempt + 1}/{max_retries})"
            )

    # If all retries fail, return a default response
    eval_logger.error(f"All grading attempts failed for rubric item: {rubric_item}")
    return {"explanation": "Grading failed", "criteria_met": False}


def calculate_score(
    rubric_items: list[RubricItem], grading_response_list: list[dict]
) -> float | None:
    """Calculate overall score from rubric items and grades."""
    total_possible_points = sum(
        rubric_item.points for rubric_item in rubric_items if rubric_item.points > 0
    )
    if total_possible_points == 0:
        return None

    achieved_points = sum(
        rubric_item.points
        for rubric_item, grading_response in zip(
            rubric_items, grading_response_list, strict=True
        )
        if grading_response["criteria_met"]
    )
    overall_score = achieved_points / total_possible_points
    return overall_score


async def grade_sample_async(
    prompt_messages: list[dict],
    response_text: str,
    rubric_items: list[RubricItem],
    example_tags: list[str],
    model: str = "gpt-4.1-2025-04-14",
) -> tuple[dict, list[dict]]:
    """Grade a complete sample asynchronously."""
    # Construct conversation with response
    convo_with_response = prompt_messages + [
        {"content": response_text, "role": "assistant"}
    ]
    convo_str = "\n\n".join(
        [f"{m['role']}: {m['content']}" for m in convo_with_response]
    )

    # Grade all rubric items
    grading_tasks = [
        grade_rubric_item(convo_str, rubric_item, model) for rubric_item in rubric_items
    ]
    grading_response_list = await asyncio.gather(*grading_tasks)

    # Calculate overall score
    overall_score = calculate_score(rubric_items, grading_response_list)

    metrics = {
        "overall_score": overall_score if overall_score is not None else 0.0,
    }

    # Add example-level tag scores
    for tag in example_tags:
        metrics[tag] = overall_score if overall_score is not None else 0.0

    # Add rubric-level tag scores
    rubric_tag_items_grades = defaultdict(list)
    for rubric_item, grading_response in zip(rubric_items, grading_response_list):
        for tag in rubric_item.tags:
            rubric_tag_items_grades[tag].append((rubric_item, grading_response))

    for tag, items_grades in rubric_tag_items_grades.items():
        items, grades = zip(*items_grades)
        score = calculate_score(list(items), list(grades))
        if score is not None:
            metrics[tag] = score

    # Prepare rubric items with grades for output
    rubric_items_with_grades = []
    for rubric_item, grading_response in zip(rubric_items, grading_response_list):
        rubric_items_with_grades.append(
            {
                **rubric_item.to_dict(),
                "criteria_met": grading_response.get("criteria_met", False),
                "explanation": grading_response.get(
                    "explanation", "No explanation provided"
                ),
            }
        )

    return metrics, rubric_items_with_grades


def process_results(doc: dict, results: List[str], repeat_window=100, unique_thresh=0.2, tokenizer=None, addtokens=False, max_len=32768, **kwargs) -> Dict[str, float]:
    """Process results for a single document."""
    if not results or not results[0]:
        eval_logger.warning("Empty results received")
        return {
            "overall_score": 0.0,
            "exact_match": 0.0,
        }

    # Get the generated response
    response_text = results[0] if isinstance(results[0], str) else str(results[0])

    # Extract required fields from doc
    prompt_messages = doc.get("prompt", [])
    rubric_items = doc.get("rubrics", [])
    example_tags = doc.get("example_tags", [])

    # Convert rubrics to RubricItem objects if needed
    if rubric_items and isinstance(rubric_items[0], dict):
        rubric_items = [RubricItem.from_dict(r) for r in rubric_items]

    # Grade the sample synchronously (wrap async function)
    try:
        metrics, _ = asyncio.run(
            grade_sample_async(
                prompt_messages=prompt_messages,
                response_text=response_text,
                rubric_items=rubric_items,
                example_tags=example_tags,
                model="gpt-4.1-2025-04-14",
            )
        )
    except Exception as e:
        eval_logger.error(f"Error grading sample: {e}")
        metrics = {"overall_score": 0.0}

    # Add exact_match as alias for overall_score for compatibility
    metrics["exact_match"] = metrics.get("overall_score", 0.0)

    # Don't include rubric_grades in returned metrics - it's not a scalar
    # and will cause aggregation errors. It's already logged if needed.

    #if True:
    if tokenizer is not None:
        n_stats_list = [1]
        metrics = {
            **metrics,
            **{"tok": [], "tok_think": [], "tok_ans": [], "too_long": [], "repetitive": []},
            **{f"tok@{n}": -1 for n in n_stats_list},
            **{f"tok_think@{n}": -1 for n in n_stats_list},
            **{f"tok_ans@{n}": -1 for n in n_stats_list},
            **{f"too_long@{n}": -1 for n in n_stats_list},
            **{f"repetitive@{n}": -1 for n in n_stats_list},
        }
        if True:
            SEP = os.getenv("SEP", "</think>")
            parts = response_text.split(SEP, 1)
            metrics["tok_think"].append(len(tokenizer.tokenize(parts[0])))
            metrics["tok_ans"].append(0 if len(parts) == 1 else len(tokenizer.tokenize(parts[1])))
            metrics["tok"].append(len(toks := tokenizer.tokenize(response_text)))
            metrics["too_long"].append(metrics["tok"][-1] >= max_len)
            metrics["repetitive"].append(len(set(w := toks[-repeat_window:]))/len(w) < unique_thresh)
            i=1
            if i in n_stats_list:
                metrics[f"tok@{i}"] = sum(metrics["tok"]) / len(metrics["tok"])
                metrics[f"tok_think@{i}"] = sum(metrics["tok_think"]) / len(metrics["tok_think"])
                metrics[f"tok_ans@{i}"] = sum(metrics["tok_ans"]) / len(metrics["tok_ans"])
                metrics[f"too_long@{i}"] = sum(metrics["too_long"]) / len(metrics["too_long"])
                metrics[f"repetitive@{i}"] = sum(metrics["repetitive"]) / len(metrics["repetitive"])


    if addtokens:
        addtoks = [2**x for x in range(6, int(np.log2(max_len)) + 1)]
        metrics_tok = ["tok", "tok_think", "tok_ans", "too_long", "repetitive", "exact_match", "tok@1", "tok_think@1", "tok_ans@1", "too_long@1", "repetitive@1"]
        metrics_tok_dict = {(k.replace("@", f"@{t}@") if "@" in k else f"{k}@{t}"): copy.copy(metrics[k]) for k in metrics_tok for t in addtoks}
        metrics = {**metrics_tok_dict, **{k: v for k, v in metrics.items() if k not in metrics_tok}}
        for t in addtoks:
            for i, t_used in enumerate(metrics[f"tok@{t}"]):
                if t_used > t:
                    if i == 0:
                        metrics[f"exact_match@{t}"] = 0
                    # metrics[f"exact_matches@{t}"][i] = 0
                    metrics[f"tok@{t}"][i] = t
                    if metrics[f"tok_think@{t}"][i] > t:
                        metrics[f"tok_think@{t}"][i] = t
                        metrics[f"tok_ans@{t}"][i] = 0
                    else:
                        metrics[f"tok_ans@{t}"][i] = t - metrics[f"tok_think@{t}"][i]
                    metrics[f"too_long@{t}"][i] = 1

            for i in n_stats_list:
                metrics[f"tok@{t}@{i}"] = sum(metrics[f"tok@{t}"][:i]) / len(metrics[f"tok@{t}"][:i])
                metrics[f"tok_think@{t}@{i}"] = sum(metrics[f"tok_think@{t}"][:i]) / len(metrics[f"tok_think@{t}"][:i])
                metrics[f"tok_ans@{t}@{i}"] = sum(metrics[f"tok_ans@{t}"][:i]) / len(metrics[f"tok_ans@{t}"][:i])
                metrics[f"too_long@{t}@{i}"] = sum(metrics[f"too_long@{t}"][:i]) / len(metrics[f"too_long@{t}"][:i])
    print(metrics)

    return metrics

