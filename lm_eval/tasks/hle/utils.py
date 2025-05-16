from collections import Counter
import logging
import os
from typing import Dict, List, Optional, Literal
import string
import asyncio
import json
from pydantic import BaseModel

from simpleverify import verify_generic
from datasets import Dataset
from simpleverify.verify_generic import clean, ChatCompletionSampler
from openai import AsyncOpenAI

client = AsyncOpenAI(timeout=300.0, max_retries=1)
eval_logger = logging.getLogger(__name__)

JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available."""

def doc_to_text(doc: dict) -> str:
    return doc.get("question")

def process_docs(dataset: Dataset) -> Dataset:
    def _process_doc(doc: dict) -> dict:
        out_doc = {
            "problem": doc.get("problem", doc.get("question")),
            "answer": doc.get("answer", doc.get("orig_answer", doc.get("orig_orig_answer"))),
        }
        if getattr(doc, "few_shot", None) is not None:
            out_doc["few_shot"] = True
        return out_doc
    return dataset.map(_process_doc)

class ExtractedAnswer(BaseModel):
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int
    strict: Literal[True] # 100% reliability

async def extract_answer(question: str, correct_answer: str, response: str, model: str ="gpt-4.1-mini-2025-04-14") -> Dict:
    prompt = JUDGE_PROMPT.format(question=question, correct_answer=correct_answer, response=response)
    try:
        completion = await client.beta.chat.completions.parse(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format=ExtractedAnswer,
            max_completion_tokens=4096,
        )
        content = completion.choices[0].message.parsed
        return {
            "correct_answer": correct_answer,
            "model_answer": content.extracted_final_answer,
            "reasoning": content.reasoning,
            "correct": content.correct,
            "confidence": content.confidence
        }
    except Exception as e:
        eval_logger.warning(f"Judging failed: {e}")
        return None

def process_results(
    doc: dict,
    results: List[str],
    tokenizer=None,
    max_len=32768,
) -> Dict[str, int]:
    metrics = {"exact_match": None, "extracted_answers": [], "confidence": None}
    # Multiple results -> we are measuring cov/maj etc
    if isinstance(results[0], list):
        results = results[0]
        n_res = len(results) # e.g. 64
        n_res_list = [2**i for i in range(1, int(n_res.bit_length()))] # e.g. [2, 4, 8, 16, 32, 64]
        metrics = {
            **metrics,
            "exact_matches": [],
            **{f"cov@{n}": -1 for n in n_res_list},
            **{f"maj@{n}": -1 for n in n_res_list},
            **{f"avg@{n}": -1 for n in n_res_list},
        }
    if tokenizer is not None:
        n_stats_list = [1]
        if "n_res_list" in locals():
            n_stats_list.extend(n_res_list)
        metrics = {
            **metrics,
            **{"tok": [], "tok_think": [], "tok_ans": [], "too_long": []},
            **{f"tok@{n}": -1 for n in n_stats_list},
            **{f"tok_think@{n}": -1 for n in n_stats_list},
            **{f"tok_ans@{n}": -1 for n in n_stats_list},
            **{f"too_long@{n}": -1 for n in n_stats_list},
        }
        
        
    SEP = os.getenv("SEP", "</think>")
    
    for i, a in enumerate(results, start=1):
        if tokenizer is not None:
            parts = a.split(SEP, 1)
            metrics["tok_think"].append(len(tokenizer.tokenize(parts[0])))
            metrics["tok_ans"].append(0 if len(parts) == 1 else len(tokenizer.tokenize(parts[1])))
            metrics["tok"].append(len(tokenizer.tokenize(a)))
            metrics["too_long"].append(metrics["tok"][-1] >= max_len)
            if i in n_stats_list:
                metrics[f"tok@{i}"] = sum(metrics["tok"]) / len(metrics["tok"])
                metrics[f"tok_think@{i}"] = sum(metrics["tok_think"]) / len(metrics["tok_think"])
                metrics[f"tok_ans@{i}"] = sum(metrics["tok_ans"]) / len(metrics["tok_ans"])
                metrics[f"too_long@{i}"] = sum(metrics["too_long"]) / len(metrics["too_long"])

        cleaned = clean(a, sep=SEP)
        answer_data = asyncio.run(extract_answer(doc["problem"], doc["answer"], cleaned, model="gpt-4.1-mini-2025-04-14"))
        if answer_data is None:
            pred_ans = "-1"
            correct = False
            confidence = 0
        else:
            pred_ans = answer_data["model_answer"]
            correct = answer_data["correct"] == "yes"
            confidence = answer_data["confidence"]

        metrics["extracted_answers"].append(pred_ans)
        metrics["exact_match"] = correct
        metrics["confidence"] = confidence

        if i == 1:
            metrics["exact_match"] = correct
        elif i > 1:
            metrics["exact_matches"].append(correct)
            if i in n_res_list:
                metrics[f"cov@{i}"] = int(1 in metrics["exact_matches"])
                metrics[f"maj@{i}"] = int(doc["answer"] == Counter(metrics["extracted_answers"]).most_common(1)[0][0])
                metrics[f"avg@{i}"] = sum(metrics["exact_matches"]) / i
    
    
    return metrics
