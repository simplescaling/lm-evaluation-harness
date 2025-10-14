from collections import Counter
import copy
import logging
import os
import numpy as np
from typing import Dict, List, Optional

from simpleverify import verify_generic
from simpleverify import verify_math
from datasets import Dataset

eval_logger = logging.getLogger(__name__)

QUERY_TEMPLATE = '{Question}'
print("QUERY_TEMPLATE: ", QUERY_TEMPLATE)

def doc_to_text(doc: dict) -> str:
    return QUERY_TEMPLATE.format(Question=doc["problem"])

def process_docs(dataset: Dataset) -> Dataset:
    def _process_doc(doc: dict) -> dict:
        solution = doc.get("solution", doc.get("orig_solution", doc.get("orig_orig_solution")))
        problem = doc.get("problem", doc.get("orig_problem", doc.get("orig_orig_problem")))
        answer = doc.get("answer", doc.get("orig_answer", doc.get("orig_orig_answer")))
        if solution is None:
            print("Warning: No solution found; DOC:", doc)
        out_doc = {
            "problem": problem,
            "solution": solution,
            "answer": answer,
        }
        if getattr(doc, "few_shot", None) is not None:
            out_doc["few_shot"] = True
        return out_doc
    return dataset.map(_process_doc)

def process_docs_openai_math_cot_quality_check(dataset: Dataset) -> Dataset:
    def _process_doc(doc: dict) -> dict:
        problem = doc.get("orig_problem", doc.get("orig_orig_problem"))
        solution = doc.get("orig_solution", doc.get("orig_orig_solution"))
        answer = doc.get("orig_answer", doc.get("orig_orig_answer"))
        thinking_trajectory = doc.get("thinking_trajectory", doc.get("orig_thinking_trajectory", doc.get("refined_thinking_trajectory")))
        try:
            out_doc = {
                "problem": problem,
                "solution": solution,
                "answer": answer,
                "thinking_trajectory": thinking_trajectory[:-1],
            }
            if getattr(doc, "few_shot", None) is not None:
                out_doc["few_shot"] = True
            return out_doc
        except:
            return {'problem': 'Drop', 'solution': 'Drop', 'answer': 'Drop', 'thinking_trajectory': ['Drop']}
    processed_dataset = dataset.map(_process_doc)
    processed_dataset = processed_dataset.filter(lambda x: x['problem'] != 'Drop')
    return processed_dataset

def process_results(
    doc: dict,
    results: List[str],
    tokenizer=None,
    max_len=32768,
    repeat_window=100,
    unique_thresh=0.2,
    addtokens=False,
) -> Dict[str, int]:
    metrics = {"exact_match": None, "extracted_answers": []}
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
            **{"tok": [], "tok_think": [], "tok_ans": [], "too_long": [], "repetitive": []},
            **{f"tok@{n}": -1 for n in n_stats_list},
            **{f"tok_think@{n}": -1 for n in n_stats_list},
            **{f"tok_ans@{n}": -1 for n in n_stats_list},
            **{f"too_long@{n}": -1 for n in n_stats_list},
            **{f"repetitive@{n}": -1 for n in n_stats_list},
        }

    if isinstance(doc["answer"], str) and doc["answer"].isdigit():
        gt = str(int(doc["answer"])) # 023 -> 23
    else:
        gt = str(doc["answer"])

    SEP = os.getenv("SEP", "</think>")
    VERIFYFN = os.getenv("VERIFYFN", "verify_math")

    for i, a in enumerate(results, start=1):
        if tokenizer is not None:
            parts = a.split(SEP, 1)
            metrics["tok_think"].append(len(tokenizer.tokenize(parts[0])))
            metrics["tok_ans"].append(0 if len(parts) == 1 else len(tokenizer.tokenize(parts[1])))
            metrics["tok"].append(len(toks := tokenizer.tokenize(a)))
            metrics["too_long"].append(metrics["tok"][-1] >= max_len)
            metrics["repetitive"].append(len(set(w := toks[-repeat_window:]))/len(w) < unique_thresh)
            if i in n_stats_list:
                metrics[f"tok@{i}"] = sum(metrics["tok"]) / len(metrics["tok"])
                metrics[f"tok_think@{i}"] = sum(metrics["tok_think"]) / len(metrics["tok_think"])
                metrics[f"tok_ans@{i}"] = sum(metrics["tok_ans"]) / len(metrics["tok_ans"])
                metrics[f"too_long@{i}"] = sum(metrics["too_long"]) / len(metrics["too_long"])
                metrics[f"repetitive@{i}"] = sum(metrics["repetitive"]) / len(metrics["repetitive"])

        if VERIFYFN == "verify_math":
            match, x, y = verify_math(a, gt, sep=SEP)[0]
        elif VERIFYFN == "verify_generic":
            match, x, y = verify_generic(a, gt, sep=SEP, m='gpt-4.1-mini-2025-04-14')[0]
        
        metrics["extracted_answers"].append(gt if match else x)
        if not(match): # Optional logging
            print("Marked incorrect\na " + metrics["extracted_answers"][-1] + "\ndoc['answer'] " + gt)
        if i == 1:
            metrics["exact_match"] = match
            if "exact_matches" in metrics:
                metrics["exact_matches"].append(match)
        elif i > 1:
            metrics["exact_matches"].append(match)
            if i in n_res_list:
                metrics[f"cov@{i}"] = int(1 in metrics["exact_matches"])
                metrics[f"maj@{i}"] = int(gt == Counter(metrics["extracted_answers"]).most_common(1)[0][0])
                metrics[f"avg@{i}"] = sum(metrics["exact_matches"]) / i

    if addtokens:
        addtoks = [2**x for x in range(6, int(np.log2(max_len)) + 1)]
        metrics = {(k.replace("@", f"@{t}@") if "@" in k else f"{k}@{t}"): copy.copy(v) for k, v in metrics.items() for t in addtoks}
        for t in addtoks:
            for i, t_used in enumerate(metrics[f"tok@{t}"]):
                if t_used > t:
                    if i == 0:
                        metrics[f"exact_match@{t}"] = 0
                    metrics[f"exact_matches@{t}"][i] = 0
                    metrics[f"tok@{t}"][i] = t
                    if metrics[f"tok_think@{t}"][i] > t:
                        metrics[f"tok_think@{t}"][i] = t
                        metrics[f"tok_ans@{t}"][i] = 0
                    else:
                        metrics[f"tok_ans@{t}"][i] = t - metrics[f"tok_think@{t}"][i]
                    metrics[f"too_long@{t}"][i] = 1
                    # make it unique so maj is unaffected
                    metrics[f"extracted_answers@{t}"][i] = "Too long at " + str(t)

            for i in n_stats_list:
                metrics[f"tok@{t}@{i}"] = sum(metrics[f"tok@{t}"][:i]) / len(metrics[f"tok@{t}"][:i])
                metrics[f"tok_think@{t}@{i}"] = sum(metrics[f"tok_think@{t}"][:i]) / len(metrics[f"tok_think@{t}"][:i])
                metrics[f"tok_ans@{t}@{i}"] = sum(metrics[f"tok_ans@{t}"][:i]) / len(metrics[f"tok_ans@{t}"][:i])
                metrics[f"too_long@{t}@{i}"] = sum(metrics[f"too_long@{t}"][:i]) / len(metrics[f"too_long@{t}"][:i])
                if i in n_res_list:
                    metrics[f"cov@{t}@{i}"] = int(1 in metrics[f"exact_matches@{t}"][:i])
                    metrics[f"maj@{t}@{i}"] = int(gt == Counter(metrics[f"extracted_answers@{t}"][:i]).most_common(1)[0][0])
                    metrics[f"avg@{t}@{i}"] = sum(metrics[f"exact_matches@{t}"][:i]) / i

    return metrics
