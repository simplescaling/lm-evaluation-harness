from collections import Counter
import logging
import os
import re
import signal
from typing import Dict, List, Optional

from simpleverify import verify_options
import datasets

eval_logger = logging.getLogger(__name__)

QUERY_TEMPLATE = '{Question}'
print("QUERY_TEMPLATE: ", QUERY_TEMPLATE)

def doc_to_text(doc: dict) -> str:
    return QUERY_TEMPLATE.format(Question=doc.get("problem", doc.get("question")))

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        out_doc = {
            "problem": doc.get("problem", doc.get("question")),
            "answer": doc.get("answer", doc.get("orig_answer", doc.get("orig_orig_answer"))),
        }
        if getattr(doc, "few_shot", None) is not None:
            out_doc["few_shot"] = True
        return out_doc
    return dataset.map(_process_doc)

def process_results(
    doc: dict, 
    results: List[str],
    tokenizer=None,
    max_len=32768,
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
            **{"tok": [], "tok_think": [], "tok_ans": [], "too_long": []},
            **{f"tok@{n}": -1 for n in n_stats_list},
            **{f"tok_think@{n}": -1 for n in n_stats_list},
            **{f"tok_ans@{n}": -1 for n in n_stats_list},
            **{f"too_long@{n}": -1 for n in n_stats_list},
        }

    if isinstance(doc["answer"], str) and doc["answer"].isdigit():
        gt = str(int(doc["answer"])) # 023 -> 23
    else:
        gt = str(doc["answer"])

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

        o = [gt] + list(set(metrics["extracted_answers"]) - {gt})
        _, tmp = verify_options(a, o, sep=SEP)
        print("a, opt, gt:", a, tmp, gt)
        a = tmp
        metrics["extracted_answers"].append(a)
        a = int(a == gt)
        if not(a): # Optional logging
            print("Marked incorrect\na " + metrics["extracted_answers"][-1] + "\ndoc['answer'] " + gt)
        if i == 1:
            metrics["exact_match"] = a
            if "exact_matches" in metrics:
                metrics["exact_matches"].append(a)
        elif i > 1:
            metrics["exact_matches"].append(a)
            if i in n_res_list:
                metrics[f"cov@{i}"] = int(1 in metrics["exact_matches"])
                metrics[f"maj@{i}"] = int(gt == Counter(metrics["extracted_answers"]).most_common(1)[0][0])
                metrics[f"avg@{i}"] = sum(metrics["exact_matches"]) / i

    return metrics


from simpleverify import verify
def process_results_v2(
    doc: dict, 
    results: List[str],
    tokenizer=None,
    max_len=32768,
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
            **{"tok": [], "tok_think": [], "tok_ans": [], "too_long": []},
            **{f"tok@{n}": -1 for n in n_stats_list},
            **{f"tok_think@{n}": -1 for n in n_stats_list},
            **{f"tok_ans@{n}": -1 for n in n_stats_list},
            **{f"too_long@{n}": -1 for n in n_stats_list},
        }

    if isinstance(doc["answer"], str) and doc["answer"].isdigit():
        gt = str(int(doc["answer"])) # 023 -> 23
    else:
        gt = str(doc["answer"])

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

        match, x, y = verify(a, gt, sep=SEP)[0]
        print("a, x, y, match:", a, x, y, match)
        metrics["extracted_answers"].append(y if match else x)
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

    return metrics
