from collections import Counter
import os
import time
from typing import Any, Dict, List, Optional
import random
import re

from simpleverify.verify import clean, ChatCompletionSampler
import datasets

QUERY_TEMPLATE = "{Question}\n\nA) {choice1}\nB) {choice2}\nC) {choice3}\nD) {choice4}"
QUERY_TEMPLATE_API = "{Question}\nAnswer Choices:\n(A) {choice1}\n(B) {choice2}\n(C) {choice3}\n(D) {choice4}"
print("QUERY_TEMPLATE: ", QUERY_TEMPLATE)

EXTRACTION_TEMPLATE = r"""
Look at the following question and an attempt by a student and extract which choice among A, B, C, D the student picked. If the student did not pick any choice, respond with "-1".

Examples:

    Question: ...
    Attempt: Answer: **A**

A

    Question: A) Dinosaur B) Elephant C) Cat D) Dog
    Attempt: ...The answer is therefore Elephant...

B

    Question: ...
    Attempt: Answer: None of the above

-1

    Question: ...
    Attempt: ...Answer: D), because...

D

    Question: ...
(A) 7 
(B) 8 
(C) 4 
(D) 10
    Attempt: 4

C

    Question: ...
    Attempt: ...\\boxed{C}...

C

---

YOUR TASK


Respond only with the capitalized alphabetic letter (without quotes) or -1. Do not include a rationale.

    Question: %(expression1)s
    Attempt: %(expression2)s
""".strip()

def extract_answer(sampler, question: str, attempt: str):
   prompt = EXTRACTION_TEMPLATE % {"expression1": question, "expression2": attempt}
   response = sampler([dict(content=prompt, role="user")])
   return response

def process_results(
    doc: dict, 
    results: List[str],
    tokenizer = None,
    max_len = 32768,
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

    sampler = ChatCompletionSampler(model="gpt-4o-mini")
    question = QUERY_TEMPLATE_API.format(Question=doc["Question"], choice1=doc["choice1"], choice2=doc["choice2"], choice3=doc["choice3"], choice4=doc["choice4"])
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

        a = clean(a, sep=SEP)

        if a in ["a", "b", "c", "d"]:
            a = a.upper()
        elif a not in ["A", "B", "C", "D"]:
            a = extract_answer(sampler, question, a)
            if a not in ["A", "B", "C", "D"]:
                print(f"Warning: Default to A as given {results[i-1]} extracted {a}")
                a = "A"

        metrics["extracted_answers"].append(a)
        a = int(a == doc["answer"])
        if not(a): # Optional logging
            print("Marked incorrect\na " + metrics["extracted_answers"][-1] + "\ndoc['answer'] " + doc["answer"])
        if i == 1:
            metrics["exact_match"] = a
            if "exact_matches" in metrics:
                metrics["exact_matches"].append(a)
        elif i > 1:
            metrics["exact_matches"].append(a)
            if i in n_res_list:
                metrics[f"cov@{i}"] = int(1 in metrics["exact_matches"])
                metrics[f"maj@{i}"] = int(doc["answer"] == Counter(metrics["extracted_answers"]).most_common(1)[0][0])
                metrics[f"avg@{i}"] = sum(metrics["exact_matches"]) / i

    return metrics

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        choices = [
            doc["Incorrect Answer 1"],
            doc["Incorrect Answer 2"],
            doc["Incorrect Answer 3"],
            doc["Correct Answer"],
        ]

        random.shuffle(choices)
        correct_answer_index = choices.index(doc["Correct Answer"])

        out_doc = {
            "choice1": choices[0],
            "choice2": choices[1],
            "choice3": choices[2],
            "choice4": choices[3],
            "answer": f"{chr(65 + correct_answer_index)}",
        }
        return out_doc

    return dataset.map(_process_doc)

def doc_to_text_gpqa(doc: dict) -> str:
    return QUERY_TEMPLATE.format(Question=doc["Question"], choice1=doc["choice1"], choice2=doc["choice2"], choice3=doc["choice3"], choice4=doc["choice4"])
