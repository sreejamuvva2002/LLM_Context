#!/usr/bin/env python3
"""
Optional LLM-as-judge pass for the Georgia EV 42-question evaluation.

Use this only as the second evaluation type for interpretive/risk-style
questions. The deterministic evaluator remains the primary score for list,
count, lookup, and aggregation questions.

The judge should not be the model under test. By default, this script skips rows
where judge_model == candidate model_name.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import requests

from evaluate_42q_model_responses import (
    DEFAULT_BATCHWISE_DIR,
    DEFAULT_DIRECT_DIR,
    DEFAULT_GOLDEN,
    DEFAULT_KB,
    DEFAULT_OUT,
    batchwise_records,
    direct_records,
    load_golden,
    load_kb,
)


OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
REQUEST_TIMEOUT = 1200

JUDGE_SYSTEM_PROMPT = """
You are an external evaluator for a structured knowledge-base QA experiment.

Score the candidate answer only against the provided Georgia EV supply-chain
knowledge base and the user question. Do not use outside knowledge.

Return strict JSON only.
"""

JUDGE_USER_TEMPLATE = """
QUESTION:
{question}

KNOWLEDGE BASE (JSONL, one record per line):
{kb_jsonl}

CANDIDATE MODEL:
{candidate_model}

CANDIDATE ANSWER:
{candidate_answer}

RUBRIC:
- faithfulness: 0, 1, or 2.
  2 = every factual claim is supported by the KB.
  1 = mostly grounded but contains a minor unsupported or ambiguous claim.
  0 = multiple unsupported claims or invented facts.
- completeness: 0, 1, or 2.
  2 = addresses all parts of the question.
  1 = partially addresses the question.
  0 = mostly fails to answer the question.
- correctness: 0, 1, or 2.
  2 = selection/reasoning is defensible given the KB.
  1 = partly correct but misses important evidence or includes some wrong evidence.
  0 = incorrect selection/reasoning.
- hallucinated_company: true if the answer names a company absent from the KB.
- needs_human_review: true if the question or answer is ambiguous enough that a
  human should inspect it before using the score in a paper.

Return JSON exactly in this shape:
{{
  "faithfulness": 0,
  "completeness": 0,
  "correctness": 0,
  "hallucinated_company": false,
  "needs_human_review": false,
  "notes": "<one concise sentence>"
}}
"""


def call_judge(judge_model, prompt, num_ctx, temperature):
    payload = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT.strip()},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "num_ctx": num_ctx,
            "temperature": temperature,
            "seed": 42,
        },
    }
    response = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    content = ((data.get("message") or {}).get("content") or "").strip()
    parsed = parse_json_object(content)
    return parsed, content, data


def parse_json_object(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.I | re.S).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    return json.loads(text)


def kb_jsonl(records):
    return "\n".join(json.dumps(record, ensure_ascii=False) for record in records)


def load_candidate_records(args):
    rows = []
    if not args.skip_direct:
        rows.extend(direct_records(Path(args.direct_dir)))
    if not args.skip_batchwise:
        rows.extend(batchwise_records(Path(args.batchwise_dir)))
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Run optional LLM judge for 42Q interpretive answers.")
    parser.add_argument("--judge-model", required=True, help="External/non-self judge model in Ollama.")
    parser.add_argument("--kb", default=DEFAULT_KB)
    parser.add_argument("--golden", default=DEFAULT_GOLDEN)
    parser.add_argument("--direct-dir", default=DEFAULT_DIRECT_DIR)
    parser.add_argument("--batchwise-dir", default=DEFAULT_BATCHWISE_DIR)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--num_ctx", type=int, default=65536)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--all-questions", action="store_true", help="Judge all questions, not only judge_recommended.")
    parser.add_argument("--allow-self-judge", action="store_true")
    parser.add_argument("--skip-direct", action="store_true")
    parser.add_argument("--skip-batchwise", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    records, _, company_aliases, county_aliases = load_kb(args.kb)
    golden, _ = load_golden(args.golden, company_aliases, county_aliases)
    kb_text = kb_jsonl(records)
    candidates = load_candidate_records(args)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_judge = args.judge_model.replace(":", "_").replace("/", "_").replace(".", "_")
    jsonl_path = out_dir / f"judge_scores_{safe_judge}.jsonl"
    csv_path = out_dir / f"judge_scores_{safe_judge}.csv"

    output_rows = []
    with jsonl_path.open("w", encoding="utf-8") as out:
        for record in candidates:
            qn = int(record["question_number"])
            gold = golden.get(qn)
            if not gold:
                continue
            if not args.all_questions and not gold["judge_recommended"]:
                continue

            base = {
                "judge_model": args.judge_model,
                "arm": record["arm"],
                "candidate_model": record["model_name"],
                "question_number": qn,
                "question_type": gold["question_type"],
                "judge_recommended": gold["judge_recommended"],
                "status": "",
                "faithfulness": "",
                "completeness": "",
                "correctness": "",
                "judge_total": "",
                "hallucinated_company": "",
                "needs_human_review": "",
                "notes": "",
                "raw_judge_response": "",
            }

            if not args.allow_self_judge and record["model_name"] == args.judge_model:
                base["status"] = "skipped_self_judge"
                output_rows.append(base)
                out.write(json.dumps(base, ensure_ascii=False) + "\n")
                out.flush()
                continue

            prompt = JUDGE_USER_TEMPLATE.format(
                question=gold["question"],
                kb_jsonl=kb_text,
                candidate_model=record["model_name"],
                candidate_answer=record["response"],
            )
            try:
                parsed, raw, data = call_judge(args.judge_model, prompt, args.num_ctx, args.temperature)
                faithfulness = int(parsed.get("faithfulness"))
                completeness = int(parsed.get("completeness"))
                correctness = int(parsed.get("correctness"))
                base.update({
                    "status": "ok",
                    "faithfulness": faithfulness,
                    "completeness": completeness,
                    "correctness": correctness,
                    "judge_total": faithfulness + completeness + correctness,
                    "hallucinated_company": bool(parsed.get("hallucinated_company")),
                    "needs_human_review": bool(parsed.get("needs_human_review")),
                    "notes": str(parsed.get("notes", "")),
                    "raw_judge_response": raw,
                    "prompt_eval_count": data.get("prompt_eval_count"),
                    "eval_count": data.get("eval_count"),
                })
            except Exception as exc:
                base.update({
                    "status": "error",
                    "notes": f"{type(exc).__name__}: {exc}",
                })
            output_rows.append(base)
            out.write(json.dumps(base, ensure_ascii=False) + "\n")
            out.flush()
            print(
                f"{base['status']} | Q{qn} | {record['arm']} | "
                f"{record['model_name']} | total={base.get('judge_total', '')}"
            )

    df = pd.DataFrame(output_rows)
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {jsonl_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
