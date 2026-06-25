#!/usr/bin/env python3
"""
DeepEval LLM-as-judge for the Georgia EV 42-question experiment.

This is the SECONDARY, semantic/rubric evaluation. The deterministic evaluator
(scripts/evaluate_42q_model_responses_v2.py) remains the primary factual score;
the DeepEval rubric scores here are advisory and never override deterministic
entity/list/count errors.

It judges the *existing* stored model responses (it never regenerates them)
using a local Ollama model (default gpt-oss:120b) via DeepEval's GEval metric.
Six rubric dimensions are scored 0-1 (higher = better) with a written reason:

  factual_faithfulness  - every factual claim is supported by the KB context
  completeness          - the answer addresses all parts of the question
  correctness           - selection/answer agrees with the gold answer + KB
  evidence_grounding    - companies/row_ids are traceable to the context rows
  company_grounding     - inverse of hallucinated-company risk (1.0 = no
                          out-of-KB companies named)
  usefulness            - the answer is clear, direct and useful to a reader

Context selection (kept small for judge latency; recorded per answer):
  the answer's own valid Supporting row_ids  +  the gold answer's KB rows,
  deduped and capped at --max-context-rows (context_truncated flagged if capped).

Outputs (under --out, default evaluation_v2/deepeval):
  deepeval_raw.jsonl           - one record per answer (scores + reasons + context meta)
  deepeval_scores_by_answer.csv
  deepeval_scores_by_model.csv

The run is resumable: answers already present in deepeval_raw.jsonl are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Opt out of DeepEval network telemetry before importing the package.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("ERROR_REPORTING", "0")

import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_common as v1  # noqa: E402

DEFAULT_OUT = "evaluation_v2/deepeval"
DEFAULT_JUDGE = "gpt-oss:120b"
DEFAULT_BASE_URL = "http://localhost:11434"

# Compact KB fields shown to the judge as context (drops address/lat/long noise).
CONTEXT_FIELDS = [
    "row_id", "Company", "Category", "Industry Group", "Location",
    "Primary Facility Type", "EV Supply Chain Role", "Primary OEMs",
    "Supplier or Affiliation Type", "Employment", "Product / Service",
    "EV / Battery Relevant",
]

METRIC_COLUMNS = [
    "factual_faithfulness", "completeness", "correctness",
    "evidence_grounding", "company_grounding", "usefulness",
]


def build_metrics(model):
    from deepeval.metrics import GEval
    from deepeval.test_case import SingleTurnParams as P

    return [
        ("factual_faithfulness", GEval(
            name="Factual Faithfulness",
            evaluation_steps=[
                "Read the knowledge-base rows provided in 'context'. Treat them as the only source of truth.",
                "Check every factual claim in 'actual output' (companies, counties, tiers, roles, products, numbers).",
                "Give 1.0 only if every claim is supported by the context; lower the score for each unsupported or invented claim.",
            ],
            evaluation_params=[P.INPUT, P.ACTUAL_OUTPUT, P.CONTEXT],
            model=model, async_mode=False)),
        ("completeness", GEval(
            name="Completeness",
            evaluation_steps=[
                "Compare 'actual output' to 'expected output' (the human-validated gold answer).",
                "Heavily penalize omission of companies, counties, or required fields that the gold answer contains.",
                "Give 1.0 only if the answer addresses every part of the question that the gold answer covers.",
            ],
            evaluation_params=[P.INPUT, P.ACTUAL_OUTPUT, P.EXPECTED_OUTPUT],
            model=model, async_mode=False)),
        ("correctness", GEval(
            name="Correctness",
            evaluation_steps=[
                "Compare the selection/answer in 'actual output' against the gold 'expected output' and the 'context' rows.",
                "Penalize wrong companies, wrong counts, wrong fields, or contradicted facts.",
                "Vague phrasing is acceptable; factual disagreement with the gold answer is not.",
            ],
            evaluation_params=[P.INPUT, P.ACTUAL_OUTPUT, P.EXPECTED_OUTPUT, P.CONTEXT],
            model=model, async_mode=False)),
        ("evidence_grounding", GEval(
            name="Evidence Grounding",
            evaluation_steps=[
                "Look at the Evidence table and any cited row_ids in 'actual output'.",
                "Check whether each named company is present in the 'context' rows and whether cited row_ids plausibly match the company.",
                "Give 1.0 only if the evidence is consistent and traceable to the context; lower it for fabricated or mismatched evidence.",
            ],
            evaluation_params=[P.ACTUAL_OUTPUT, P.CONTEXT],
            model=model, async_mode=False)),
        ("company_grounding", GEval(
            name="Company Grounding (inverse hallucination risk)",
            evaluation_steps=[
                "List the company names asserted in 'actual output'.",
                "A company is grounded if it appears in the 'context' rows or is an obvious spelling variant of one.",
                "Give 1.0 only if every asserted company is grounded; lower the score for each company that is absent from the context (a hallucinated company).",
            ],
            evaluation_params=[P.ACTUAL_OUTPUT, P.CONTEXT],
            model=model, async_mode=False)),
        ("usefulness", GEval(
            name="Usefulness and Readability",
            evaluation_steps=[
                "Judge whether 'actual output' is a clear, direct, well-structured answer to the question.",
                "Reward concise answers that state the result first; mildly penalize rambling, contradictions, or missing the point.",
                "This dimension is about communication quality, not factual correctness.",
            ],
            evaluation_params=[P.INPUT, P.ACTUAL_OUTPUT],
            model=model, async_mode=False)),
    ]


def context_for_answer(record, gold, by_company, kb_by_row_id, valid_row_ids, max_rows):
    """Targeted KB context: the answer's valid Supporting row_ids + the gold
    answer's KB rows, deduped and capped."""
    row_ids = []
    seen = set()

    def add_row_id(rid):
        rid = int(rid)
        if rid in valid_row_ids and rid not in seen:
            seen.add(rid)
            row_ids.append(rid)

    for rid in v1.extract_supporting_row_ids(record["response"]):
        add_row_id(rid)

    gold_basis = "supporting_row_ids"
    for company in gold.get("gold_companies", []):
        for rec in by_company.get(company, []):
            add_row_id(rec["row_id"])
    if row_ids and not v1.extract_supporting_row_ids(record["response"]):
        gold_basis = "gold_companies"
    elif row_ids:
        gold_basis = "supporting+gold"
    else:
        gold_basis = "none"

    truncated = len(row_ids) > max_rows
    row_ids = sorted(row_ids)[:max_rows]

    context = []
    for rid in row_ids:
        rec = kb_by_row_id.get(rid, {})
        compact = {f: rec.get(f) for f in CONTEXT_FIELDS if rec.get(f) is not None}
        context.append(json.dumps(compact, ensure_ascii=False))
    return context, gold_basis, truncated, len(row_ids)


def load_done(raw_path: Path):
    done = set()
    if raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                done.add((rec["arm"], rec["model_name"], int(rec["question_number"])))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def parse_args():
    p = argparse.ArgumentParser(description="DeepEval LLM-as-judge (secondary) for 42Q responses.")
    p.add_argument("--kb", default=v1.DEFAULT_KB)
    p.add_argument("--golden", default=v1.DEFAULT_GOLDEN)
    p.add_argument("--direct-dir", default=v1.DEFAULT_DIRECT_DIR)
    p.add_argument("--batchwise-dir", default=v1.DEFAULT_BATCHWISE_DIR)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--judge-model", default=DEFAULT_JUDGE)
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--models", default="", help="comma-separated model_name filter")
    p.add_argument("--questions", default="", help="comma-separated question numbers")
    p.add_argument("--limit", type=int, default=0, help="only the first N question numbers")
    p.add_argument("--max-context-rows", type=int, default=45)
    p.add_argument("--skip-direct", action="store_true")
    p.add_argument("--skip-batchwise", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records, by_company, company_aliases, county_aliases = v1.load_kb(args.kb)
    kb_by_row_id = {int(r["row_id"]): r for r in records}
    valid_row_ids = set(kb_by_row_id)
    golden, _ = v1.load_golden(args.golden, company_aliases, county_aliases)

    run_records = []
    if not args.skip_direct:
        run_records.extend(v1.direct_records(Path(args.direct_dir)))
    if not args.skip_batchwise:
        run_records.extend(v1.batchwise_records(Path(args.batchwise_dir)))

    models_filter = {m.strip() for m in args.models.split(",") if m.strip()} or None
    if models_filter:
        run_records = [r for r in run_records if r["model_name"] in models_filter]
    questions_filter = {int(q) for q in args.questions.split(",") if q.strip()} or None
    if questions_filter:
        run_records = [r for r in run_records if int(r["question_number"]) in questions_filter]
    elif args.limit:
        allowed = set(sorted({int(r["question_number"]) for r in run_records})[: args.limit])
        run_records = [r for r in run_records if int(r["question_number"]) in allowed]
    if not run_records:
        raise SystemExit("No model response records found for the requested filters.")

    from deepeval.models import OllamaModel
    from deepeval.test_case import LLMTestCase

    judge = OllamaModel(model=args.judge_model, base_url=args.base_url, temperature=0)
    metrics = build_metrics(judge)

    raw_path = out_dir / "deepeval_raw.jsonl"
    done = load_done(raw_path)
    todo = [r for r in run_records if (r["arm"], r["model_name"], int(r["question_number"])) not in done]
    print(f"Judge={args.judge_model}  answers to score={len(todo)} (skipping {len(done)} already done)")

    started = time.time()
    with raw_path.open("a", encoding="utf-8") as fout:
        for i, record in enumerate(todo, 1):
            qn = int(record["question_number"])
            if qn not in golden:
                continue
            gold = golden[qn]
            context, gold_basis, truncated, n_ctx = context_for_answer(
                record, gold, by_company, kb_by_row_id, valid_row_ids, args.max_context_rows)
            test_case = LLMTestCase(
                input=gold["question"],
                actual_output=record["response"] or "(empty answer)",
                expected_output=gold["golden_answer"],
                context=context or ["(no targeted KB rows available)"],
            )

            rec = {
                "arm": record["arm"],
                "model_name": record["model_name"],
                "question_number": qn,
                "judge_model": args.judge_model,
                "question_type": gold["question_type"],
                "judge_recommended": bool(gold["judge_recommended"]),
                "context_rows": n_ctx,
                "context_basis": gold_basis,
                "context_truncated": truncated,
                "status": "ok",
            }
            t0 = time.time()
            for name, metric in metrics:
                try:
                    metric.measure(test_case)
                    rec[name] = round(float(metric.score), 4)
                    rec[f"{name}_reason"] = (metric.reason or "")[:600]
                except Exception as exc:  # noqa: BLE001
                    rec[name] = ""
                    rec[f"{name}_reason"] = f"ERROR {type(exc).__name__}: {exc}"
                    rec["status"] = "partial_error"
            scored = [rec[c] for c in METRIC_COLUMNS if isinstance(rec.get(c), (int, float))]
            rec["deepeval_mean"] = round(sum(scored) / len(scored), 4) if scored else ""
            rec["judge_seconds"] = round(time.time() - t0, 2)

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            elapsed = time.time() - started
            print(f"[{i}/{len(todo)}] {record['arm']}::{record['model_name']} Q{qn} "
                  f"mean={rec['deepeval_mean']} ({rec['judge_seconds']}s, ctx={n_ctx}) "
                  f"elapsed={elapsed/60:.1f}m", flush=True)

    write_summaries(raw_path, out_dir)
    print(f"\nWrote {raw_path}")
    print(f"Wrote {out_dir / 'deepeval_scores_by_answer.csv'}")
    print(f"Wrote {out_dir / 'deepeval_scores_by_model.csv'}")


def write_summaries(raw_path: Path, out_dir: Path):
    rows = [json.loads(l) for l in raw_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        return
    df = pd.DataFrame(rows)
    answer_cols = (["arm", "model_name", "question_number", "question_type",
                    "judge_recommended", "context_rows", "context_basis",
                    "context_truncated"] + METRIC_COLUMNS
                   + ["deepeval_mean", "status", "judge_seconds"]
                   + [f"{c}_reason" for c in METRIC_COLUMNS])
    answer_cols = [c for c in answer_cols if c in df.columns]
    df[answer_cols].sort_values(["arm", "model_name", "question_number"]).to_csv(
        out_dir / "deepeval_scores_by_answer.csv", index=False)

    agg_rows = []
    for (arm, model), sub in df.groupby(["arm", "model_name"], dropna=False):
        row = {"arm": arm, "model_name": model, "answers_judged": len(sub)}
        for col in METRIC_COLUMNS + ["deepeval_mean"]:
            row[f"mean_{col}"] = round(pd.to_numeric(sub[col], errors="coerce").mean(), 4)
        row["mean_judge_seconds"] = round(pd.to_numeric(sub["judge_seconds"], errors="coerce").mean(), 2)
        row["partial_error_answers"] = int((sub["status"] == "partial_error").sum())
        agg_rows.append(row)
    summary = pd.DataFrame(agg_rows)
    if not summary.empty:
        summary = summary.sort_values("mean_deepeval_mean", ascending=False)
        summary["rank_by_deepeval_mean"] = range(1, len(summary) + 1)
    summary.to_csv(out_dir / "deepeval_scores_by_model.csv", index=False)


if __name__ == "__main__":
    main()
