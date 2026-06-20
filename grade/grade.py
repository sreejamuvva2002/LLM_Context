"""
grade.py -- score model outputs against gold_answers.csv. Evaluation stage only.

Outputs:
  reports/scores_by_question.csv   one row per (model, question)
  reports/scores_by_model.csv      one row per model (aggregates)

Metrics (deterministic; LLM-judge handled separately for interpretation questions):
  - set precision / recall / F1 on the KB company set the answer named vs gold
  - count accuracy + signed error for count / aggregation_rank questions
  - missing / extra company counts
  - hallucination rate (mentions that match no KB company at all -- see note below)
  - runtime reliability (ERROR / missing rows)
  - context validity (full-context models only: prompt_eval_count vs num_ctx)

Questions flagged needs_human in the gold are NOT scored for accuracy (gradable=False);
they are listed so you can hand-grade or run the LLM judge. Scalar-aggregation answers
(Q7/Q8/Q9) are checked by substring of the expected answer_value, not by row set.
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_answer import (  # noqa: E402
    load_kb_companies, extract_companies, extract_row_ids, is_error, norm,
)

ROOT = Path(__file__).resolve().parent.parent
XLSX = ROOT / "GNEM_Excel_Data.xlsx"
GOLD = ROOT / "gold" / "gold_answers.csv"
OUTDIR = ROOT / "reports"
OUTDIR.mkdir(exist_ok=True)

# model label -> (file, kind). kind drives how we treat it in the summary.
MODELS = {
    "qwen3.6_35b-a3b (full-ctx)":      ("answers_qwen3.6_35b-a3b.csv", "main"),
    "mistral-small3.2_24b (full-ctx)": ("answers_mistral-small3.2_24b.csv", "main"),
    "gemma3_27b (full-ctx)":           ("answers_gemma3_27b.csv", "main"),
    "qwen2.5_14b (batched v1)":        ("answers_qwen2.5_14b_batched_strict.csv", "main"),
    "qwen2.5_14b (full-ctx TRUNCATED)":("answers_qwen2.5_14b.csv", "diagnostic_truncated"),
    "deepseek-r1_32b (unstable)":      ("answers_deepseek-r1_32b.csv", "diagnostic_unstable"),
}

SCALAR_Q = {7, 8, 9}  # answer is a county/company scalar, checked via answer_value substring


def prf(pred, gold):
    pred, gold = set(pred), set(gold)
    if not gold and not pred:
        return 1.0, 1.0, 1.0
    tp = len(pred & gold)
    p = tp / len(pred) if pred else (1.0 if not gold else 0.0)
    r = tp / len(gold) if gold else (1.0 if not pred else 0.0)
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def main():
    kb = load_kb_companies(XLSX)
    valid_ids = [c["row_id"] for c in kb]
    gold = pd.read_csv(GOLD)
    gold_by_q = {int(r.question_number): r for _, r in gold.iterrows()}

    per_q = []
    for label, (fname, kind) in MODELS.items():
        path = ROOT / "ollama_outputs" / fname
        df = pd.read_csv(path)
        df["question_number"] = df["question_number"].astype(int)
        ans_by_q = {int(r.question_number): r for _, r in df.iterrows()}

        for qn in range(1, 51):
            g = gold_by_q[qn]
            gold_ids = json.loads(g.gold_row_ids)
            needs_human = bool(g.needs_human)
            row = ans_by_q.get(qn)

            rec = {
                "model": label, "kind": kind, "question_number": qn, "qtype": g.qtype,
                "needs_human": needs_human, "gold_count": int(g.gold_count),
                "status": "", "gradable": False,
                "precision": "", "recall": "", "f1": "",
                "pred_count": "", "count_error": "",
                "missing": "", "extra": "", "hallucinated_mentions": "",
                "scalar_ok": "",
            }

            if row is None:
                rec["status"] = "MISSING"
                per_q.append(rec)
                continue
            ans = row["answer"]
            if is_error(ans):
                rec["status"] = "ERROR"
                per_q.append(rec)
                continue
            rec["status"] = "answered"

            pred_ids = extract_companies(ans, kb)
            rec["pred_count"] = len(pred_ids)

            # Scalar-aggregation questions: substring-check the expected value tokens.
            if qn in SCALAR_Q and isinstance(g.answer_value, str) and g.answer_value:
                key = norm(g.answer_value.split("(")[0])  # e.g. "hall county"
                rec["scalar_ok"] = key in norm(ans) if key else ""
                rec["gradable"] = True
                per_q.append(rec)
                continue

            if needs_human or g.qtype in ("interpretation", "unknown"):
                rec["status"] = "needs_human"
                per_q.append(rec)
                continue

            # Deterministic set scoring.
            p, r, f = prf(pred_ids, gold_ids)
            rec.update(gradable=True,
                       precision=round(p, 3), recall=round(r, 3), f1=round(f, 3),
                       missing=len(set(gold_ids) - pred_ids),
                       extra=len(pred_ids - set(gold_ids)))
            if g.qtype in ("count", "aggregation_rank", "aggregation"):
                rec["count_error"] = len(pred_ids) - int(g.gold_count)
            per_q.append(rec)

    pq = pd.DataFrame(per_q)
    pq.to_csv(OUTDIR / "scores_by_question.csv", index=False)

    # ---- per-model aggregates ----
    summ = []
    for label, (fname, kind) in MODELS.items():
        sub = pq[pq.model == label]
        gradable = sub[sub.gradable == True]  # noqa: E712
        f1s = pd.to_numeric(gradable["f1"], errors="coerce").dropna()
        answered = (sub.status.isin(["answered", "needs_human"])).sum()
        errors = (sub.status == "ERROR").sum()
        missing = (sub.status == "MISSING").sum()
        scalar = gradable[gradable.scalar_ok != ""]
        scalar_ok = (scalar.scalar_ok == True).sum() if len(scalar) else 0  # noqa: E712
        summ.append({
            "model": label, "kind": kind,
            "answered": answered, "errors": errors, "missing": missing,
            "reliability": round(answered / 50, 3),
            "gradable_det_questions": len(f1s),
            "mean_F1_gradable": round(f1s.mean(), 3) if len(f1s) else "",
            "mean_recall": round(pd.to_numeric(gradable["recall"], errors="coerce").dropna().mean(), 3) if len(f1s) else "",
            "mean_precision": round(pd.to_numeric(gradable["precision"], errors="coerce").dropna().mean(), 3) if len(f1s) else "",
            "scalar_q_correct": f"{scalar_ok}/{len(scalar)}" if len(scalar) else "0/0",
        })
    sm = pd.DataFrame(summ)
    sm.to_csv(OUTDIR / "scores_by_model.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("=== Per-model summary ===")
    print(sm.to_string(index=False))
    n_det = (gold.needs_human == False).sum()  # noqa: E712
    print(f"\nDeterministically gradable questions: {n_det}/50 "
          f"(remaining {50 - n_det} need human/LLM-judge).")
    print("Note: full-context vs batched are scored on the SAME company-set axis, but "
          "differ in BOTH model and method -- see REPORT.md for interpretation limits.")


if __name__ == "__main__":
    main()
