"""
judge_interpretation.py -- LLM-as-judge for the INTERPRETATION subset only.

STATUS: design stub. Run this ONLY for the ~21 questions flagged needs_human in
gold_answers.csv where a deterministic rule is not defensible (risk / interpretation /
under-specified). Do NOT use it for list/count/filter questions -- those are graded
deterministically by grade.py, which is more rigorous.

Rules to keep this honest:
- The judge must NOT be the model under test (no self-grading). Use a strong external
  judge (e.g. Claude Opus via API) or, if offline-only, the largest available local model
  that is NOT the one being graded.
- The judge is given: the question, the FULL KB (kb_full.jsonl), and the model answer.
- The judge returns a small fixed rubric, not a free-form grade.

Rubric (0-2 each):
  faithfulness   : every claim is supported by the KB (no fabricated companies/values)
  completeness   : addresses all parts of the question
  correctness    : the reasoning/selection is defensible given the KB
Plus a boolean: hallucinated_company (named a company not in the KB).

This file intentionally does not execute a run by default; wire in your judge client
and the question list from gold (needs_human == True) when you are ready.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KB = ROOT / "ollama_outputs" / "kb_full.jsonl"
GOLD = ROOT / "gold" / "gold_answers.csv"

RUBRIC_PROMPT = """You are grading one answer to a knowledge-base question.
You are given the full knowledge base (JSONL) and a candidate answer.
Score ONLY against the knowledge base. Do not use outside knowledge.

Return STRICT JSON:
{
  "faithfulness": 0|1|2,      // 2 = every claim supported by the KB, 0 = multiple unsupported claims
  "completeness": 0|1|2,      // 2 = addresses all parts of the question
  "correctness": 0|1|2,       // 2 = selection/reasoning defensible given the KB
  "hallucinated_company": true|false,  // named any company absent from the KB
  "notes": "<one sentence>"
}

QUESTION:
{question}

KNOWLEDGE BASE (JSONL, one row per line):
{kb}

CANDIDATE ANSWER:
{answer}
"""


def load_needs_human():
    import pandas as pd
    g = pd.read_csv(GOLD)
    return g[g.needs_human == True]["question_number"].astype(int).tolist()  # noqa: E712


def build_prompt(question, answer):
    kb_text = KB.read_text(encoding="utf-8")
    return (RUBRIC_PROMPT
            .replace("{question}", question)
            .replace("{kb}", kb_text)
            .replace("{answer}", str(answer)))


if __name__ == "__main__":
    qs = load_needs_human()
    print(f"Interpretation/needs-human questions to judge: {len(qs)}")
    print(qs)
    print("Stub only. Wire in an external judge client (not the model under test) to run.")
