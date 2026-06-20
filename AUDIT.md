# Experiment Audit — Georgia EV Supply-Chain Local-LLM Evaluation
_Audited 2026-06-18. All findings verified directly against the files, not the run notes._

## Goal recap
Test whether local LLMs can answer 50 human-validated Georgia EV supply-chain questions
using only the spreadsheet KB (205 rows × 15 cols). The LLM must do the selection and
synthesis; Python is orchestration only. Grading (this document's harness) is separate and
is allowed to use Python.

## Dataset (verified)
- `GNEM_Excel_Data.xlsx`: 205 rows × 15 cols, single sheet `Sheet1`. ✓
- `questions.txt`: exactly 50 questions. ✓
- `kb_full.jsonl`: 205 rows, all 15 columns + `row_id` (= dataframe index + 1). ✓
- Data is **messy** and this affects grading: `Category` has both `OEM Footprint` and
  `OEM (Footprint)`; `Industry Group` values carry leading spaces and ~30 one-off freeform
  strings; `EV Supply Chain Role` has 22 freeform one-off values beyond the clean set;
  `Primary Facility Type` has case variants (`Manufacturing Plant` vs `Manufacturing plant`).
  The gold builder normalizes for these; some questions are genuinely under-specified
  (see "needs human" below).

## Per-file state and classification
`prompt_eval_count` = tokens the model actually ingested; it is the truncation test.

| File | rows | errors | prompt_eval (min/med/max) | num_ctx | Classification |
|---|---|---|---|---|---|
| answers_qwen3.6_35b-a3b.csv | 50 | 0 | 39565 / 39574 / 39592 | 65536 | **1. Valid main result** |
| answers_mistral-small3.2_24b.csv | 50 | 0 | 41322 / 41332 / 41350 | 65536 | **1. Valid main result** |
| answers_gemma3_27b.csv | 50 | **1 (Q29)** | 38398 / 38407 / 38424 | 65536 | **4. Needs patch** — 49/50 valid; Q29 is a 1200s read-timeout, not a wrong answer |
| answers_qwen2.5_14b_batched_strict.csv | 50 | 0 | per-batch (small) | 32768/batch | **1. Valid main result (batched v1)** — completes, but structurally unreliable on complex questions (Part B) |
| answers_qwen2.5_14b.csv | 50 | 0 | **32768 / 32768 / 32768** | 65536 | **2. Truncated baseline** — hard 32768 cap; never ingested the KB tail |
| answers_deepseek-r1_32b.csv | 29 | **11** | 39334 / 39339 / 39353 | 65536 | **3. Unstable / diagnostic + 5. Exclude from accuracy** — only 18/50 valid |

### Key confirmations
- **Qwen2.5 full-context is conclusively truncated.** `prompt_eval_count` is *exactly*
  32768 on all 50 rows while sibling models ingested 38–41K. It silently dropped ~6–9K
  tokens (the tail rows of the JSONL). It is a truncated baseline, never a valid full-KB run.
- **Gemma Q29 was never actually patched.** Both `answers_gemma3_27b.csv` **and** the
  misnamed `BACKUP_AFTER_PATCH_answers_gemma3_27b.csv` still contain the Q29 read-timeout.
  Treat Q29 as a runtime failure (reliability), grade Gemma on 49/50 for accuracy.
- **DeepSeek resume was never run.** Still 29 rows; 11 errors on Q5,7,8,9,12,13,18,22,23,27,28;
  Q30–Q50 missing → 18/50 valid answers. It accepted 65K context but is too slow/unstable on
  the 24 GB A5000 (repeated 1200s timeouts). Diagnostic only.
- **No golden-answer file existed** anywhere in the project — grading had nothing to score
  against. This audit creates one (`gold/gold_answers.csv`).

## Part B — Qwen2.5 batched method (v1): structural analysis
Source: `run_qwen25_batched_eval.py`. Pipeline: split 205 rows into batches of 5 → LLM
returns matching row_ids per batch → Python unions/dedupes the ids → LLM transcribes the
selected rows into a fixed table.

### Did Python become the answerer?
- **Row selection: the LLM does it.** ✓ (`batch_scan` parses `matching/matched_row_ids`).
- **But Python computes `REQUIRED_ROW_COUNT = len(selected_rows)`** and injects it into the
  final prompt (`synthesize_answer`). For a **list** question this is benign metadata. For a
  **count/aggregation** question ("how many…"), the count *is* the answer, so Python is
  computing the answer for that question type — a partial violation of the professor's
  constraint. v2 must let the LLM state the count.
- The final prompt is a **pure transcription task with a hard-coded 4-column schema**
  (`row_id | Company | EV Supply Chain Role | Product / Service`) that **forbids re-filtering**.

### Failure modes (verified from the actual v1 output + grading)
1. **Compound-AND collapse / runaway over-selection.** With only 5 rows of local context the
   model cannot hold multi-condition filters and defaults to including rows. **13 of 50
   questions selected >50 of 205 rows** (Q9=203, Q46=143, Q45=106, Q48=102, Q29=96, Q13=76…);
   Q18 selected 0. Grading confirms the signature: batched v1 has **high recall (0.87) but low
   precision (0.48)** across deterministic questions, with precision near zero on
   aggregation/multi-filter (Q50=0.00, Q49=0.02, Q7=0.04, Q13=0.06).
2. **No re-verification = false positives are permanent.** Python takes the set union and the
   final prompt forbids dropping rows, so a single bad batch selection is locked into the answer.
3. **Fixed output schema ignores the question.** Q2 asks "what tier" but the table shows
   EV Role/Product, not Category/tier; questions about Employment, Primary OEMs, or mapping get
   the same 4 columns — structurally wrong fields even when selection is right.
4. **"50/50, 0 errors" is completion, not correctness.** Zero runtime errors; ~13–14 answers are
   content-wrong. The earlier "Q1 passes all checks" was a best case, not representative.

### Verdict
v1 is valid for **single-field categorical filters** (Q1 Tier 1/2, Q2 battery roles) and fails
on multi-condition / aggregation / ranking / interpretation questions. **Keep v1 as an honest
baseline; do not overwrite it.** A redesigned v2 is specified in the plan and in REPORT.md.

## Backups taken (2026-06-18)
`backups/2026-06-18/` contains copies of all 6 `answers_*.csv`,
`audit_qwen2.5_14b_batched_strict.jsonl`, and `kb_full.jsonl`. **Note:** the pre-existing
`ollama_outputs/BACKUP_AFTER_PATCH_answers_gemma3_27b.csv` is misnamed — it still contains the
Q29 error and is *not* a fixed copy. Rename/relabel it to avoid confusion.
