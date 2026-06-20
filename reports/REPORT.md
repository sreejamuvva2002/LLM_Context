# Evaluation Report — Local LLMs on Georgia EV Supply-Chain KB QA
_Generated 2026-06-18. Companion to AUDIT.md. Numbers come from `grade/grade.py`._

## Research question
Can local LLMs answer structured Georgia EV supply-chain questions using only the
spreadsheet KB, with the LLM (not Python) performing selection and synthesis — and how do
**(a) full-context ingestion** and **(b) exhaustive batched scanning** compare when a model
cannot fit the whole KB in context?

## Method
- **Answer generation (LLM-only):** full-KB-in-context (`run_ollama_jsonl_eval.py`) for
  long-context models; exhaustive batched scan + synthesis (`run_qwen25_batched_eval.py`)
  for Qwen2.5:14B.
- **Grading (Python, separate from generation):**
  - `gold/build_gold.py` derives reference row-sets/counts from the spreadsheet — **29/50
    questions** have a defensible deterministic rule; **21/50** are interpretive/under-specified
    and are flagged `needs_human` (not guessed).
  - `grade/parse_answer.py` extracts, format-agnostically, the **set of KB companies** each
    answer names (anchored to the KB master list, so hallucination is measurable).
  - `grade/grade.py` computes set precision/recall/F1, count error, missing/extra, reliability,
    context validity.
  - `grade/judge_interpretation.py` (stub) — LLM-as-judge rubric for the 21 interpretive
    questions; must use an external judge, never the model under test.

## Headline results (deterministic subset, 29 gradable questions)
| Model | Kind | Answered/50 | Reliability | Mean F1 | Mean recall | Mean precision | Scalar Qs (Q8/Q9) |
|---|---|---|---|---|---|---|---|
| **qwen3.6:35b-a3b** | full-ctx | 50 | 1.00 | **0.77** | 0.83 | 0.78 | 2/2 |
| qwen2.5:14b batched **v1** | batched | 50 | 1.00 | 0.54 | **0.87** | 0.48 | 0/2 |
| deepseek-r1:32b | unstable | 19 | 0.38 | 0.41* | 0.39 | 0.47 | 0/0 |
| gemma3:27b | full-ctx | 49 | 0.98 | 0.31 | 0.41 | 0.28 | 0/2 |
| mistral-small3.2:24b | full-ctx | 50 | 1.00 | 0.29 | 0.40 | 0.28 | 0/2 |
| qwen2.5:14b full-ctx | **truncated** | 50 | 1.00 | 0.20 | 0.21 | 0.20 | 2/2 |

\* DeepSeek F1 is over only 11 gradable answers (rest error/missing) — not comparable.

### How to read this
- **Qwen3.6 (full context) is the clear winner**: F1 0.77, balanced precision/recall, 9/29
  deterministic questions answered *exactly* (F1 = 1.0), both scalar aggregations correct.
- **Mistral and Gemma over-answer badly.** Verified: Mistral printed **96 companies** for Q1
  (true answer = 18 Tier 1/2), dumping most of the KB and ignoring the filter. Their low
  precision (~0.28) is a real behavior, not a parsing artifact. Full context ≠ correct filtering.
- **Batched v1 has the over-selection signature**: highest recall (0.87) but lowest precision
  among completing models (0.48); precision collapses to ~0 on aggregation/multi-filter
  (Q50 0.00, Q49 0.02, Q7 0.04, Q13 0.06). It finds the right rows *and* many wrong ones.
- **Truncated Qwen2.5 full-ctx is worst (F1 0.20)** — direct evidence that exceeding the
  context cap destroys accuracy, which is exactly what motivated batching.

## Error taxonomy
| Class | Where seen | Description |
|---|---|---|
| Context truncation | qwen2.5 full-ctx | prompt capped at 32768; KB tail never seen → low recall everywhere |
| Filter ignored / over-listing | mistral, gemma (full-ctx) | lists large KB swaths regardless of the question's filter → precision collapse |
| Compound-AND collapse | batched v1 | 5-row batches can't hold multi-condition filters → 13 questions select >50 rows |
| No re-verification | batched v1 | set-union + "don't drop rows" locks in false positives |
| Output-schema mismatch | batched v1 | fixed 4 columns ignore fields the question asks for (tier, OEMs, employment) |
| Python-computed count | batched v1 | `REQUIRED_ROW_COUNT` injects the count → Python answers count questions |
| Runtime timeout | gemma Q29 | single 1200s read-timeout; technical, not a wrong answer |
| Instability / non-completion | deepseek | 11 errors + 21 missing on 24 GB GPU |

## Metrics used (and why)
- **List / multi_filter:** set precision, recall, F1 on the KB company set; plus missing/extra.
- **Count / aggregation_rank:** count accuracy + signed error (and F1 on the underlying set).
- **Scalar aggregation (Q7/Q8/Q9):** substring check of the expected county/company value.
- **Cross-cutting:** hallucination (mentions matching no KB company), reliability
  (errors/missing), context validity (prompt_eval_count vs num_ctx).
- **Interpretation (21 qs):** LLM-judge rubric (faithfulness/completeness/correctness +
  hallucinated-company flag) — external judge only.
- **RAGAS: not used as primary.** This is structured KB QA with computable gold sets; exact
  set/count metrics are more rigorous than RAGAS's retrieval-oriented scores and avoid
  reintroducing a retrieval framing. RAGAS `faithfulness` is at most an optional secondary
  cross-check on the interpretation subset.

## Conclusions that ARE valid
1. Exceeding the context cap (Qwen2.5 full-ctx, hard-truncated to 32768) **measurably destroys
   accuracy** (F1 0.20) — the batching motivation is justified.
2. Among long-context models on this KB, **Qwen3.6:35b-a3b is clearly strongest** (F1 0.77).
3. **Larger/full context does not guarantee correct filtering** — Mistral and Gemma ingested
   the whole KB yet over-listed and scored ~0.29.
4. **Batched v1 trades precision for recall** and is reliable only for single-field categorical
   filters; it fails compound-AND, aggregation, and ranking questions.
5. DeepSeek-r1:32b is **not viable on this 24 GB GPU** for this workload (38% completion).

## Conclusions that are NOT valid (yet)
1. ✗ "Batching makes Qwen2.5 competitive with Qwen3.6." The comparison **confounds model and
   method** (different model *and* different pipeline). A clean test needs **Qwen2.5 batched vs
   Qwen2.5 full-context**, or Qwen3.6 batched vs Qwen3.6 full-context.
2. ✗ Any ranking on the **21 interpretive questions** — these are not yet graded (need
   human/LLM-judge).
3. ✗ Treating Gemma's Q29 timeout or DeepSeek's failures as "wrong answers" — they are
   reliability events; keep them out of the accuracy numbers.
4. ✗ Citing the truncated Qwen2.5 full-context file as a full-KB result.

## Future work
- **Better data representation:** canonicalize the messy columns (Category, Industry Group,
  Role) before runs; a clean schema would lift every model's filtering accuracy.
- **Batched v2 (separate experiment, do not overwrite v1):** per-question output schema
  (LLM names needed fields); larger batches (15–25) for compound conditions; an LLM
  re-verification pass over the unioned rows so the *model* prunes false positives; remove the
  Python-computed count. Then grade v1 vs v2 vs full-context.
- **Controlled method test:** run Qwen2.5 **and** Qwen3.6 in *both* full-context and batched
  modes to separate model effect from method effect.
- **Verifier-assisted LLM:** a lightweight LLM check that each emitted row satisfies the stated
  filter (precision guard for the over-listing failure mode).
- **Prompt work:** explicit "filter vs output field" separation and "list only matching rows"
  constraints for the full-context models that over-list (Mistral, Gemma).
- **Fine-tuning / continued pretraining:** only after the prompt/representation fixes; current
  errors look like prompt-following/representation issues, not missing knowledge.

## Recommended next steps (in order)
1. **Hand-validate the 21 `needs_human` gold rows** in `gold/gold_answers.csv` (you said the 50
   are human-validated — this is a short pass), then re-run `grade/grade.py` to cover all 50.
2. **Run the interpretation judge** on those 21 with an external judge model.
3. **Patch Gemma Q29** (one call) and re-grade; optionally resume DeepSeek (likely keep excluded).
4. **Build & run Qwen2.5 batched v2** as a separate experiment; grade v1 vs v2 vs full-context.
5. Present using the two tables above (accuracy + reliability), the error taxonomy, and the
   valid/invalid conclusion lists.

## Reproduce
```bash
python gold/build_gold.py      # writes gold/gold_answers.csv, gold/question_types.csv
python grade/grade.py          # writes reports/scores_by_*.csv and prints the summary
python grade/judge_interpretation.py   # lists the 21 interpretive qs (judge stub)
```
Validity checks baked in: Q1 gold == the 18 hand-verified Tier 1/2 row_ids; batched v1 scores
low precision on the 13 over-selection questions; Qwen2.5 full-ctx flagged truncated (32768).
