# Evaluation Framework — Direct-Context 42-Question Georgia EV Supply-Chain LLM Comparison

_Drafted 2026-06-23. **Planning document only — no evaluation is implemented here.** Do not
build `grade.py` or run metrics from this file. Stage 1 (generation) and Stage 2 (clean output)
come first; this describes Stage 3+ so the response files can be designed to support it._

---

## 1. Purpose

Compare multiple **local** LLMs on the **same** 42-question Georgia EV supply-chain
knowledge-base (KB) task, run under an identical **direct-context** setup (the full 205-row ×
15-column KB is placed in the prompt as JSONL; the model answers using only that), and identify
the **strongest candidate for fine-tuning** (supervised FT / LoRA / QLoRA).

Scope boundary: this is **structured KB question-answering**, not retrieval-augmented
generation. There is no SQL, no vector retrieval, no RAG routing, and no Python-computed
answers in the generation stage. The evaluation must therefore measure the model's own
selection/synthesis ability over a fully-provided KB.

## 2. Research questions

- Which local LLM answers the structured Georgia EV KB questions most **accurately**?
- Which model best follows the **KB-only grounding rule** (no outside knowledge, exact
  "Not found in the data." when absent)?
- Which model handles exact **list / count / multi-condition filter** questions best?
- Which model is most **reliable** under local hardware constraints (no timeouts/crashes)?
- Does **model size / context length** correlate with factual accuracy on this task?
- Did each model actually **ingest the full KB** (context validity), or silently truncate?
- Which model should be selected for **supervised fine-tuning or LoRA**, balancing accuracy,
  reliability, runtime, and trainable size?

## 3. Model comparison dimensions

| Dimension | What it captures |
|---|---|
| Answer accuracy | Correctness vs the 42 human-validated golden answers |
| Grounding faithfulness | Uses only KB; no hallucinated companies/fields/outside facts |
| List completeness | Recall of the correct company set |
| Count correctness | Stated count matches the true count and the listed items |
| Field-level correctness | Right values for requested fields (role, tier, OEM, employment, …) |
| Hallucination rate | Names/values not present in the KB |
| Format compliance | Follows list-then-count, slash-category, "Not found in the data." rules |
| Runtime reliability | Completed answers / 42, error & timeout rate |
| Context validity | Whether the full KB was ingested (`prompt_eval_count` vs `num_ctx`) |
| Latency / runtime | total/eval duration per question and per model |
| Resource fit | Fits/serves on the available local GPU/CPU |
| Suitability for fine-tuning | Trainable size, headroom, instruction-following, schema handling |

## 4. Question-type taxonomy

The 42 questions fall into the types below. (A `question_types.csv` for the new 42q set should
be built during Stage 3 — the existing one in `gold/` is for the old 50q set and must not be
reused.) Each type maps to the metrics that are valid for it:

| Type | Description | Best metrics |
|---|---|---|
| `list` | "Show all X, list their …" | List precision / recall / F1, field accuracy |
| `count` | "How many …" | Count accuracy, signed count error, count==list consistency |
| `lookup` | Single-entity / single-field fact | Exact correctness, field accuracy |
| `multi_filter` | Multiple AND conditions | List P/R/F1, precision-heavy (over-listing is the failure mode) |
| `aggregation/ranking` | Group, rank, max/min, "most/largest" | Count + ranking correctness, field accuracy |
| `risk/interpretation` | "suggesting…", "reflecting…", "emerging trend" | LLM-as-judge + human review (set-matching insufficient) |
| `not-found / insufficiency` | Answer absent from KB | Exact match to "Not found in the data." |

The Georgia KB is **messy** (slash categories like `Tier 1/2`, freeform role values, case
variants, `OEM Footprint` vs `OEM (Footprint)`), so deterministic graders must normalize the
KB side while anchoring company matching to the master KB name list (so hallucinations are
measurable).

## 5. Recommended metrics (for later implementation)

- **Exact correctness** (lookup / not-found)
- **List precision, recall, F1** (set of KB companies named)
- **Count accuracy** and **signed count error** (predicted − true)
- **Missing companies** / **extra companies** (recall and precision error breakdown)
- **Hallucination rate** (named entities not in the KB master list)
- **Field-level accuracy** (requested fields correct)
- **Format compliance** (list-then-count, count==len(list), slash categories not split,
  exact "Not found in the data.")
- **Grounding faithfulness** (no outside knowledge; everything traceable to a KB row)
- **Runtime success rate** (answered / 42), **timeout/error rate**
- **Context validity** (per §"context validity" — silent truncation disqualifies a
  "full-context" claim)
- **Average response time**, **token/runtime efficiency** (eval tokens per second; tokens per
  correct answer)

## 6. Golden-answer use

The 42 entries in `Human validated questions.xlsx` are the **gold standard**. They are
verified joinable to the questions by `question_number` (Num 1..42, 0 positional mismatches).

**Hard rule:** golden answers are used **only during evaluation**, **never** in the generation
prompt. In the response workbook they live in the `golden_answers` sheet and in each model
sheet's `golden_answer` column purely as a reference for later scoring. The generation script
must assert the golden text is not present in any prompt (leakage guard, enforced in the smoke
test).

## 7. Deterministic vs LLM-as-judge (recommended: hybrid)

- **Deterministic grading** for `list`, `count`, `lookup`, `multi_filter`, `aggregation/ranking`,
  and `not-found` — these have defensible reference sets/values; use set P/R/F1, count error,
  exact match.
- **LLM-as-judge** only for `risk/interpretation` questions where exact set-matching is
  insufficient. The judge must be an **external** model, **never the model under test**, with a
  fixed rubric.
- **Human review** for ambiguous / under-specified questions and to spot-check judge verdicts.
- **RAGAS is not recommended** as the primary method: this is structured KB QA, not
  retrieval-based RAG, so RAGAS's retrieval-centric metrics (context precision/recall,
  faithfulness-to-retrieved-chunks) don't fit. Set-overlap + count + faithfulness-to-KB are the
  right primitives.

## 8. Fine-tuning selection criteria

Choose the FT candidate on a weighted combination, **not** raw accuracy alone:

- Highest factual accuracy (F1 / exact correctness)
- High recall **without** excessive hallucination (recall gated by precision)
- Stable local inference (high runtime success, low timeout/error rate)
- Reasonable runtime / latency
- Good instruction following (format compliance)
- Good schema / category handling (slash categories, exact field values)
- **Valid full context** (did not silently truncate the KB)
- Manageable size for LoRA/QLoRA on the available hardware
- **Improvement potential** (headroom: a model with strong grounding but fixable formatting
  errors is a better FT target than a saturated or already-overfit one)

Explicitly: **the best fine-tuning candidate is not necessarily the highest-raw-accuracy
model.** A model that is too large or too slow to train, or one that truncated context, may be
rejected in favor of a slightly less accurate but trainable, stable, full-context model.

## 9. Suggested final tables (for the later results report)

1. **Model setup table** — model, params, quant, `num_ctx`, hardware.
2. **Context validity table** — `prompt_eval_count` min/median/max vs `num_ctx`, classification.
3. **Overall accuracy table** — mean F1 / exact correctness / hallucination across 42.
4. **Question-type accuracy table** — accuracy per taxonomy type (§4).
5. **List/count metrics table** — precision, recall, F1, count error, missing/extra.
6. **Hallucination / error table** — hallucination rate, error count, timeout rate.
7. **Runtime reliability table** — answered/42, mean/median latency, tokens/sec.
8. **Fine-tuning candidate comparison table** — the §8 criteria scored side by side.

## 10. Expected research result structure (future report)

```
Introduction
Dataset and KB description
Question set description
Model setup
Direct-context prompting method
Evaluation methodology
Results
Error analysis
Model selection for fine-tuning
Limitations
Future work
```

## 11. Limitations to track

- Direct-context (full KB in prompt) **does not scale** to larger KBs — works here only because
  205 rows ≈ 38–41K tokens.
- Smaller / lower-real-context models **truncate the KB silently** (observed: `qwen2.5:14b`
  capped at exactly 32768 `prompt_eval_count`); such runs are not valid full-context.
- Models tend to **over-list** companies on multi-condition filters (precision collapse).
- **Counting and aggregation** are intrinsically hard for LLMs; count==list-length consistency
  must be checked explicitly.
- **Runtime instability** on local GPU/CPU (timeouts, OOM) confounds accuracy with reliability.
- Gold answers, though human-validated, may still need review on under-specified questions.
- Direct-context comparison **predicts but does not equal** fine-tuned performance — the chosen
  model's post-FT behavior must be re-measured.

## 12. Future work

- Supervised fine-tuning with synthetic QA pairs generated from the KB.
- LoRA / QLoRA on the selected model.
- Continued pretraining on a company-fact corpus.
- Batched-context v2 (an LLM-only batched scan that lets the model state counts, fixing the
  v1 method where Python computed the count).
- LLM self-verification pass after generation.
- A deterministic verifier applied **after** LLM generation (not as the answerer).
- Compact KB representations to push context limits further.
- Comparison of direct-context vs RAG vs SQL-assisted methods on the same questions.

---

_End of framework. Evaluation implementation is deferred until explicitly requested._
