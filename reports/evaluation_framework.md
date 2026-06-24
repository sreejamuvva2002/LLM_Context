# Evaluation Framework (Design Only — Not Yet Implemented)

This document defines **how** the direct-context 42Q outputs will be scored and
**how** that score selects a model for later LoRA/QLoRA fine-tuning. No scoring
code is written yet. Golden answers live only in the evaluation layer and are
**never** injected into model prompts.

---

## 1. Purpose

Two distinct decisions, do not conflate them:

1. **Direct-context ranking** — which model answers best when handed the full KB
   in context. This is the headline research comparison (local vs. frontier).
2. **Fine-tuning candidate selection** — which model to take forward for a
   domain-specific local chatbot. Direct-context accuracy is an input here, but
   **not the only one** (see §4). A model can win on direct-context accuracy yet
   be a poor FT choice (e.g., 70B/235B, or a reasoning model).

> Note carried from project history: for **exact factual recall and aggregation
> over a structured KB, fine-tuning underperforms in-context injection.** FT is
> appropriate for domain *style/format/behavior* adaptation, not for memorizing
> the KB. Whether FT is the right end-goal at all is a question for the professor;
> this framework selects the *best FT base* conditional on that decision.

---

## 2. Inputs

- Per-model run files: `{model}_42q_responses.jsonl` (one record per question:
  `q_id`, `question`, `raw_response`, `clean_response` with `<think>` stripped,
  `prompt_eval_count`, `eval_count`, latency, `num_ctx`, status).
- `golden_answers_42q.*` (held separately; evaluation-side only).
- KB integrity flags (for grounding: correct answer = correct query over the KB
  as written, not real-world truth).

---

## 3. Metrics (to implement later)

| Metric | Definition | Applies to |
|---|---|---|
| Exact correctness | Normalized exact match to golden answer | scalar/count/single-entity answers |
| List precision | \|pred ∩ gold\| / \|pred\| | list answers |
| List recall | \|pred ∩ gold\| / \|gold\| | list answers |
| List F1 | harmonic mean of precision/recall | list answers |
| Count accuracy | exact integer match; report signed error too | count answers |
| Field-level accuracy | per-field correctness (company, county, employment, tier, OEM, industry) | structured answers |
| Hallucination rate | fraction of answers asserting entities/values absent from KB | all |
| Format compliance | adherence to requested output shape (list vs. number vs. sentence) | all |
| Runtime reliability | completion without timeout/crash/empty output; tok/s | all |
| Context validity | **gate**, pass/fail: did `prompt_eval_count` show full KB ingested (no truncation)? | all |
| Fine-tuning feasibility score | composite: model size, dense vs. MoE, PEFT/Unsloth support, template stability, single-GPU QLoRA fit | per model |

Scoring rules (to match prior project decisions):
- **Deterministic grading, not LLM self-grading.** Exact match for counts;
  set comparison (P/R/F1) for lists. LLM-as-judge only as a secondary,
  clearly-labeled cross-check, never the primary number.
- Normalize before matching: case, whitespace, company-name aliases
  (`Company` vs `Company_Clean`), county suffixes.
- **Context validity is a hard gate.** Any question whose run shows truncation is
  scored invalid for that model, not silently counted as wrong.

---

## 4. Model-selection weighting (direct-context ranking)

Proposed weights (endorsed, with definitions above):

| Weight | Component |
|---|---|
| 40% | Factual accuracy (exact correctness + field-level accuracy) |
| 20% | List/count reliability (F1 + count accuracy) |
| 15% | Hallucination control (1 − hallucination rate) |
| 10% | Runtime stability |
| 10% | Fine-tuning feasibility |
| 5% | Format compliance |

Gate (not weighted): **Context validity must PASS**, else the model is
disqualified from ranking on the affected questions.

---

## 5. From ranking to fine-tuning pick

The FT pick is **not** `argmax(direct_context_score)`. Apply, in order:

1. **Gate:** context validity PASS on ≥ ~95% of questions.
2. **Quality floor:** factual accuracy + hallucination control above a threshold
   (set after seeing the score distribution).
3. **FT practicality filter:** dense, ≤ ~14B, single-GPU QLoRA-feasible, mature
   PEFT/Unsloth support, stable chat template. This is where 70B/235B/MoE and
   reasoning models fall out even if accurate.
4. Among survivors, rank by the weighted score in §4 with **fine-tuning
   feasibility upweighted** for this specific decision.

Expected practical outcome: the FT base is most likely one of
`qwen3:14b`, `qwen2.5:14b`, `mistral-nemo:12b`, `gemma3:12b`, with the larger
Track-A models serving as the in-context quality ceiling the FT'd model is
measured against.

---

## 6. Out of scope for now

No scoring implementation, no judge prompts, no threshold values fixed. Those are
set in a later, separate evaluation pass once direct-context responses exist.
