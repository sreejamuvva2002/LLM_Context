# Evaluation V2 Audit

## Repo Cleanup

The repo was reorganized so final materials are easy to find:

```text
Data/
scripts/
results/
evaluation_v2/
reports/
unused_files/
```

Archived under `unused_files/cleanup_20260625_final_research/`:

- old reports and Claude prompts
- legacy V1 deterministic evaluation outputs
- old non-DeepEval judge outputs
- smoke-test outputs
- Python cache files
- previous local DeepEval virtual environment and `.deepeval` config
- obsolete `scripts/run_judge_42q.py`
- old standalone evaluator script, now replaced by V2

The reusable loading/parsing helpers from the old evaluator were kept under `scripts/eval_common.py`. The old standalone evaluator filename was removed from active `scripts/`.

## Completeness Checks

Final deterministic V2 outputs:

- `evaluation_v2/scores_by_answer.csv`: 294 rows, 42 questions for each of 7 models.
- `evaluation_v2/scores_by_model.csv`: 7 model rows.
- `evaluation_v2/mention_classification.csv`: 1,791 classified mentions.
- `evaluation_v2/row_id_company_checks.csv`: 1,829 row-id/company checks.

Final DeepEval outputs:

- `evaluation_v2/deepeval/deepeval_raw.jsonl`: 294 rows.
- `evaluation_v2/deepeval/deepeval_scores_by_answer.csv`: 294 rows.
- `evaluation_v2/deepeval/deepeval_scores_by_model.csv`: 7 model rows.

No DeepEval partial errors were reported.

## Prompt Leakage

Direct-context generation contains a leakage guard that raises if a golden answer appears in a prompt. Batchwise generation does not read the golden-answer file. Golden answers are used only during evaluation and workbook/report construction.

## Known V1 Issue Fixed

The legacy `evaluation/unknown_company_mentions.csv` had false positives because it scanned the whole answer. Examples included:

- `Jackson America`, which appeared in a Product/Service field.
- `Primary Metal Industries`, an Industry Group value.
- `Miscellaneous Manufacturing Industries`, an Industry Group value.

V2 keeps hallucination scoring focused on company evidence and labels non-company KB field values separately.

## V2 Mention Classification Summary

```text
known_kb_company:                  1768
misspelled_kb_company:               13
true_out_of_kb_hallucination:         5
not_a_company_phrase:                 3
kb_supported_non_company_field:       1
product_service_false_positive:       1
```

True hallucinations were concentrated mainly in `gemma3:27b`; misspellings appeared in several models.

## Row ID / Company Consistency

V2 checks whether the cited row ID belongs to the company claimed in the Evidence table.

| Model | Checked | Mismatches |
|---|---:|---:|
| qwen3.6:35b-a3b | 334 | 0 |
| qwen3:30b | 190 | 2 |
| qwen2.5:14b | 105 | 0 |
| gemma3:12b | 408 | 35 |
| deepseek-r1:32b | 219 | 35 |
| mistral-small3.2:24b | 184 | 16 |
| gemma3:27b | 389 | 121 |

This is a useful failure-mode metric but is not currently part of the composite score.

## Caveats

- Deterministic entity extraction depends on exact/alias matching and may still miss some semantically equivalent wording.
- DeepEval is judge-based and should not override deterministic entity/count errors.
- Composite-score weights are methodological choices and should be reported explicitly.
- Manual validation is still recommended for the highest-stakes examples and for publication tables.
