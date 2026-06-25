# Evaluation V2 Methods

## Objective

This evaluation compares seven local LLM response sets on 42 Georgia EV supply-chain questions. The task is structured KB question answering over `Data/GNEM_Excel_Data.xlsx`, with human-validated answers from `Data/Human validated questions.xlsx`.

## Experimental Arms

Direct context sends the full KB to the model for each question.

Batchwise/map-reduce splits the KB into batches, extracts candidate rows, verifies them, then produces a final answer from verified candidate evidence.

## Primary Evaluation: Deterministic V2

Deterministic V2 is the primary factual score because the questions often require exact companies, counts, counties, row IDs, and field values.

Main metrics:

- `mean_entity_f1`: F1 over expected companies or counties.
- `count_accuracy_rate`: percentage of count-scored questions where predicted count equals golden count.
- `field_value_accuracy`: whether required KB field values appear in the answer.
- `format_core_ok_rate`: whether required answer sections are present.
- `reliability`: whether the model produced all 42 answers successfully.
- `true_hallucination_answer_rate`: answer-level rate of true out-of-KB company hallucinations.
- `misspelled_kb_company_answer_rate`: answer-level rate of misspelled KB company names.
- `row_id_company_mismatch_rate`: answer-level rate where cited row IDs do not match claimed companies.

Composite formula:

```text
research_score_deterministic =
  0.45 * mean_entity_f1
+ 0.20 * count_accuracy_rate
+ 0.15 * field_value_accuracy
+ 0.10 * (1 - true_hallucination_answer_rate)
+ 0.05 * format_core_ok_rate
+ 0.05 * reliability
```

Count accuracy is intentionally separate from entity F1. A model can get the right count while listing the wrong companies; that is not treated as fully correct.

## Hallucination Fix

The original evaluator scanned whole answers for company-shaped text and produced false positives. V2 fixes this by parsing the Evidence table and prioritizing the `Company` column. It also classifies suspicious mentions as:

- `true_out_of_kb_hallucination`
- `misspelled_kb_company`
- `known_kb_company`
- `industry_group_false_positive`
- `product_service_false_positive`
- `kb_supported_non_company_field`
- `not_a_company_phrase`

This prevents KB field values such as industry groups or product/service text from being counted as hallucinated companies.

## Secondary Evaluation: DeepEval

DeepEval is used as a secondary LLM-as-judge framework with local Ollama model `gpt-oss:120b`.

DeepEval evaluates six rubric dimensions from 0 to 1:

- `factual_faithfulness`: factual claims supported by KB context.
- `completeness`: answer covers the golden-answer content.
- `correctness`: answer agrees with gold and KB.
- `evidence_grounding`: evidence rows and row IDs trace to KB context.
- `company_grounding`: inverse hallucination risk for company names.
- `usefulness`: clarity, directness, and readability.

`mean_deepeval_mean` is the average of these six dimensions.

DeepEval is not the primary score because LLM judges can miss exact list/count errors or be lenient about omissions. It is used to support qualitative interpretation and failure-mode analysis.

## Final Output Files

```text
evaluation_v2/scores_by_model.csv
evaluation_v2/scores_by_answer.csv
evaluation_v2/scores_by_question.csv
evaluation_v2/mention_classification.csv
evaluation_v2/row_id_company_checks.csv
evaluation_v2/deepeval/deepeval_scores_by_model.csv
evaluation_v2/deepeval/deepeval_scores_by_answer.csv
```

