# Final Research Findings: Georgia EV Supply-Chain 42Q LLM Evaluation

## Executive Summary

This study evaluated seven local LLM response sets on 42 Georgia EV supply-chain questions over a 205-row knowledge base. The strongest model was `qwen3.6:35b-a3b`, ranking first in both deterministic V2 scoring and DeepEval semantic judging.

The main result is clear: the best direct-context models outperformed the batchwise models, but the batchwise 12B/14B models outperformed several larger direct-context models. Exact entity/list correctness and count accuracy remained the hardest part of the task.

Primary ranking uses deterministic V2 score. DeepEval is reported as a secondary judge score.

## Experiment Setup

Inputs:

```text
Data/GNEM_Excel_Data.xlsx
Data/questions.txt
Data/Human validated questions.xlsx
```

Question set:

- 42 total questions
- 11 judge-recommended / higher-interpretation questions
- Use-case areas:
  - Supply Chain Mapping & Visibility
  - Product & Technology Trends
  - Supplier Discovery & Matchmaking
  - Supply Chain Risk & Resilience
  - Site Selection & Expansion Planning

Models:

| Arm | Models |
|---|---|
| Direct context | `qwen3.6:35b-a3b`, `gemma3:27b`, `mistral-small3.2:24b`, `qwen3:30b`, `deepseek-r1:32b` |
| Batchwise/map-reduce | `gemma3:12b`, `qwen2.5:14b` |
| DeepEval judge | `gpt-oss:120b` |

## Evaluation Design

Two evaluation types were used.

Deterministic V2 evaluation is the primary score because this task requires exact factual matching: company names, counties, counts, row IDs, tiers, roles, OEM strings, and field values.

DeepEval is the secondary evaluation. It uses `gpt-oss:120b` as a local judge to rate semantic/rubric dimensions such as faithfulness, completeness, correctness, grounding, and usefulness.

## Deterministic Metrics

Primary deterministic metrics:

- `mean_entity_f1`: exact entity F1 over expected companies/counties.
- `count_accuracy_rate`: whether predicted count equals golden count.
- `field_value_accuracy`: whether required KB field values appear in the response.
- `true_hallucination_answer_rate`: true out-of-KB company hallucination rate.
- `format_core_ok_rate`: required answer structure present.
- `reliability`: model produced all answers.

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

Important interpretation: count accuracy alone is not enough. If a model lists one wrong company and omits one right company, the count can still match while entity F1 correctly penalizes the answer.

## DeepEval Metrics

DeepEval scored each answer on six 0-1 dimensions:

- `factual_faithfulness`: all factual claims are supported by KB context.
- `completeness`: answer covers the expected/golden content.
- `correctness`: answer agrees with the KB and gold answer.
- `evidence_grounding`: evidence rows and row IDs are traceable.
- `company_grounding`: company names are grounded in the KB.
- `usefulness`: answer is clear and readable.

`mean_deepeval_mean` is the average of these six scores.

## Final Model Scores

| Rank | Arm | Model | Deterministic Score | Entity F1 | Count Accuracy | Field Accuracy | True Hallucination Rate | Row-ID Company Mismatch Rate | DeepEval Mean |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | direct_context | qwen3.6:35b-a3b | 0.8455 | 0.8131 | 0.7778 | 0.8272 | 0.0000 | 0.0000 | 0.8186 |
| 2 | direct_context | qwen3:30b | 0.7618 | 0.7659 | 0.5000 | 0.7811 | 0.0000 | 0.0476 | 0.7381 |
| 3 | batchwise_map_reduce | qwen2.5:14b | 0.5879 | 0.5287 | 0.2222 | 0.7038 | 0.0000 | 0.0000 | 0.6605 |
| 4 | batchwise_map_reduce | gemma3:12b | 0.5867 | 0.5039 | 0.2222 | 0.7703 | 0.0000 | 0.2381 | 0.5266 |
| 5 | direct_context | deepseek-r1:32b | 0.5292 | 0.4420 | 0.1944 | 0.6252 | 0.0238 | 0.2619 | 0.4972 |
| 6 | direct_context | mistral-small3.2:24b | 0.4609 | 0.3193 | 0.1944 | 0.5221 | 0.0000 | 0.2857 | 0.4460 |
| 7 | direct_context | gemma3:27b | 0.4349 | 0.2627 | 0.1111 | 0.6770 | 0.0714 | 0.7381 | 0.3012 |

## DeepEval Score Breakdown

| Rank | Model | Faithfulness | Completeness | Correctness | Evidence Grounding | Company Grounding | Usefulness | DeepEval Mean |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | qwen3.6:35b-a3b | 0.8095 | 0.5714 | 0.6952 | 0.9500 | 0.9714 | 0.9143 | 0.8186 |
| 2 | qwen3:30b | 0.7357 | 0.3357 | 0.4857 | 0.9405 | 0.9929 | 0.9381 | 0.7381 |
| 3 | qwen2.5:14b | 0.6226 | 0.2429 | 0.3071 | 0.9119 | 0.9524 | 0.9262 | 0.6605 |
| 4 | gemma3:12b | 0.4357 | 0.1643 | 0.2190 | 0.6690 | 0.9381 | 0.7333 | 0.5266 |
| 5 | deepseek-r1:32b | 0.3286 | 0.1238 | 0.1452 | 0.6262 | 0.9071 | 0.8524 | 0.4972 |
| 6 | mistral-small3.2:24b | 0.2667 | 0.0786 | 0.1000 | 0.5000 | 0.9810 | 0.7500 | 0.4460 |
| 7 | gemma3:27b | 0.1083 | 0.0071 | 0.0429 | 0.2595 | 0.7607 | 0.6286 | 0.3012 |

## Key Findings

`qwen3.6:35b-a3b` is the best overall model. It has the highest deterministic score, highest DeepEval mean, strongest count accuracy, high entity F1, zero true hallucination rate, and zero row-id/company mismatches.

`qwen3:30b` is the second-best model. It is close to qwen3.6 on entity F1 and DeepEval grounding, but weaker on count accuracy and some completeness/correctness dimensions.

The batchwise models are competitive despite being smaller. `qwen2.5:14b` and `gemma3:12b` rank third and fourth by deterministic score and outperform three larger direct-context models. This suggests that batching and verification can help smaller models produce more controlled outputs, especially on field lookup and grounding.

Batchwise still struggles with exact counts. Both batchwise models have `count_accuracy_rate = 0.2222`, far below `qwen3.6:35b-a3b` at `0.7778`.

The weakest models fail primarily through omission, wrong entity selection, and row-id/company mismatch. They often produce readable answers, so usefulness alone is not a reliable indicator of factual correctness.

## Direct Context vs Batchwise

Average deterministic scores:

| Arm | Mean Deterministic Score | Mean Entity F1 | Mean Count Accuracy | Mean Field Accuracy |
|---|---:|---:|---:|---:|
| direct_context | 0.6065 | 0.5206 | 0.3555 | 0.6865 |
| batchwise_map_reduce | 0.5873 | 0.5163 | 0.2222 | 0.7370 |

Average DeepEval scores:

| Arm | Mean DeepEval | Faithfulness | Completeness | Correctness | Evidence Grounding |
|---|---:|---:|---:|---:|---:|
| direct_context | 0.5602 | 0.4498 | 0.2233 | 0.2938 | 0.6552 |
| batchwise_map_reduce | 0.5936 | 0.5292 | 0.2036 | 0.2630 | 0.7905 |

Interpretation: direct context produces the top two individual models, but batchwise improves grounding for smaller models. Batchwise loses more on exact counts and global aggregation, likely because the final stage sees verified candidate rows rather than the full KB.

## Failure Modes

The most important failure modes were:

- missing correct companies
- adding extra wrong companies
- correct count but wrong entity set
- row ID assigned to the wrong company
- misspelled KB company names
- aggregation/ranking mistakes
- batchwise candidate loss
- low completeness despite high readability

Hardest questions by mean entity F1 were Q30, Q41, Q39, Q7, Q8, Q32, Q25, Q23, Q26, and Q9. These are mostly aggregation, ranking, risk, or exact-selection questions.

Easiest questions by mean entity F1 included Q6 and Q11, both reaching 1.0000 across models.

## Hallucination And Mention Audit

V2 mention classification found:

```text
known_kb_company:                  1768
misspelled_kb_company:               13
true_out_of_kb_hallucination:         5
not_a_company_phrase:                 3
kb_supported_non_company_field:       1
product_service_false_positive:       1
```

This confirms the old unknown-company metric was too broad. The corrected true hallucination rate is lower and more defensible.

True hallucination answer rates:

- `qwen3.6:35b-a3b`: 0.0000
- `qwen3:30b`: 0.0000
- `qwen2.5:14b`: 0.0000
- `gemma3:12b`: 0.0000
- `deepseek-r1:32b`: 0.0238
- `mistral-small3.2:24b`: 0.0000
- `gemma3:27b`: 0.0714

## Research-Ready Claims

The following claims are supported by the final outputs:

- `qwen3.6:35b-a3b` is the strongest model in this experiment.
- `qwen3:30b` is the second strongest model.
- Smaller batchwise models are competitive and outperform several larger direct-context models.
- Exact count and entity-list accuracy remain difficult, especially for aggregation/ranking questions.
- DeepEval scores broadly agree with deterministic ranking, but deterministic metrics are more appropriate as the primary factual score.
- V2 hallucination detection is more reliable than the previous unknown-company metric because it separates true hallucinations from KB-supported non-company field values.

## Claims To Avoid

Do not claim that DeepEval alone proves factual correctness. It is useful, but deterministic entity/count/evidence checks remain primary.

Do not use the old `evaluation/unknown_company_mentions.csv` hallucination rates as final. Those legacy files are archived.

Do not treat correct count as correct answer unless entity F1 and evidence checks also support it.

Do not claim batchwise is categorically better or worse than direct context. The best individual models are direct-context, but batchwise helps smaller models remain competitive.

## Limitations

The deterministic evaluator depends on the human-validated golden answers and exact/alias entity matching. Some nuanced semantic equivalence may still require manual review.

DeepEval is judge-based and can be influenced by rubric wording and context selection. It should be used as secondary evidence, not as a replacement for exact scoring.

The composite score weights are research-design choices. They are transparent and reasonable for this task, but other studies may choose different weights.

Manual validation is recommended for the final publication examples, especially for low-scoring questions and model comparisons near decision boundaries.

## Final Conclusion

The cleaned V2 evaluation supports a defensible research finding: `qwen3.6:35b-a3b` is the most reliable local model for this Georgia EV supply-chain QA task, followed by `qwen3:30b`. Batchwise evaluation improves the competitiveness of smaller models, but exact count and aggregation questions remain hard. The final score tables should report deterministic V2 metrics as primary and DeepEval judge metrics as secondary.

