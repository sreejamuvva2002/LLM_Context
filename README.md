# Georgia EV Supply-Chain LLM Evaluation

This repository contains a 42-question evaluation of local LLMs on a Georgia EV supply-chain knowledge base. It includes direct-context model responses, batchwise/map-reduce responses, deterministic V2 evaluation, and DeepEval judge scores.

## Final Layout

```text
Data/                 Input KB, questions, and human-validated golden answers
scripts/              Generation and evaluation scripts
results/              Stored model responses from direct-context and batchwise runs
evaluation_v2/        Final research-ready deterministic + DeepEval evaluation
reports/              Final research reports
unused_files/         Archived old reports, legacy evaluation, smoke outputs, caches, and local env files
```

## Inputs

```text
Data/GNEM_Excel_Data.xlsx
Data/questions.txt
Data/Human validated questions.xlsx
```

The KB has 205 rows and 15 source fields. The evaluation uses 42 questions and 42 human-validated golden answers.

## Models

Direct-context models:

```text
qwen3.6:35b-a3b
gemma3:27b
mistral-small3.2:24b
qwen3:30b
deepseek-r1:32b
```

Batchwise/map-reduce models:

```text
gemma3:12b
qwen2.5:14b
```

DeepEval judge:

```text
gpt-oss:120b
```

## Setup

```bash
python3 -m venv .venv_deepeval
.venv_deepeval/bin/pip install -r requirements.txt
```

Ollama must be running locally:

```bash
ollama list
curl -s http://localhost:11434/api/tags
```

Optional DeepEval Ollama config:

```bash
.venv_deepeval/bin/deepeval set-ollama \
  --model=gpt-oss:120b \
  --base-url="http://localhost:11434" \
  --save=dotenv:.env.local
```

## Re-run Final Evaluation

Deterministic V2 evaluation:

```bash
python3 scripts/evaluate_42q_model_responses_v2.py --out evaluation_v2
```

DeepEval judge evaluation:

```bash
.venv_deepeval/bin/python scripts/run_deepeval_42q.py \
  --out evaluation_v2/deepeval \
  --judge-model "gpt-oss:120b" \
  --base-url "http://localhost:11434"
```

The DeepEval script is resumable. It skips rows already present in `evaluation_v2/deepeval/deepeval_raw.jsonl`.

Active scripts:

```text
scripts/run_direct_context_42q.py              Generate direct-context model responses
scripts/run_batchwise_eval.py                  Generate batchwise/map-reduce model responses
scripts/evaluate_42q_model_responses_v2.py     Current deterministic V2 evaluator
scripts/run_deepeval_42q.py                    Current DeepEval judge runner
scripts/eval_common.py                         Shared parsing/loading helpers used by V2/DeepEval
```

## Final Outputs

Primary deterministic scores:

```text
evaluation_v2/scores_by_model.csv
evaluation_v2/scores_by_answer.csv
evaluation_v2/scores_by_question.csv
evaluation_v2/mention_classification.csv
evaluation_v2/row_id_company_checks.csv
evaluation_v2/evaluation_summary_42q.xlsx
```

Secondary DeepEval judge scores:

```text
evaluation_v2/deepeval/deepeval_scores_by_model.csv
evaluation_v2/deepeval/deepeval_scores_by_answer.csv
evaluation_v2/deepeval/deepeval_raw.jsonl
```

Reports:

```text
reports/FINAL_RESEARCH_FINDINGS_42Q.md
reports/EVALUATION_V2_METHODS.md
reports/EVALUATION_V2_AUDIT.md
```

## Final Ranking

The primary ranking uses deterministic V2 research score.

| Rank | Arm | Model | Deterministic Score | DeepEval Mean |
|---:|---|---|---:|---:|
| 1 | direct_context | qwen3.6:35b-a3b | 0.8455 | 0.8186 |
| 2 | direct_context | qwen3:30b | 0.7618 | 0.7381 |
| 3 | batchwise_map_reduce | qwen2.5:14b | 0.5879 | 0.6605 |
| 4 | batchwise_map_reduce | gemma3:12b | 0.5867 | 0.5266 |
| 5 | direct_context | deepseek-r1:32b | 0.5292 | 0.4972 |
| 6 | direct_context | mistral-small3.2:24b | 0.4609 | 0.4460 |
| 7 | direct_context | gemma3:27b | 0.4349 | 0.3012 |

## Notes

- Deterministic V2 scores are the primary factual evaluation.
- DeepEval scores are secondary semantic/rubric evaluation.
- The old V1 evaluation and old judge artifacts were archived under `unused_files/`.
- Local virtual environment files were also archived; recreate from `requirements.txt` when needed.
