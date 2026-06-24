# Setup Checklist — Direct-Context 42Q Run

**Do not run any model command until Ollama is confirmed working (Step 3) and you tell me to
proceed.** Steps 1–3 are environment setup/verification only. Run them yourself via the `!`
prefix in this session (e.g. `! python3 -m pip install ...`) so the output lands here.

Detected on this machine right now:
- `pandas`, `requests`, `openpyxl` → **NOT installed** (the script needs all three).
- `ollama` → **NOT on PATH** (no daemon detected).

---

## Step 1 — Python dependencies (safe to run)

```bash
python3 -m pip install --user pandas openpyxl requests
```

Verify:

```bash
python3 -c "import pandas, openpyxl, requests; print('deps OK', pandas.__version__)"
```

## Step 2 — Install Ollama (if not already on a GPU host)

If runs happen on another GPU machine, do Steps 2–3 there instead. To install locally:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

## Step 3 — Confirm Ollama is working (VERIFY ONLY — no model pulls yet)

```bash
ollama --version          # binary present?
curl -s http://localhost:11434/api/tags   # daemon responding? (empty list is fine)
```

**STOP here and tell me the output of Step 3.** I will not pull models or run the smoke test
until you confirm Ollama responds.

---

## Step 4 — Pull models (ONLY after you confirm Step 3, and after you finalize models.txt)

`scripts/models.txt` is a **draft** — confirm the final list first. Then:

```bash
# pull each tag in scripts/models.txt
while read -r m; do [ -n "$m" ] && [ "${m#\#}" = "$m" ] && ollama pull "$m"; done < scripts/models.txt
ollama list
```

## Step 5 — Smoke test (1 model, 2 questions) — only after Step 4

```bash
python3 scripts/run_direct_context_42q.py --models "<ONE_MODEL>" --limit 2 --num_ctx 65536
```

Expected: `results/kb_full.jsonl`, a per-model sheet (2 rows) in
`results/model_responses_direct_context_42q.xlsx`, plus
`results/raw_csv/answers_<safe>.csv`, `results/raw_json/answers_<safe>.jsonl`,
`logs/run_<safe>.log`. The run prints KB=205 rows, questions=42, golden=42, and a
`prompt_eval_count` per question (the truncation signal).

## Step 6 — Full run — only after the smoke test looks right

```bash
python3 scripts/run_direct_context_42q.py --models-file scripts/models.txt --limit 42 --num_ctx 65536
```

---

### Notes
- Inputs are read directly from `Data/` and never modified or moved.
- Golden answers are loaded only for the workbook; the script asserts they never enter a prompt.
- `num_ctx=65536`, but the script records the **actual** `prompt_eval_count` per question and
  flags `truncated_context` if a model silently caps lower than the tokens sent.
