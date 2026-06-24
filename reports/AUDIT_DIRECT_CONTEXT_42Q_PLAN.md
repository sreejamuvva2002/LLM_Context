# AUDIT — Direct-Context 42-Question Georgia EV Supply-Chain LLM Run

_Audited 2026-06-23. Every fact below was verified directly against the files in this repo
(stdlib zip/XML inspection — pandas/openpyxl are not installed), not from prior run notes._

**Status: AUDIT ONLY. No files moved, no scripts edited, no models run. Awaiting your approval.**

---

## 0. Headline findings (read these first)

1. ✅ **`Data/questions.txt` has exactly 42 questions.** (The naive `wc -l` shows 41 only
   because the final line has no trailing newline; `awk 'NF'` confirms 42 non-blank lines,
   no blank lines, no duplicates.)
2. ✅ **`Data/Human validated questions.xlsx` has exactly 42 answer rows** (`Num` 1..42) and
   they align **positionally 1:1** with `questions.txt` — 0 mismatches. Golden answers map
   cleanly by `question_number`.
3. ✅ **KB `Data/GNEM_Excel_Data.xlsx` = 205 data rows × 15 columns**, single sheet `Sheet1`.
4. ⚠️ **There are TWO copies of the KB with different content.** Top-level
   `GNEM_Excel_Data.xlsx` (md5 `924d9f0d…`) ≠ `Data/GNEM_Excel_Data.xlsx` (md5 `801cba83…`).
   Both are 205×15. **The `Data/` copy is the final one per your instructions** — the
   top-level one is stale and must not be used.
5. ⛔ **HARD BLOCKER for running models:** `pandas`, `requests`, `openpyxl` are **not
   installed**, and **`ollama` is not on PATH**. Nothing can run until these are resolved.
   This does not affect the audit/framework deliverables.
6. ⚠️ **Filenames differ from your spec.** Folder is `Data/` (capital D), not `data/`; golden
   file is `Human validated questions.xlsx`, not `human_validated_answers.xlsx`. The new
   script must take these as parameters, not hardcode the spec names.
7. ⚠️ **All existing gold/grade/report artifacts are from the OLD 50-question experiment**
   and must not be reused as golden truth for the 42q run.

---

## 1. Current repo state

Git: branch `main`, one commit (`1f5301b Initial commit`). Working tree shows
`questions.txt` **deleted** at top level (tracked) and `Data/` **untracked**.

```
LLM_Context/
  AUDIT.md                         # OLD 50q audit (2026-06-18)
  Claude chat.txt                  # 63 KB chat log (not inspected in full)
  GNEM_Excel_Data.xlsx             # OLD/stale KB copy (md5 924d9f0d…) — differs from Data/
  resume_deepseek_missing.py       # OLD 50q helper
  retry_gemma_q29.py               # OLD 50q helper
  run_ollama_jsonl_eval.py         # OLD 50q DIRECT-CONTEXT runner  ← reusable base
  run_qwen25_batched_eval.py       # OLD 50q batched method (violates "no Python answering")
  run_qwen25_batched_eval_old.py   # OLD 50q batched (superseded)
  __pycache__/                     # compiled OLD scripts
  Data/                            # NEW final inputs (untracked)
    GNEM_Excel_Data.xlsx           #   final KB (md5 801cba83…)
    Human validated questions.xlsx #   final golden answers (42 rows)
    questions.txt                  #   final 42 questions
    .DS_Store                      #   macOS cruft (harmless)
  gold/                            # OLD 50q derived gold
    build_gold.py, gold_answers.csv (50), question_types.csv (50)
  grade/                           # OLD 50q grading harness
    grade.py, parse_answer.py, judge_interpretation.py, __pycache__/
  ollama_outputs/                  # OLD 50q model outputs (6 models) + kb_full.jsonl + BACKUP_* files
  reports/                         # OLD 50q REPORT.md + scores_by_model.csv + scores_by_question.csv
  backups/2026-06-18/              # OLD 50q backup snapshot (8 files)
```

## 2. Detected data files (in `Data/`)

| File | Type | What it is |
|---|---|---|
| `GNEM_Excel_Data.xlsx` | KB | Georgia EV supply-chain KB, `Sheet1`, 205 rows × 15 cols. md5 `801cba83…` |
| `Human validated questions.xlsx` | Golden | `Sheet1`, 42 answer rows (`Num` 1..42). md5 not pinned; structure verified. |
| `questions.txt` | Questions | 42 questions, one per line (no trailing newline on last line) |
| `.DS_Store` | cruft | macOS metadata, ignore |

## 3. Confirmation — `questions.txt` has exactly 42 questions

- `awk 'NF'` non-blank line count = **42**.
- `grep -nE '^[[:space:]]*$'` (blank lines) = **none**.
- Duplicate check (`sort | uniq -d`) = **none**.
- Q1 = `Show all "Tier 1/2" suppliers in Georgia, list their EV Supply Chain Role and Product / Service.`
- Q42 = `Which Georgia areas have R&D facility types in the automotive sector, suggesting innovation infrastructure suitable for EV technology development centers?`
- Caveat noted: `wc -l` = 41 (missing trailing newline). The runner must use a
  newline-split-then-strip read (the old `read_questions()` already does this correctly).

## 4. Confirmation — golden answers match the 42 questions

- `Human validated questions.xlsx` → 42 data rows, `Num` = 1..42 (contiguous).
- Positional comparison of the golden `Question` column vs `questions.txt`: **0 mismatches**
  across all 42. Safe to join on `question_number`.

## 5. KB Excel — row count and columns

`Data/GNEM_Excel_Data.xlsx`, `Sheet1`, **205 data rows** (+1 header), **15 columns**:

```
Company, Category, Industry Group, Location, Address, Latitude, Longitude,
Primary Facility Type, EV Supply Chain Role, Primary OEMs,
Supplier or Affiliation Type, Employment, Product / Service,
EV / Battery Relevant, Classification Method
```

First data row: `JTEKT North America Corp.`; last: `ZF Gainesville LLC`.
Note: the sheet's stored dimension spans ~998 rows, but only 205 contain a `Company` value —
the rest are empty trailing rows. The JSONL builder must drop fully-empty rows (the old
`excel_to_jsonl()` already does `dropna(how="all")`).

**Known data messiness (from old AUDIT.md, still relevant for evaluation, NOT for generation):**
`Category` has both `OEM Footprint` and `OEM (Footprint)`; `Industry Group` has leading
spaces and freeform one-offs; `EV Supply Chain Role` has freeform values beyond the clean
vocabulary; `Primary Facility Type` has case variants. This affects later grading, not the
direct-context prompt (the model gets the raw values).

## 6. Golden-answer Excel — columns

`Human validated questions.xlsx`, `Sheet1`. Header row:

```
['Num', 'Use Case Category', 'Question', 'Human validated answers', '', '']
```

→ **4 meaningful columns** (`Num`, `Use Case Category`, `Question`, `Human validated answers`)
plus **2 trailing empty-named header cells** that the writer must drop/ignore to avoid
blank/duplicate column names. Answers are long multi-line strings (company-per-line format
with `[tier] | Role: … | Product: …`).

## 7. Old files that should be moved to backup (NOT deleted)

All of these are 50-question-era artifacts. **Proposed action: move to a dated backup folder
after your approval. Nothing deleted.**

| Item | Reason |
|---|---|
| `GNEM_Excel_Data.xlsx` (top-level) | Stale KB; differs from `Data/` copy; risks being read by old scripts |
| `AUDIT.md` | 50q audit (kept for provenance) |
| `gold/` (gold_answers.csv, question_types.csv, build_gold.py) | 50q derived gold — does NOT match 42q golden |
| `grade/` | 50q grading harness (Stage-3+ only; not needed now) |
| `reports/REPORT.md`, `reports/scores_*.csv` | 50q results |
| `ollama_outputs/` (incl. `BACKUP_*` and `kb_full.jsonl`) | 50q model outputs |
| `run_qwen25_batched_eval.py`, `run_qwen25_batched_eval_old.py`, `resume_deepseek_missing.py`, `retry_gemma_q29.py` | 50q one-offs; batched method partially violates "no Python answering" |
| `__pycache__/` | compiled old scripts |
| `Claude chat.txt` | old chat log |

Proposed destination: `backups/old_50q_experiment_20260623_HHMMSS/` (timestamp filled at move time).
**I will show you the exact `mv` list and wait for approval before moving anything.**

## 8. Existing scripts that can be reused

- **`run_ollama_jsonl_eval.py` — reuse as the base.** It already implements the exact
  direct-context method you want:
  - `excel_to_jsonl()` — reads Excel, drops empty rows, adds `row_id`, emits JSONL. ✅
  - `read_questions()` — strip + skip-blank line reader (handles the no-trailing-newline case). ✅
  - `build_prompt()` — concatenates grounding rules + data dictionary + full JSONL KB + question. ✅
  - `call_ollama()` — `/api/generate`, `temperature=0`, `seed=42`, `num_ctx` option, captures
    `prompt_eval_count`, `eval_count`, durations. ✅
  - `ANSWER_PROMPT` + `DATA_DICTIONARY` — already encode "use only the KB", "Not found in the
    data.", list-then-count, count==len(list), AND-filters, and the **Tier 1/2 / Tier 2/3
    "do not split slash categories" rules**. ✅ → **reuse these definitions verbatim; do not
    reinvent.** (This satisfies your "reuse methodology if it exists" instruction.)

## 9. Scripts that must be changed / created

**Create a new script `scripts/run_direct_context_42q.py`** (do not overwrite the old runner).
It reuses the functions above but adds everything the old CSV-only runner lacks:

| Gap in old runner | Required for 42q run |
|---|---|
| Hardcoded top-level `XLSX_FILE` / `questions.txt` | Configurable paths → default to `data/` (the new experiment dir) |
| Single `--model` only | `--models "a,b,c"` **and** `--models-file models.txt` |
| `--limit` default 50 | default 42 |
| CSV output only | Rich **Excel workbook** (`run_summary`, `questions`, `golden_answers`, one sheet per model) + raw CSV + raw JSONL + per-model log |
| No golden capture | Load golden xlsx, attach `golden_answer` per question for the per-model sheet **(never inserted into the prompt)** |
| No hashes | `prompt_hash`, `kb_hash`, `question_hash` (sha256) |
| No context-validity logic | Classify each model: `valid_full_context` / `truncated_context` / `runtime_unstable` / `failed` / `unknown_context_status` from `prompt_eval_count` vs `num_ctx` |
| No char count / notes | `response_char_count`, `notes` |
| Old grounding output format `Q:/Answer:/…` | Keep grounding rules; output format is fine but document it |

## 10. Risks before running

1. ⛔ **Dependencies missing** — `pandas`, `requests`, `openpyxl` (and `xlsxwriter` if used)
   not installed; **`ollama` not on PATH**. Must `pip install` + ensure ollama daemon +
   `ollama pull` each model. **Blocks all runs.**
2. ⚠️ **Two KB copies, different md5** — must read `data/GNEM_Excel_Data.xlsx` only; remove the
   stale top-level copy from the working path (move to backup) so no old script reads it.
3. ⚠️ **Silent context truncation** — in the 50q run, `qwen2.5:14b` silently capped at exactly
   32768 `prompt_eval_count` while siblings ingested 38–41K. The new runner must flag this as
   `truncated_context`, never `valid_full_context`. KB at 205 rows ≈ 38–41K prompt tokens, so
   `num_ctx=65536` is needed; models whose real cap is lower will truncate silently.
4. ⚠️ **Excel cell limits / newlines** — golden + model answers are long multi-line strings;
   Excel's 32,767-char cell limit must be guarded (truncate-with-flag, store full text in raw
   JSONL).
5. ⚠️ **Long runtimes / instability** — 50q run saw 1200 s timeouts and unstable DeepSeek on a
   24 GB GPU. Per-question and per-model failures must be caught and logged, run continues.
6. ⚠️ **Golden leakage** — golden answers must be loaded for the workbook only and must be
   provably absent from the prompt (smoke test asserts the golden string is not a substring of
   the built prompt).
7. ℹ️ **`.DS_Store`** files present — harmless, ignore.

## 11. Proposed clean folder structure

Per your suggestion. New experiment dir created fresh; inputs copied (not moved) from `Data/`:

```
ev_direct_context_42q/
  data/
    GNEM_Excel_Data.xlsx           # copy of Data/GNEM_Excel_Data.xlsx (the final KB)
    questions.txt                  # copy of Data/questions.txt (42 Qs)
    human_validated_answers.xlsx   # copy of Data/Human validated questions.xlsx (renamed for clarity)
  scripts/
    run_direct_context_42q.py
    models.txt                     # one model per line (you provide final list)
  results/
    model_responses_direct_context_42q.xlsx
    raw_csv/answers_<safe_model_name>.csv
    raw_json/answers_<safe_model_name>.jsonl
    kb_full.jsonl
  reports/
    EVALUATION_FRAMEWORK_DIRECT_CONTEXT_42Q.md
  logs/
    run_<safe_model_name>.log
  backups/
    old_50q_experiment_20260623_HHMMSS/   # everything from §7
```

(Open question Q-A below: build inside a new `ev_direct_context_42q/` dir, or restructure the
current repo in place. I will not create or move anything until you choose.)

## 12. Proposed output filenames

- Workbook: `results/model_responses_direct_context_42q.xlsx`
  - Sheet `run_summary` — one row per model (columns per your spec §"Required output 1").
  - Sheet `questions` — `question_number`, `question` (42 rows).
  - Sheet `golden_answers` — all meaningful golden columns (`Num`→`question_number`,
    `Use Case Category`, `Question`, `Human validated answers`); empty-named columns dropped.
  - One sheet per model named by `safe_model_name`, columns per your spec §"One sheet per model".
- `results/raw_csv/answers_<safe_model_name>.csv`
- `results/raw_json/answers_<safe_model_name>.jsonl`
- `results/kb_full.jsonl`
- `logs/run_<safe_model_name>.log`

`safe_model_name` = model with `:`→`_` and `/`→`_` (matches old convention, e.g.
`qwen2.5:14b` → `qwen2.5_14b`).

## 13. Proposed direct-context model-run plan

For each model in the list:
1. Build KB JSONL once (`excel_to_jsonl(data/GNEM_Excel_Data.xlsx)`), write `kb_full.jsonl`,
   compute `kb_hash`. KB is identical across models → built once, reused.
2. For each of the 42 questions (in order):
   - Build prompt = grounding rules + data dictionary + **full** JSONL KB + question
     (**no golden answer**). Record `prompt_hash`, `question_hash`.
   - Call Ollama `/api/generate`, `stream=false`, `temperature=0`, `seed=42`,
     `options.num_ctx=<num_ctx>`, `timeout=1200`.
   - Capture response + `prompt_eval_count`, `eval_count`, `total/load/prompt_eval/eval_duration`,
     `created_at`, `response_char_count`.
   - On per-question failure: write `status=error` + `error_message`, **continue** to next question.
3. After 42 questions: classify `context_status` from `prompt_eval_count` distribution vs
   `num_ctx` (see §10.3). Write per-model sheet + raw csv/jsonl + log + a `run_summary` row.
4. On per-model failure: log it, mark `status=failed`, **continue** to next model.
5. Grounding rules + data dictionary reused **verbatim** from `run_ollama_jsonl_eval.py`.
6. Golden answers are loaded only to populate the workbook; an assertion guarantees they are
   not in any prompt.

## 14. Proposed smoke test

After you approve this audit (and deps/ollama are installed):

```bash
python scripts/run_direct_context_42q.py --models "<ONE_MODEL>" --limit 2 --num_ctx 65536
```

Asserts:
- `data/` files load; KB = 205×15; questions = 42; golden = 42 and joinable by number.
- `kb_full.jsonl` written; `kb_hash` stable.
- **Golden answer string is NOT a substring of the built prompt** (leakage guard).
- 2 model responses captured with metadata (`prompt_eval_count` etc.).
- `context_status` computed.
- Workbook created with `run_summary`, `questions`, `golden_answers`, and the model sheet
  (2 rows); raw csv/jsonl/log written.

## 15. Questions for you before execution

- **Q-A (structure):** Build a fresh `ev_direct_context_42q/` dir (inputs copied from `Data/`),
  or restructure the current repo in place? I recommend the fresh dir — cleaner separation
  from the 50q work.
- **Q-B (final model list):** Please confirm the exact models. The list in your message
  (`qwen3.6:35b-a3b`, `gemma3:27b`, `mistral-small3.2:24b`, `qwen2.5:14b`, `deepseek-r1:32b`)
  is treated as a draft, not final, per your instruction.
- **Q-C (deps/runtime):** `pandas`/`requests`/`openpyxl` and `ollama` are not present here.
  Should I prepare a `pip install` + `ollama pull` checklist for you to run (via `!`), or is
  there another machine/GPU host where the runs will actually happen?
- **Q-D (golden columns):** Keep all meaningful golden columns (`Num`, `Use Case Category`,
  `Question`, `Human validated answers`) in the `golden_answers` sheet? (I'll drop the 2
  empty-named columns.)
- **Q-E (num_ctx):** Confirm `num_ctx=65536` as the default for all models (matches the 50q
  run; needed because the KB ≈ 38–41K prompt tokens).
- **Q-F (backup):** Approve moving the §7 items to `backups/old_50q_experiment_20260623_HHMMSS/`?
  I will show the exact file list first.

---

_No models will be run, and no files moved/edited/deleted, until you approve._
