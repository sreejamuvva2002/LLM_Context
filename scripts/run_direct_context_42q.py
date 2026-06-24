#!/usr/bin/env python3
"""
Direct-context response generation — Georgia EV supply-chain KB, 42 questions.

STAGE 1 + 2 only: generate responses from local LLMs and save them cleanly.
NO evaluation, NO grading, NO scoring happens here.

Professor/project constraints enforced:
  * The full KB is placed in the prompt as JSONL; the LLM answers from that only.
  * No SQL, no vector retrieval, no RAG routing, no Python-computed answers,
    no deterministic filtering. Python only reads Excel, builds prompts, calls
    Ollama, collects responses, and saves outputs/metadata.
  * Golden (human-validated) answers are loaded ONLY to populate the output
    workbook for later evaluation. They are asserted to never appear in any
    prompt (leakage guard).

Inputs (read directly from Data/, treated as final — never moved/modified):
  Data/GNEM_Excel_Data.xlsx              final KB (205 rows x 15 cols)
  Data/questions.txt                     42 questions, one per line
  Data/Human validated questions.xlsx    42 golden answers (Num 1..42)

Uses the v2 GNEM prompts for stricter exact-category filtering, row_id evidence,
and filter-vs-output separation.
"""

import argparse
import csv
import hashlib
import json
import statistics
import time
from pathlib import Path

import pandas as pd
import requests


# --------------------------------------------------------------------------- #
# Config / defaults
# --------------------------------------------------------------------------- #
OLLAMA_URL = "http://localhost:11434/api/generate"

DEFAULT_DATA_DIR = "Data"
DEFAULT_KB_FILE = "GNEM_Excel_Data.xlsx"
DEFAULT_QUESTIONS_FILE = "questions.txt"
DEFAULT_GOLDEN_FILE = "Human validated questions.xlsx"

RESULTS_DIR = "results/direct_context"
RAW_CSV_DIR = f"{RESULTS_DIR}/raw_csv"
RAW_JSON_DIR = f"{RESULTS_DIR}/raw_json"
LOGS_DIR = f"{RESULTS_DIR}/logs"
WORKBOOK_PATH = f"{RESULTS_DIR}/model_responses_direct_context_42q.xlsx"

REQUEST_TIMEOUT = 1200          # seconds, per question
EXCEL_CELL_LIMIT = 32767        # Excel hard cap per cell; full text still in raw JSONL


# --------------------------------------------------------------------------- #
# Grounding rules + data dictionary.
# --------------------------------------------------------------------------- #
DATA_DICTIONARY = """
DATA DICTIONARY AND RULES

Each JSONL line is one record with a unique integer row_id and these fields:
row_id, Company, Category, Industry Group, Location, Address, Latitude, Longitude,
Primary Facility Type, EV Supply Chain Role, Primary OEMs,
Supplier or Affiliation Type, Employment, Product / Service,
EV / Battery Relevant, Classification Method.

Field meaning:
- Category = OEM, or the supply-chain tier, written exactly as one of:
  OEM, Tier 1, Tier 1/2, Tier 2, Tier 2/3, Tier 3.
- EV Supply Chain Role = one controlled value, e.g. Vehicle Assembly, Battery Cell,
  Battery Pack, Power Electronics, Charging Infrastructure, Thermal Management,
  Materials, General Automotive.
- Location is a single text field formatted like "City, County".
- Address, Latitude, and Longitude describe the Georgia site when available.
- Primary Facility Type = Manufacturing Plant, Headquarters, R&D, Distribution, etc.
- EV / Battery Relevant = Yes, Indirect (a.k.a. "Indirectly Relevant"), or No.
- Employment = the Employment value shown in that record.
- "Area" means a County, which is the County part of Location. Group by County
  for any question about Georgia areas.

EXACT-MATCH RULES:
- Treat "Tier 1/2" as one exact category. Never split it into Tier 1 or Tier 2.
- Treat "Tier 2/3" as one exact category. Never split it into Tier 2 or Tier 3.
- A quoted category in the question, e.g. "Tier 1/2", matches only rows whose
  Category equals that value exactly.
- Match all categorical and coded values exactly as written in the JSONL.

FILTER vs OUTPUT:
- A field named only as something to show is an output field, not a filter.
  Example: "Show all Tier 1/2 suppliers and list their EV Supply Chain Role"
  means FILTER: Category == "Tier 1/2"; OUTPUT: EV Supply Chain Role.
- Do not filter on EV Supply Chain Role or Product / Service unless the question
  explicitly asks for that filter.
- For quoted category filters, inspect the Category field itself. Do not infer
  category from company name, product, role, row order, or general automotive/EV
  relevance.
- Do not drop a row because its EV Supply Chain Role is "General Automotive" or
  because its Product / Service looks less EV-specific. If it matches the stated
  filter, include it.
- Combine multiple stated conditions with AND. Do not include partial matches.

EMPLOYMENT AGGREGATION:
- For any sum, total, average, ranking, highest/lowest, or top-N of Employment,
  use the Employment values exactly as shown in the selected records.
- The same company may appear under multiple row_ids because the KB may include
  multiple Georgia sites. Add each site's Employment separately for site-level
  totals. Count distinct companies only when the question asks about companies
  rather than records/sites.

GROUNDING:
- Answer only from the records provided.
- Do not use outside knowledge.
- Never invent a Company, row_id, county, number, tier, role, product, address,
  employment value, or OEM.
- If the records do not answer the question, answer exactly:
  Not found in the data.
"""


FULL_CONTEXT_PROMPT_TEMPLATE = """
You are a helpful Georgia automotive/EV supply-chain assistant.

You are given the complete spreadsheet knowledge base as JSONL, one record per
line, and one user question. Answer naturally for a user, but stay strictly
grounded in the provided records.

Use only the KB records below. Do not use prior knowledge.

{data_dictionary}

KNOWLEDGE BASE (JSONL, one record per line):
{kb_jsonl}

USER QUESTION:
{question}

TASK:
1. Identify the filter conditions and the output fields separately.
2. Scan every record in row_id order, first to last.
3. Select all rows that satisfy all stated filter conditions exactly.
   Do not stop after finding a few examples. Do not apply an implicit top-N,
   sample, or limit unless the question explicitly asks for one.
   If the question says all, list every, show all, or similar, include every
   matching row. There may be more than 10 matching rows; never stop at 10 or
   any other round number unless the question explicitly asks for top-N.
4. If the question asks for count, sum, ranking, highest/lowest, top-N, or
   areas, compute the result only from the selected rows.
5. Every company, number, county, role, tier, product, and conclusion must be
   supported by real rows from the KB.
6. For list questions, put the complete list of matching records in the Evidence
   table, not as a long bullet list in Answer.
7. Write a concise user-facing Answer, then provide structured Evidence for
   verification.

STRICT CONSISTENCY RULES:
- Every company named in Answer must appear in the Evidence table.
- Every Evidence row must use a real row_id from the KB.
- Every Evidence row must satisfy all stated filter conditions exactly. If a row
  does not match the exact filter field/value, remove it before answering.
- Supporting row_ids must exactly match the row_ids shown in Evidence.
- Count must equal the number of Evidence rows.
- For list/count questions, the count stated in Answer must equal Count.
- Do not include companies in Answer that are missing from Evidence.
- Do not include Evidence rows that are not actually used in Answer.
- Do not copy examples or invent placeholders.

SILENT SELF-CHECK before answering. Do not print this checklist:
- Did I filter only on the stated conditions, not on output-only fields?
- Did I scan the full KB?
- Is every reported row_id actually present in the KB?
- Does Count equal the number of Evidence rows?
- For employment math, did I use only Employment values from selected records?
- If I say "Not found in the data.", is it because the KB truly does not contain
  enough information?

OUTPUT FORMAT:
Answer: <clean, direct, natural-language answer for the user. Start with the
result. For list questions, state how many matching records/companies were found
and briefly say that the requested fields are shown in Evidence; do not repeat
the full list in Answer. For count/sum/ranking/area questions, give the result
directly and briefly explain how it was derived from the KB. If the KB does not
contain the answer, write exactly: Not found in the data.>

Evidence:
| row_id | Company | <field A needed for this question> | <field B needed for this question> |
|---|---|---|---|
| <id> | <company> | ... | ... |

Count: <number of Evidence rows>

Supporting row_ids: <comma-separated row_id list, each appearing once>

Notes: <ties, ambiguity, data limitation, or blank>
"""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_name(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


def clean_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    if isinstance(value, str):
        value = value.strip()
        return value if value else None
    return value


def excel_to_jsonl(xlsx_file):
    """Full KB -> JSONL string. Drops fully-empty rows; adds 1-based row_id."""
    df = pd.read_excel(xlsx_file)
    df = df.dropna(how="all").reset_index(drop=True)
    df.columns = [str(c).strip() for c in df.columns]

    records = []
    for idx, row in df.iterrows():
        record = {"row_id": int(idx + 1)}
        for col in df.columns:
            record[col] = clean_value(row[col])
        records.append(record)

    jsonl_text = "\n".join(
        json.dumps(record, ensure_ascii=False) for record in records
    )
    return jsonl_text, records, list(df.columns)


def read_questions(path):
    """Strip + skip-blank reader (robust to a missing trailing newline)."""
    questions = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            questions.append(line)
    return questions


def read_golden(path):
    """
    Golden answers for later evaluation ONLY.
    Returns {question_number: golden_answer_text}.
    Only the required columns are kept (Num -> question_number,
    Human validated answers -> golden_answer); empty-named columns dropped.
    """
    df = pd.read_excel(path)
    df = df.loc[:, [c for c in df.columns if str(c).strip() != ""]]
    df.columns = [str(c).strip() for c in df.columns]

    num_col = next((c for c in df.columns if c.lower() == "num"), df.columns[0])
    ans_col = next(
        (c for c in df.columns if "validated" in c.lower() or "answer" in c.lower()),
        df.columns[-1],
    )

    golden = {}
    for _, row in df.iterrows():
        try:
            qn = int(row[num_col])
        except (ValueError, TypeError):
            continue
        val = row[ans_col]
        golden[qn] = "" if pd.isna(val) else str(val).strip()
    return golden, num_col, ans_col


def build_prompt(jsonl_kb, question):
    """Grounding rules + data dictionary + full JSONL KB + question. NO golden."""
    return FULL_CONTEXT_PROMPT_TEMPLATE.format(
        data_dictionary=DATA_DICTIONARY.strip(),
        kb_jsonl=jsonl_kb,
        question=question,
    ).strip()


def call_ollama(model, prompt, num_ctx):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "seed": 42, "num_ctx": num_ctx},
    }
    # Ask Ollama to return the model's native reasoning in a SEPARATE `thinking`
    # field, kept out of the answer text. Thinking models (qwen3, deepseek-r1)
    # populate it; models that don't support thinking reject `think: true` with
    # a 400 -> retry once without it (native_thinking stays blank for those).
    response = requests.post(
        OLLAMA_URL, json={**payload, "think": True}, timeout=REQUEST_TIMEOUT
    )
    if response.status_code == 400:
        response = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    return {
        "answer": data.get("response", "").strip(),
        "native_thinking": (data.get("thinking") or "").strip(),
        "prompt_eval_count": data.get("prompt_eval_count"),
        "eval_count": data.get("eval_count"),
        "total_duration": data.get("total_duration"),
        "load_duration": data.get("load_duration"),
        "prompt_eval_duration": data.get("prompt_eval_duration"),
        "eval_duration": data.get("eval_duration"),
        "created_at": data.get("created_at"),
    }


def classify_context(rows, num_ctx, approx_sent_tokens):
    """
    Decide whether the model actually ingested the full KB.
    The truncation test is prompt_eval_count (tokens the model really ingested)
    vs the tokens we sent. A model that silently caps lower is TRUNCATED, not valid.
    Returns (context_status, note).
    """
    ok = [r for r in rows if r["status"] == "ok" and r["prompt_eval_count"] is not None]
    if not ok:
        return "failed", "no successful responses with prompt_eval_count"

    pec = [int(r["prompt_eval_count"]) for r in ok]
    med = statistics.median(pec)
    mx = max(pec)

    # We expect prompt_eval_count to be close to the tokens we actually sent.
    # If the model ingested materially fewer than we sent, it truncated.
    if approx_sent_tokens and med < 0.85 * approx_sent_tokens:
        return (
            "truncated_context",
            f"median prompt_eval_count {med} << ~{approx_sent_tokens} sent tokens "
            f"(approx len/4); model silently capped context",
        )

    # All-identical prompt_eval_count at a round number strictly below num_ctx is
    # the classic silent-cap signature.
    if len(set(pec)) == 1 and pec[0] < num_ctx and pec[0] in (2048, 4096, 8192, 16384, 32768):
        return (
            "truncated_context",
            f"prompt_eval_count pinned at {pec[0]} on every question (hard cap < num_ctx)",
        )

    return "valid_full_context", f"prompt_eval_count median {med}, max {mx} (num_ctx {num_ctx})"


# --------------------------------------------------------------------------- #
# Per-model run
# --------------------------------------------------------------------------- #
PER_MODEL_COLUMNS = [
    "question_number", "question", "golden_answer", "model_name", "model_response",
    "status", "error_message", "num_ctx_requested", "actual_context_observed_if_available",
    "prompt_eval_count", "eval_count", "total_duration", "load_duration",
    "prompt_eval_duration", "eval_duration", "created_at", "prompt_hash", "kb_hash",
    "question_hash", "response_char_count", "notes",
]


def run_model(model, questions, golden, jsonl_kb, kb_hash, num_ctx, approx_sent_tokens):
    sname = safe_name(model)
    log_path = Path(LOGS_DIR) / f"run_{sname}.log"
    csv_path = Path(RAW_CSV_DIR) / f"answers_{sname}.csv"
    jsonl_path = Path(RAW_JSON_DIR) / f"answers_{sname}.jsonl"

    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    log(f"=== MODEL {model} (safe={sname}) | num_ctx={num_ctx} | kb_hash={kb_hash[:12]} ===")

    rows = []
    model_failed = False
    model_fail_msg = ""
    runtimes = []

    # open raw files for incremental, durable writes
    jf = jsonl_path.open("w", encoding="utf-8")

    for q_num, question in enumerate(questions, start=1):
        prompt = build_prompt(jsonl_kb, question)
        gold = golden.get(q_num, "")

        # LEAKAGE GUARD: the golden answer must never be inside the prompt.
        if gold and gold in prompt:
            raise AssertionError(
                f"GOLDEN LEAKAGE: golden answer for Q{q_num} found inside the prompt"
            )

        row = {c: None for c in PER_MODEL_COLUMNS}
        row.update({
            "question_number": q_num,
            "question": question,
            "golden_answer": gold,
            "model_name": model,
            "num_ctx_requested": num_ctx,
            "prompt_hash": sha256(prompt),
            "kb_hash": kb_hash,
            "question_hash": sha256(question),
            "notes": "",
            "native_thinking": "",  # raw-JSONL only; never enters answer sheets/grading
        })

        log(f"\n--- Q{q_num} ---\n{question}")
        start = time.time()
        try:
            res = call_ollama(model, prompt, num_ctx)
            elapsed = time.time() - start
            runtimes.append(elapsed)
            answer = res["answer"]
            row.update({
                "model_response": answer,
                "native_thinking": res["native_thinking"],
                "status": "ok",
                "error_message": "",
                "actual_context_observed_if_available": res["prompt_eval_count"],
                "prompt_eval_count": res["prompt_eval_count"],
                "eval_count": res["eval_count"],
                "total_duration": res["total_duration"],
                "load_duration": res["load_duration"],
                "prompt_eval_duration": res["prompt_eval_duration"],
                "eval_duration": res["eval_duration"],
                "created_at": res["created_at"],
                "response_char_count": len(answer),
            })
            log(f"OK in {elapsed:.1f}s | prompt_eval_count={res['prompt_eval_count']} "
                f"| chars={len(answer)}")
            log("Answer preview: " + answer[:400].replace("\n", " "))
        except Exception as e:  # per-question failure: log and CONTINUE
            elapsed = time.time() - start
            row.update({
                "model_response": "",
                "status": "error",
                "error_message": f"{type(e).__name__}: {e}",
                "response_char_count": 0,
                "notes": "per-question failure; continued to next question",
            })
            log(f"ERROR after {elapsed:.1f}s: {type(e).__name__}: {e}")

        rows.append(row)
        jf.write(json.dumps(row, ensure_ascii=False) + "\n")
        jf.flush()

    jf.close()

    # raw CSV (answer columns only; native_thinking is excluded here and lives
    # in raw JSONL + its own workbook sheet, so grading inputs stay clean).
    with csv_path.open("w", newline="", encoding="utf-8") as cf:
        w = csv.DictWriter(cf, fieldnames=PER_MODEL_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # summary + context classification
    completed = sum(1 for r in rows if r["status"] == "ok")
    errors = sum(1 for r in rows if r["status"] == "error")

    if completed == 0:
        model_failed = True
        model_fail_msg = "every question errored"

    context_status, ctx_note = classify_context(rows, num_ctx, approx_sent_tokens)

    pec_ok = [int(r["prompt_eval_count"]) for r in rows
              if r["status"] == "ok" and r["prompt_eval_count"] is not None]
    pec_min = min(pec_ok) if pec_ok else None
    pec_med = int(statistics.median(pec_ok)) if pec_ok else None
    pec_max = max(pec_ok) if pec_ok else None

    if model_failed:
        status = "failed"
    elif errors > 0 and errors >= max(1, len(questions) // 3):
        status = "runtime_unstable"
    elif errors > 0:
        status = "completed_with_errors"
    else:
        status = "completed"

    summary = {
        "model_name": model,
        "safe_model_name": sname,
        "total_questions": len(questions),
        "completed_answers": completed,
        "error_count": errors,
        "context_status": context_status,
        "num_ctx_requested": num_ctx,
        "prompt_eval_count_min": pec_min,
        "prompt_eval_count_median": pec_med,
        "prompt_eval_count_max": pec_max,
        "total_runtime_seconds": round(sum(runtimes), 2),
        "average_runtime_seconds": round(sum(runtimes) / len(runtimes), 2) if runtimes else None,
        "status": status,
        "notes": "; ".join(filter(None, [ctx_note, model_fail_msg])),
    }
    log(f"\n=== SUMMARY {model}: status={status}, completed={completed}/{len(questions)}, "
        f"errors={errors}, context={context_status} ===")
    return rows, summary


# --------------------------------------------------------------------------- #
# Workbook writer (truncates oversized cells; full text lives in raw JSONL)
# --------------------------------------------------------------------------- #
def cap_cell(v):
    if isinstance(v, str) and len(v) > EXCEL_CELL_LIMIT:
        return v[: EXCEL_CELL_LIMIT - 40] + "...[TRUNCATED FOR EXCEL; see raw_json]"
    return v


def cap_cells(df):
    return df.apply(lambda col: col.map(cap_cell))


def write_workbook(questions, golden, golden_meta, per_model_rows, summaries):
    out = Path(WORKBOOK_PATH)
    with pd.ExcelWriter(out, engine="openpyxl") as xl:
        # run_summary
        pd.DataFrame(summaries, columns=[
            "model_name", "safe_model_name", "total_questions", "completed_answers",
            "error_count", "context_status", "num_ctx_requested",
            "prompt_eval_count_min", "prompt_eval_count_median", "prompt_eval_count_max",
            "total_runtime_seconds", "average_runtime_seconds", "status", "notes",
        ]).to_excel(xl, sheet_name="run_summary", index=False)

        # questions
        pd.DataFrame(
            [{"question_number": i + 1, "question": q} for i, q in enumerate(questions)]
        ).to_excel(xl, sheet_name="questions", index=False)

        # golden_answers (required columns only)
        cap_cells(pd.DataFrame(
            [{"question_number": qn, "question": questions[qn - 1] if qn - 1 < len(questions) else "",
              "golden_answer": golden.get(qn, "")} for qn in sorted(golden)]
        )).to_excel(xl, sheet_name="golden_answers", index=False)

        # one sheet per model (sheet name <= 31 chars). PER_MODEL_COLUMNS excludes
        # native_thinking, so these answer sheets never carry the reasoning trace.
        for model, rows in per_model_rows.items():
            sheet = safe_name(model)[:31]
            df = cap_cells(pd.DataFrame(rows, columns=PER_MODEL_COLUMNS))
            df.to_excel(xl, sheet_name=sheet, index=False)

        # native_thinking — isolated reasoning traces (thinking models only).
        # Separate sheet, not merged into any answer sheet; full text in raw JSONL.
        thinking_rows = [
            {"model_name": model,
             "question_number": r.get("question_number"),
             "native_thinking": r.get("native_thinking", "")}
            for model, rows in per_model_rows.items() for r in rows
        ]
        if any(t["native_thinking"] for t in thinking_rows):
            cap_cells(pd.DataFrame(
                thinking_rows,
                columns=["model_name", "question_number", "native_thinking"],
            )).to_excel(xl, sheet_name="native_thinking", index=False)
    print(f"Workbook written: {out}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def resolve_models(args):
    if args.models_file:
        models = [l.strip() for l in Path(args.models_file).read_text().splitlines()
                  if l.strip() and not l.strip().startswith("#")]
    elif args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        raise SystemExit("Provide --models 'a,b' or --models-file models.txt")
    return models


def main():
    p = argparse.ArgumentParser(description="Direct-context 42q LLM response generation")
    p.add_argument("--models", help="comma-separated model list")
    p.add_argument("--models-file", help="file with one model per line")
    p.add_argument("--num_ctx", type=int, default=65536)
    p.add_argument("--limit", type=int, default=42)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--kb-file", default=DEFAULT_KB_FILE)
    p.add_argument("--questions-file", default=DEFAULT_QUESTIONS_FILE)
    p.add_argument("--golden-file", default=DEFAULT_GOLDEN_FILE)
    args = p.parse_args()

    for d in (RESULTS_DIR, RAW_CSV_DIR, RAW_JSON_DIR, LOGS_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)

    data = Path(args.data_dir)
    models = resolve_models(args)

    # --- load inputs (Data/ is read-only) ---
    jsonl_kb, records, columns = excel_to_jsonl(data / args.kb_file)
    questions = read_questions(data / args.questions_file)[: args.limit]
    golden, gnum_col, gans_col = read_golden(data / args.golden_file)

    kb_hash = sha256(jsonl_kb)
    Path(RESULTS_DIR, "kb_full.jsonl").write_text(jsonl_kb, encoding="utf-8")

    # approx tokens we will SEND (KB dominates; question text is tiny) -> truncation baseline
    sample_prompt = build_prompt(jsonl_kb, questions[0] if questions else "")
    approx_sent_tokens = len(sample_prompt) // 4

    print(f"KB rows: {len(records)} | columns: {len(columns)}")
    print(f"Questions: {len(questions)} (limit {args.limit})")
    print(f"Golden answers: {len(golden)} (from cols '{gnum_col}' / '{gans_col}')")
    print(f"kb_hash: {kb_hash}")
    print(f"approx sent tokens (len/4): ~{approx_sent_tokens}")
    print(f"Models: {models}")

    # sanity warnings (non-fatal)
    if len(questions) != 42:
        print(f"WARNING: expected 42 questions, found {len(questions)}")
    missing_gold = [i for i in range(1, len(questions) + 1) if i not in golden]
    if missing_gold:
        print(f"WARNING: no golden answer for questions {missing_gold}")

    per_model_rows = {}
    summaries = []
    for model in models:
        try:
            rows, summary = run_model(
                model, questions, golden, jsonl_kb, kb_hash, args.num_ctx, approx_sent_tokens
            )
        except Exception as e:  # per-MODEL failure: log and CONTINUE to next model
            print(f"MODEL-LEVEL FAILURE for {model}: {type(e).__name__}: {e}")
            rows = []
            summary = {
                "model_name": model, "safe_model_name": safe_name(model),
                "total_questions": len(questions), "completed_answers": 0,
                "error_count": len(questions), "context_status": "failed",
                "num_ctx_requested": args.num_ctx,
                "prompt_eval_count_min": None, "prompt_eval_count_median": None,
                "prompt_eval_count_max": None, "total_runtime_seconds": None,
                "average_runtime_seconds": None, "status": "failed",
                "notes": f"model-level failure: {type(e).__name__}: {e}",
            }
        per_model_rows[model] = rows
        summaries.append(summary)
        # rewrite workbook after every model so partial progress is durable
        write_workbook(questions, golden, (gnum_col, gans_col), per_model_rows, summaries)

    print("\nDone. Workbook + raw csv/json + logs written.")


if __name__ == "__main__":
    main()
