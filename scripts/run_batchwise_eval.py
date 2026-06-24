#!/usr/bin/env python3
"""
Batch-wise LLM map-reduce evaluation for the Georgia EV supply-chain 42q run.

This is a separate experimental arm from full-context direct prompting.
The LLM performs both stages:
  * MAP: select candidate-relevant record IDs from each disjoint KB batch.
  * REDUCE: answer from the union of selected rows.

Python only loads data, batches records, validates JSON, reconstructs selected
rows by ID, logs diagnostics, and saves output. It never computes or corrects
the final answer.
"""

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests


OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
REQUEST_TIMEOUT = 1200

DEFAULT_KB = "Data/GNEM_Excel_Data.xlsx"
DEFAULT_QUESTIONS = "Data/questions.txt"
DEFAULT_MODELS = "scripts/models_batchwise_candidates.txt"
DEFAULT_OUT = "results/batchwise"

BATCH_DATA_DICTIONARY = """
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

DATA_DICTIONARY = BATCH_DATA_DICTIONARY

MAP_SYSTEM_PROMPT = """
You are scanning one small batch from a larger Georgia EV supply-chain knowledge
base.

Your job is NOT to answer the user question.

Your job is only to identify which rows in this batch satisfy the question's
filter conditions.

Return strict JSON only. Do not include prose, markdown, reasoning text, or a
final answer.
"""

MAP_USER_TEMPLATE = """
USER QUESTION:
{question}

{data_dictionary}

BATCH {batch_id} (JSONL, one record per line):
{batch_jsonl}

BATCH-SCANNING RULES:
- Use only the rows in this batch.
- Do not use outside knowledge.
- Scan the batch records in row_id order, first to last.
- Separate filter conditions from output fields.
- Filter only on conditions explicitly stated in the question.
- Apply exact-category rules. "Tier 1/2" and "Tier 2/3" are single categories.
- Combine multiple stated conditions with AND.
- Do not include partial matches.
- If the question says all, list every, show all, or similar, include every
  matching row in this batch. There may be more than 10 matches across batches;
  never stop at 10 or any other round number unless the question explicitly asks
  for top-N.
- Do not exclude a matching row for being "General Automotive" or for looking
  less EV-specific.
- For employment, count, ranking, top-N, highest/lowest, or area questions, only
  select rows that are relevant evidence. Do not compute the final aggregate here.
- Be careful: high recall is useful, but do not include rows that clearly fail
  the stated conditions.

Return JSON exactly in this shape:
{{
  "batch_id": "{batch_id}",
  "filter_interpretation": "<briefly state the exact filters applied>",
  "matched_row_ids": [<row_id>, ...]
}}

If nothing matches, return:
{{
  "batch_id": "{batch_id}",
  "filter_interpretation": "<briefly state the exact filters applied>",
  "matched_row_ids": []
}}
"""

VERIFY_SYSTEM_PROMPT = """
You are verifying candidate rows selected from an exhaustive batch scan.

Your job is NOT to answer the user question.

Your job is to check whether each candidate row truly satisfies all stated filter
conditions in the original question.

Return strict JSON only.
"""

VERIFY_USER_TEMPLATE = """
USER QUESTION:
{question}

{data_dictionary}

CANDIDATE ROWS (JSONL, one record per line):
{candidate_rows_jsonl}

VERIFICATION RULES:
- Use only the candidate rows shown here.
- Do not use outside knowledge.
- Re-check every candidate row against the original question.
- Apply all stated filter conditions with AND.
- Keep a row only if it truly satisfies the stated filters.
- Remove a row if it is only generally related to automotive or EV but does not
  satisfy the exact question.
- Do not drop a row because an output field looks less relevant if it satisfies
  the actual filter conditions.
- Do not compute the final answer here.

Return JSON exactly in this shape:
{{
  "verified_row_ids": [<row_id>, ...],
  "removed_row_ids": [<row_id>, ...],
  "verification_notes": "<brief explanation>"
}}
"""

REDUCE_SYSTEM_PROMPT = """
You are a helpful Georgia automotive/EV supply-chain assistant.

The relevant rows have already been selected and verified from the full KB.
Your job is to answer the user's question naturally using only these verified
rows.

Do not use outside knowledge.
Do not invent companies, numbers, counties, tiers, roles, products, or row_ids.
"""

REDUCE_USER_TEMPLATE = """
{data_dictionary}

ORIGINAL USER QUESTION:
{question}

VERIFIED_ROW_IDS:
{verified_row_ids}

VERIFIED ROWS (JSONL, one full record per line):
{relevant_rows_jsonl}

TASK:
1. Use only the verified rows above.
2. Determine whether the question asks for a list, lookup, count, aggregation,
   ranking, highest/lowest, top-N, or areas/counties.
3. For list or lookup questions, answer naturally and cite the supporting rows
   in the Evidence table.
4. For count questions, list the evidence rows first and then state the count.
5. For Employment aggregation questions, compute using only the Employment values
   shown in the verified rows. Briefly state how the total/ranking was derived.
6. For area questions, group by County using the County part of Location.
7. If the verified rows do not contain enough evidence to answer, answer exactly:
   Not found in the data.
8. Do not use rows outside VERIFIED_ROW_IDS.
9. Do not invent missing values.

STRICT CONSISTENCY RULES:
- Every company named in Answer must appear in the Evidence table.
- Every Evidence row must come from VERIFIED_ROW_IDS.
- Every Evidence row must satisfy all stated filter conditions exactly. If a row
  does not match the exact filter field/value, remove it before answering.
- Supporting row_ids must exactly match the row_ids shown in Evidence.
- Count must equal the number of Evidence rows.
- For list/count questions, the count stated in Answer must equal Count.
- Do not include companies in Answer that are missing from Evidence.
- Do not include Evidence rows that are not used in Answer.
- Do not copy examples or invent placeholders.

SILENT SELF-CHECK before answering. Do not print this checklist:
- Did I use only the verified rows?
- Does every row_id in Evidence come from VERIFIED_ROW_IDS?
- Does Count equal the number of Evidence rows?
- For employment math, did I use only Employment values from the verified rows?
- Is my answer natural and useful for a user, while still evidence-grounded?

OUTPUT FORMAT:
Answer: <clean, direct, natural-language answer for the user. Start with the
result. For list questions, state how many matching records/companies were found
and name them with the requested fields. For count/sum/ranking/area questions,
give the result directly and briefly explain how it was derived from the verified
rows. If the verified rows do not contain the answer, write exactly:
Not found in the data.>

Evidence:
| row_id | Company | <field A needed for this question> | <field B needed for this question> |
|---|---|---|---|
| <id> | <company> | ... | ... |

Count: <number of Evidence rows>

Supporting row_ids: <comma-separated verified row_id list, each appearing once>

Notes: <ties, ambiguity, data limitation, verification issue, or blank>
"""


def safe_model_name(model):
    return model.replace(":", "_").replace("/", "_").replace(".", "_")


def clean_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    if isinstance(value, str):
        value = value.strip()
        return value if value else None
    return value


def normalize_bool(value):
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "yes", "y", "1"}


def load_kb(path, sheet, sort_key):
    read_kwargs = {"sheet_name": sheet} if sheet else {}
    df = pd.read_excel(path, **read_kwargs)
    df = df.dropna(how="all").reset_index(drop=True)
    df.columns = [str(c).strip() for c in df.columns]

    id_col = sort_key if sort_key in df.columns else None
    if id_col is None:
        id_col = "Record_ID" if "Record_ID" in df.columns else "row_id"
    if id_col not in df.columns:
        df.insert(0, id_col, range(1, len(df) + 1))

    df[id_col] = df[id_col].map(lambda v: str(clean_value(v)))
    df = df.sort_values(id_col, key=lambda s: s.map(natural_sort_key)).reset_index(drop=True)

    records = []
    for _, row in df.iterrows():
        record = {}
        for col in df.columns:
            record[col] = clean_value(row[col])
        record["Record_ID"] = str(record[id_col])
        if "Is_Announcement" not in record:
            record["Is_Announcement"] = False
        records.append(record)

    row_json_by_id = {
        record["Record_ID"]: json.dumps(record, ensure_ascii=False)
        for record in records
    }
    row_by_id = {record["Record_ID"]: record for record in records}
    return records, row_json_by_id, row_by_id, id_col


def natural_sort_key(value):
    text = "" if value is None else str(value)
    parts = re.split(r"(\d+)", text)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def make_batches(records, batch_size):
    batches = []
    for idx in range(0, len(records), batch_size):
        batch_records = records[idx: idx + batch_size]
        batch_id = f"b{len(batches):03d}"
        batch_jsonl = "\n".join(
            json.dumps(record, ensure_ascii=False) for record in batch_records
        )
        batch_ids = {record["Record_ID"] for record in batch_records}
        batches.append({
            "batch_id": batch_id,
            "records": batch_records,
            "jsonl": batch_jsonl,
            "ids": batch_ids,
        })
    return batches


def read_questions(path, limit):
    text = Path(path).read_text(encoding="utf-8")
    questions = []
    if path.endswith(".jsonl"):
        for line in text.splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            questions.append({
                "q_id": str(obj.get("q_id", len(questions) + 1)),
                "question": str(obj["question"]).strip(),
            })
    else:
        for idx, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if line:
                questions.append({"q_id": str(idx), "question": line})
    return questions[:limit] if limit else questions


def read_models(path):
    models = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            models.append(line)
    if not models:
        raise SystemExit(f"No active models found in {path}")
    return models


def strip_think_blocks(text):
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE).strip()


def strip_json_fences_and_prose(text):
    text = strip_think_blocks(text)
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start: end + 1]
    return text


def call_ollama_chat(model, system_prompt, user_prompt, num_ctx, temperature):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "num_ctx": num_ctx,
            "temperature": temperature,
            "seed": 42,
        },
    }
    response = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    message = data.get("message") or {}
    return {
        "content": message.get("content", ""),
        "prompt_eval_count": data.get("prompt_eval_count"),
        "eval_count": data.get("eval_count"),
        "total_duration": data.get("total_duration"),
        "load_duration": data.get("load_duration"),
        "prompt_eval_duration": data.get("prompt_eval_duration"),
        "eval_duration": data.get("eval_duration"),
    }


def run_map(model, question, batch, args):
    user_prompt = MAP_USER_TEMPLATE.format(
        question=question,
        data_dictionary=DATA_DICTIONARY.strip(),
        batch_id=batch["batch_id"],
        batch_jsonl=batch["jsonl"],
    )
    last_error = ""
    last_raw = ""

    for attempt in range(args.max_map_retries + 1):
        try:
            res = call_ollama_chat(
                model=model,
                system_prompt=MAP_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                num_ctx=args.map_num_ctx,
                temperature=args.temperature,
            )
            last_raw = res["content"]
            parsed = json.loads(strip_json_fences_and_prose(last_raw))
            ids = parsed.get("matched_row_ids", parsed.get("relevant_record_ids"))
            if not isinstance(ids, list):
                raise ValueError("matched_row_ids is not a list")
            clean_ids = []
            seen = set()
            for rid in ids:
                rid = str(rid)
                if rid in batch["ids"] and rid not in seen:
                    clean_ids.append(rid)
                    seen.add(rid)
            return {
                "batch_id": batch["batch_id"],
                "relevant_record_ids": clean_ids,
                "matched_row_ids": clean_ids,
                "filter_interpretation": str(parsed.get("filter_interpretation", "")),
                "map_status": "ok",
                "attempts": attempt + 1,
                "dropped_hallucinated_count": len(ids) - len(clean_ids),
                "prompt_eval_count": res["prompt_eval_count"],
                "eval_count": res["eval_count"],
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"

    return {
        "batch_id": batch["batch_id"],
        "relevant_record_ids": [],
        "matched_row_ids": [],
        "filter_interpretation": "",
        "map_status": "failed",
        "attempts": args.max_map_retries + 1,
        "error": last_error,
        "raw_response_preview": last_raw[:500],
    }


def run_verify(model, question, candidate_ids, candidate_rows_jsonl, args):
    if not candidate_ids:
        return {
            "verify_status": "ok",
            "verified_row_ids": [],
            "removed_row_ids": [],
            "verification_notes": "No candidate rows to verify.",
            "prompt_eval_count": None,
            "eval_count": None,
        }

    user_prompt = VERIFY_USER_TEMPLATE.format(
        question=question,
        data_dictionary=DATA_DICTIONARY.strip(),
        candidate_rows_jsonl=candidate_rows_jsonl,
    )
    try:
        res = call_ollama_chat(
            model=model,
            system_prompt=VERIFY_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            num_ctx=args.reduce_num_ctx,
            temperature=args.temperature,
        )
        parsed = json.loads(strip_json_fences_and_prose(res["content"]))
        verified = parsed.get("verified_row_ids", [])
        removed = parsed.get("removed_row_ids", [])
        if not isinstance(verified, list) or not isinstance(removed, list):
            raise ValueError("verified_row_ids/removed_row_ids must be lists")

        candidate_set = set(candidate_ids)
        verified_ids = []
        seen = set()
        for rid in verified:
            rid = str(rid)
            if rid in candidate_set and rid not in seen:
                verified_ids.append(rid)
                seen.add(rid)
        removed_ids = []
        seen_removed = set()
        for rid in removed:
            rid = str(rid)
            if rid in candidate_set and rid not in seen_removed:
                removed_ids.append(rid)
                seen_removed.add(rid)

        return {
            "verify_status": "ok",
            "verified_row_ids": sorted(verified_ids, key=natural_sort_key),
            "removed_row_ids": sorted(removed_ids, key=natural_sort_key),
            "verification_notes": str(parsed.get("verification_notes", "")),
            "dropped_hallucinated_verified_count": len(verified) - len(verified_ids),
            "dropped_hallucinated_removed_count": len(removed) - len(removed_ids),
            "prompt_eval_count": res.get("prompt_eval_count"),
            "eval_count": res.get("eval_count"),
        }
    except Exception as exc:
        return {
            "verify_status": "failed",
            "verified_row_ids": candidate_ids,
            "removed_row_ids": [],
            "verification_notes": (
                f"Verification failed ({type(exc).__name__}: {exc}); "
                "falling back to MAP union."
            ),
            "prompt_eval_count": None,
            "eval_count": None,
        }


def run_reduce(model, question, verified_row_ids, relevant_rows_jsonl, args):
    user_prompt = REDUCE_USER_TEMPLATE.format(
        data_dictionary=DATA_DICTIONARY.strip(),
        question=question,
        verified_row_ids=", ".join(verified_row_ids) if verified_row_ids else "(none)",
        relevant_rows_jsonl=relevant_rows_jsonl,
    )
    res = call_ollama_chat(
        model=model,
        system_prompt=REDUCE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        num_ctx=args.reduce_num_ctx,
        temperature=args.temperature,
    )
    res["answer"] = strip_think_blocks(res["content"])
    return res


def diagnostics(union_ids, row_by_id, total_rows, n_batches, n_map_failed):
    companies = {
        row_by_id[rid].get("Company_Clean") or row_by_id[rid].get("Company")
        for rid in union_ids
        if rid in row_by_id
    }
    companies = {c for c in companies if c}
    announced = sum(
        1 for rid in union_ids
        if rid in row_by_id and normalize_bool(row_by_id[rid].get("Is_Announcement"))
    )
    return {
        "n_batches": n_batches,
        "n_map_failed": n_map_failed,
        "union_size": len(union_ids),
        "union_distinct_companies": len(companies),
        "union_announced_count": announced,
        "selection_rate": round(len(union_ids) / total_rows, 4) if total_rows else 0.0,
    }


def run_model(model, questions, batches, row_json_by_id, row_by_id, total_rows, args):
    safe = safe_model_name(model)
    out_dir = Path(args.out)
    raw_json_dir = out_dir / "raw_json"
    logs_dir = out_dir / "logs"
    raw_json_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_json_dir / f"{safe}.jsonl"
    log_path = logs_dir / f"{safe}.log"

    log_lines = []

    def log(message):
        print(message)
        log_lines.append(message)
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    config = {
        "batch_size": args.batch_size,
        "overlap": 0,
        "map_num_ctx": args.map_num_ctx,
        "reduce_num_ctx": args.reduce_num_ctx,
        "temperature": args.temperature,
        "sort_key": args.sort_key,
    }

    log(f"=== MODEL {model} | arm=batchwise_map_reduce | batches={len(batches)} ===")

    with output_path.open("w", encoding="utf-8") as out:
        for index, q in enumerate(questions, start=1):
            q_id = q["q_id"]
            question = q["question"]
            log(f"\n--- Q{q_id} ({index}/{len(questions)}) --- {question}")

            map_start = time.time()
            map_results = []
            for batch in batches:
                result = run_map(model, question, batch, args)
                map_results.append(result)
                log(
                    f"MAP {batch['batch_id']}: {result['map_status']} "
                    f"ids={len(result['relevant_record_ids'])} "
                    f"attempts={result.get('attempts')}"
                )
            map_seconds = time.time() - map_start

            union_set = set()
            for result in map_results:
                union_set.update(result["relevant_record_ids"])
            union_ids = sorted(union_set, key=natural_sort_key)
            candidate_rows_jsonl = "\n".join(row_json_by_id[rid] for rid in union_ids)

            verify_start = time.time()
            verification = run_verify(model, question, union_ids, candidate_rows_jsonl, args)
            verify_seconds = time.time() - verify_start
            verified_ids = verification["verified_row_ids"]
            relevant_rows_jsonl = "\n".join(row_json_by_id[rid] for rid in verified_ids)
            log(
                f"VERIFY {verification['verify_status']}: "
                f"candidate={len(union_ids)} verified={len(verified_ids)} "
                f"removed={len(verification['removed_row_ids'])}"
            )

            reduce_start = time.time()
            reduce_error = ""
            try:
                reduce_res = run_reduce(model, question, verified_ids, relevant_rows_jsonl, args)
                reduce_answer = reduce_res["answer"]
                reduce_status = "ok"
            except Exception as exc:
                reduce_res = {}
                reduce_answer = ""
                reduce_status = "failed"
                reduce_error = f"{type(exc).__name__}: {exc}"
            reduce_seconds = time.time() - reduce_start

            n_map_failed = sum(1 for r in map_results if r["map_status"] == "failed")
            record = {
                "q_id": q_id,
                "question": question,
                "model": model,
                "arm": "batchwise_map_reduce",
                "map": map_results,
                "candidate_record_ids": union_ids,
                "verification": verification,
                "union_record_ids": verified_ids,
                "verified_record_ids": verified_ids,
                "reduce_answer": reduce_answer,
                "reduce_status": reduce_status,
                "reduce_error": reduce_error,
                "diagnostics": diagnostics(
                    union_ids=verified_ids,
                    row_by_id=row_by_id,
                    total_rows=total_rows,
                    n_batches=len(batches),
                    n_map_failed=n_map_failed,
                ),
                "timing": {
                    "map_seconds": round(map_seconds, 3),
                    "verify_seconds": round(verify_seconds, 3),
                    "reduce_seconds": round(reduce_seconds, 3),
                },
                "config": config,
                "ollama": {
                    "reduce_prompt_eval_count": reduce_res.get("prompt_eval_count"),
                    "reduce_eval_count": reduce_res.get("eval_count"),
                    "verify_prompt_eval_count": verification.get("prompt_eval_count"),
                    "verify_eval_count": verification.get("eval_count"),
                },
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            log(
                f"REDUCE {reduce_status}: union={len(verified_ids)} "
                f"map_failed={n_map_failed} chars={len(reduce_answer)}"
            )

    log(f"\nWrote {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch-wise LLM map-reduce arm for Georgia EV 42q evaluation"
    )
    parser.add_argument("--kb", default=DEFAULT_KB, help="Excel KB path")
    parser.add_argument("--sheet", default=None, help="Excel sheet name; default first sheet")
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS, help="questions .txt or .jsonl")
    parser.add_argument("--models", default=DEFAULT_MODELS, help="newline-delimited model file")
    parser.add_argument("--out", default=DEFAULT_OUT, help="output directory")
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--map_num_ctx", type=int, default=16384)
    parser.add_argument("--reduce_num_ctx", type=int, default=40960)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_map_retries", type=int, default=2)
    parser.add_argument("--sort_key", default="Record_ID")
    parser.add_argument("--limit", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch_size must be positive")

    records, row_json_by_id, row_by_id, id_col = load_kb(args.kb, args.sheet, args.sort_key)
    batches = make_batches(records, args.batch_size)
    questions = read_questions(args.questions, args.limit)
    models = read_models(args.models)

    print(f"KB rows: {len(records)} | id column: {id_col} | batches: {len(batches)}")
    print(f"Questions: {len(questions)}")
    print(f"Models: {models}")
    print(f"Output: {args.out}")

    if len(questions) != 42:
        print(f"WARNING: expected 42 questions, loaded {len(questions)}")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    config_path = Path(args.out) / "run_config.json"
    config_path.write_text(
        json.dumps({
            "kb": args.kb,
            "sheet": args.sheet,
            "questions": args.questions,
            "models": args.models,
            "batch_size": args.batch_size,
            "overlap": 0,
            "map_num_ctx": args.map_num_ctx,
            "reduce_num_ctx": args.reduce_num_ctx,
            "temperature": args.temperature,
            "max_map_retries": args.max_map_retries,
            "sort_key": args.sort_key,
            "resolved_id_column": id_col,
            "kb_rows": len(records),
            "n_batches": len(batches),
        }, indent=2),
        encoding="utf-8",
    )

    for model in models:
        run_model(
            model=model,
            questions=questions,
            batches=batches,
            row_json_by_id=row_json_by_id,
            row_by_id=row_by_id,
            total_rows=len(records),
            args=args,
        )

    print("\nDone. Batchwise JSONL outputs written.")


if __name__ == "__main__":
    main()
