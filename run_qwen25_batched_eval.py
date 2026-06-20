import argparse
import csv
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests


MODEL = "qwen2.5:14b"
OLLAMA_URL = "http://localhost:11434/api/generate"

XLSX_FILE = "GNEM_Excel_Data.xlsx"
QUESTIONS_FILE = "questions.txt"
OUTPUT_DIR = "ollama_outputs"

NUM_CTX = 32768
TEMPERATURE = 0
SEED = 42

BATCH_TIMEOUT = 600
FINAL_TIMEOUT = 900


DATA_DICTIONARY = """
DATA DICTIONARY AND DEFINITIONS

Definitions:
- OEM = Original Equipment Manufacturer responsible for final vehicle or system production.
- Tier 1 = Supplier directly supplies systems or components to OEMs.
- Tier 2 = Supplier provides components or materials to Tier 1 suppliers.
- Tier 3 = Supplier provides raw materials or upstream processing.
- EV Battery Relevant = direct or enabling involvement in electric mobility supply chains.
- Category = OEM when the company is an original equipment manufacturer; otherwise it reports the supply chain tier.
- Primary OEMs = one or more OEMs that a company primarily supplies or is affiliated with.
- Supplier or Affiliation Type = how a company participates in the automotive value chain.
- Location = City and County in one field.
- EV Supply Chain Role = vehicle assembly, battery cell, battery pack, power electronics, charging infrastructure, thermal management, materials, or general automotive.
- Primary Facility Type = dominant function of the company’s Georgia site.

Critical category rules:
- Treat "Tier 1/2" as ONE exact category.
- Do NOT split "Tier 1/2" into Tier 1 or Tier 2.
- Treat "Tier 2/3" as ONE exact category.
- Do NOT split "Tier 2/3" into Tier 2 or Tier 3.
- If a question contains a quoted category like "Tier 1/2", include ONLY rows where Category exactly equals that quoted value.
- Apply ALL conditions in the question together using AND.
- Do not include rows that only partially match.
"""


BATCH_PROMPT = """
You are scanning ONE SMALL BATCH of a larger JSONL knowledge base.

Your task is to decide which rows in THIS batch match the user question.

IMPORTANT REASONING RULE:
Separate FILTER CONDITIONS from OUTPUT FIELDS.

Example:
If the question says:
"Show all Tier 1/2 suppliers and list their EV Supply Chain Role and Product / Service"

Then:
- FILTER CONDITION = Category exactly equals "Tier 1/2"
- OUTPUT FIELDS = EV Supply Chain Role and Product / Service
Do NOT use EV Supply Chain Role or Product / Service as filters unless the question explicitly asks to filter by them.

Rules:
- Use ONLY the rows in this batch.
- Do NOT use prior knowledge.
- Apply all explicit filter conditions exactly.
- If the question contains "Tier 1/2", include every row where Category exactly equals "Tier 1/2".
- Do NOT split "Tier 1/2" into Tier 1 or Tier 2.
- If the question contains "Tier 2/3", include every row where Category exactly equals "Tier 2/3".
- Do NOT exclude a matching row because its EV Supply Chain Role is General Automotive.
- Do NOT exclude a matching row because its Product / Service looks less EV-specific.
- If a row matches the filter, include it.
- If no rows match, return an empty matched_rows list.
- Return ONLY valid JSON.

Return this exact JSON format:
{
  "batch_id": <batch_id>,
  "filter_interpretation": "<state the exact filters you applied>",
  "matched_rows": [
    {
      "row_id": <row_id>,
      "Company": "<company>",
      "Category": "<category>",
      "EV Supply Chain Role": "<role>",
      "Product / Service": "<product/service>",
      "why_included": "<short reason>"
    }
  ],
  "matched_row_ids": [<row_id_1>, <row_id_2>],
  "batch_notes": ""
}
"""


FINAL_PROMPT = """
You are a precise data-transcription assistant.

You are given:
1. The original question.
2. REQUIRED_ROW_COUNT: how many rows your answer table MUST contain.
3. REQUIRED_ROW_IDS: the exact set of row_id values that MUST each appear once.
4. SELECTED ROWS: the rows an exhaustive batch scan already selected.

This is a TRANSCRIPTION task, NOT a filtering task.
The selection has ALREADY been done for you. You must NOT re-filter, re-judge,
re-evaluate, summarize, merge, deduplicate, or drop any row. Every selected row
is correct and final. Your only job is to lay them out in the output table.

ABSOLUTE RULES:
- Do NOT use prior knowledge.
- Do NOT add any row_id that is not in REQUIRED_ROW_IDS.
- Do NOT omit any row_id that is in REQUIRED_ROW_IDS.
- Output EXACTLY one table row for EVERY row_id in REQUIRED_ROW_IDS, in the same order.
- The number of table rows MUST equal REQUIRED_ROW_COUNT.
- Do NOT collapse, merge, or skip rows even if Company / role / product look similar
  or appear duplicated. Similar-looking rows are still separate rows.
- Preserve Company, EV Supply Chain Role, and Product / Service exactly as shown.
- Use the compact table below. No bullet lists, no commentary, no per-row explanations.
- If REQUIRED_ROW_COUNT is 0, answer exactly: Not found in the data.

MANDATORY SELF-CHECK before you output (do this silently, do not print it):
- Go down REQUIRED_ROW_IDS one id at a time and confirm each id has its own table row.
- Count your table rows. If the count does not equal REQUIRED_ROW_COUNT, you missed
  rows: add the missing row_ids before producing the final output.
- Set Count to the number of table rows. Count MUST equal REQUIRED_ROW_COUNT.

OUTPUT FORMAT (and nothing else):
Q: <brief restatement of the question>
Answer:
| row_id | Company | EV Supply Chain Role | Product / Service |
|---|---|---|---|
| ... | ... | ... | ... |

Count: <number of table rows; MUST equal REQUIRED_ROW_COUNT>

Supporting row_ids: <comma-separated list of every row_id in the table>
Notes: <blank unless there is ambiguity>
"""

def clean_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return value
    if isinstance(value, str):
        value = value.strip()
        return value if value else None
    return value


def load_records(xlsx_file):
    df = pd.read_excel(xlsx_file)
    df = df.dropna(how="all")
    df.columns = [str(c).strip() for c in df.columns]

    records = []
    for idx, row in df.iterrows():
        record = {"row_id": int(idx + 1)}
        for col in df.columns:
            record[col] = clean_value(row[col])
        records.append(record)

    return records, list(df.columns)


def read_questions(path):
    questions = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            questions.append(line)
    return questions


def chunks(items, size):
    for i in range(0, len(items), size):
        yield i // size + 1, items[i:i + size]


def records_to_jsonl(records):
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in records)


def extract_json(text):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return None


def call_ollama(prompt, timeout, json_mode=False, num_predict=3000):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "seed": SEED,
            "num_ctx": NUM_CTX,
            "num_predict": num_predict
        }
    }

    if json_mode:
        payload["format"] = "json"

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=timeout
    )
    response.raise_for_status()
    data = response.json()

    return {
     "response": data.get("response", "").strip(),
        "prompt_eval_count": data.get("prompt_eval_count"),
        "eval_count": data.get("eval_count"),
        "total_duration": data.get("total_duration"),
    }


def batch_scan(question, batch_id, batch_records):
    prompt = f"""
{BATCH_PROMPT}

{DATA_DICTIONARY}

QUESTION:
{question}

BATCH_ID:
{batch_id}

JSONL ROWS:
{records_to_jsonl(batch_records)}
""".strip()

    result = call_ollama(prompt, timeout=BATCH_TIMEOUT, json_mode=True)
    parsed = extract_json(result["response"])

    if parsed is None:
        return {
            "batch_id": batch_id,
            "matching_row_ids": [],
            "reason": "parse_error",
            "raw_response": result["response"]
        }, result

    ids = parsed.get("matching_row_ids")
    if ids is None:
        ids = parsed.get("matched_row_ids")
    if ids is None and isinstance(parsed.get("matched_rows"), list):
        ids = [row.get("row_id") for row in parsed.get("matched_rows", [])]
    if ids is None:
        ids = []
    clean_ids = []

    for x in ids:
        try:
            clean_ids.append(int(x))
        except Exception:
            pass

    parsed["matching_row_ids"] = clean_ids
    return parsed, result


def synthesize_answer(question, selected_rows):
    # Orchestration only: Python reports back HOW MANY rows and WHICH row_ids the
    # LLM already selected during batch scanning. Python does not decide the answer
    # content; it only hands the LLM its own selections as a transcription checklist.
    required_ids = [r["row_id"] for r in selected_rows]
    required_count = len(selected_rows)

    # One JSON object per line keeps the rows easy to transcribe one-for-one.
    rows_block = "\n".join(
        json.dumps(r, ensure_ascii=False) for r in selected_rows
    )

    prompt = f"""
{FINAL_PROMPT}

ORIGINAL QUESTION:
{question}

REQUIRED_ROW_COUNT:
{required_count}

REQUIRED_ROW_IDS (your table MUST contain exactly these, each exactly once):
{required_ids}

SELECTED ROWS (one JSON object per line; transcribe EVERY line into one table row):
{rows_block}
""".strip()

    # Give synthesis extra headroom so a large table can never be truncated.
    return call_ollama(
        prompt,
        timeout=FINAL_TIMEOUT,
        json_mode=False,
        num_predict=4096,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=25)
    args = parser.parse_args()

    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    records, columns = load_records(XLSX_FILE)
    questions = read_questions(QUESTIONS_FILE)[:args.limit]

    record_by_id = {r["row_id"]: r for r in records}

    output_csv = f"{OUTPUT_DIR}/answers_qwen2.5_14b_batched_strict.csv"
    audit_jsonl = f"{OUTPUT_DIR}/audit_qwen2.5_14b_batched_strict.jsonl"

    print("Qwen2.5 strict batched evaluation")
    print(f"Rows: {len(records)}")
    print(f"Columns: {len(columns)}")
    print(f"Questions: {len(questions)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Model: {MODEL}")
    print(f"num_ctx: {NUM_CTX}")
    print(f"Output CSV: {output_csv}")
    print(f"Audit JSONL: {audit_jsonl}")

    with open(output_csv, "w", newline="", encoding="utf-8") as csv_file, \
         open(audit_jsonl, "w", encoding="utf-8") as audit_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "question_number",
                "question",
                "answer",
                "model",
                "method",
                "batch_size",
                "num_batches",
                "selected_row_count",
                "selected_row_ids",
                "batch_errors"
            ]
        )
        writer.writeheader()

        for q_num, question in enumerate(questions, start=1):
            print("\n" + "=" * 90)
            print(f"Q{q_num}: {question}")
            print("=" * 90)

            selected_ids = []
            batch_errors = 0
            num_batches = 0

            for batch_id, batch_records in chunks(records, args.batch_size):
                num_batches += 1
                print(f"Scanning batch {batch_id} ({len(batch_records)} rows)...")

                try:
                    parsed, raw = batch_scan(question, batch_id, batch_records)
                    ids = parsed.get("matching_row_ids")
                    if ids is None:
                        ids = parsed.get("matched_row_ids")
                    if ids is None and isinstance(parsed.get("matched_rows"), list):
                        ids = [row.get("row_id") for row in parsed.get("matched_rows", [])]
                    if ids is None:
                        ids = []
                    selected_ids.extend(ids)

                    print(f"  matched ids: {ids}")

                    audit_file.write(json.dumps({
                        "question_number": q_num,
                        "question": question,
                        "batch_id": batch_id,
                        "parsed": parsed,
                        "raw": raw
                    }, ensure_ascii=False) + "\n")
                    audit_file.flush()

                except Exception as e:
                    batch_errors += 1
                    print(f"  ERROR: {e}")
                    audit_file.write(json.dumps({
                        "question_number": q_num,
                        "question": question,
                        "batch_id": batch_id,
                        "error": str(e)
                    }, ensure_ascii=False) + "\n")
                    audit_file.flush()

            selected_ids = sorted(set(selected_ids))
            selected_rows = [record_by_id[i] for i in selected_ids if i in record_by_id]

            print(f"Total selected rows: {len(selected_rows)}")
            print(f"Selected row_ids: {selected_ids}")

            try:
                final_result = synthesize_answer(question, selected_rows)
                answer = final_result["response"]

                print("Final answer preview:")
                print(answer[:1000])

                writer.writerow({
                    "question_number": q_num,
                    "question": question,
                    "answer": answer,
                    "model": MODEL,
                    "method": "Exhaustive Batched LLM Scanning with Strict Row-ID Extraction",
                    "batch_size": args.batch_size,
                    "num_batches": num_batches,
                    "selected_row_count": len(selected_rows),
                    "selected_row_ids": json.dumps(selected_ids),
                    "batch_errors": batch_errors
                })
                csv_file.flush()

            except Exception as e:
                print(f"FINAL ERROR: {e}")
                writer.writerow({
                    "question_number": q_num,
                    "question": question,
                    "answer": f"ERROR: {e}",
                    "model": MODEL,
                    "method": "Exhaustive Batched LLM Scanning with Strict Row-ID Extraction",
                    "batch_size": args.batch_size,
                    "num_batches": num_batches,
                    "selected_row_count": len(selected_rows),
                    "selected_row_ids": json.dumps(selected_ids),
                    "batch_errors": batch_errors
                })
                csv_file.flush()

    print("\nDone.")
    print(f"Saved answers to: {output_csv}")
    print(f"Saved audit log to: {audit_jsonl}")


if __name__ == "__main__":
    main()
