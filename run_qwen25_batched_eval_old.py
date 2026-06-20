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

Methodology:
This dataset compiles automotive OEMs and suppliers with 50 or more employees operating in Georgia.
Company inclusion was based on public state economic development listings, industry directories, and OEM announcements.
Supply chain tiers reflect functional roles: OEM, Tier 1, Tier 2, and Tier 3, and are assigned using a structured heuristic approach.
EV and battery relevance indicates whether a company directly produces electric vehicles, batteries, charging equipment, or supplies enabling components.
OEM linkage identifies primary relationships where publicly known; otherwise companies are marked as supporting multiple OEMs.
Primary OEMs identifies known or inferred original equipment manufacturers served by each company based on product type, public announcements, and supply chain role.
Supplier or Affiliation Type classifies companies by their functional relationship to OEMs including direct supply, component supply, upstream materials, or OEM status.
Location is reported as City and County.
EV Supply Chain Role classifies each company based on its primary functional contribution to electric vehicle systems using a controlled vocabulary.
Primary Facility Type identifies the dominant operational role of the Georgia site including manufacturing, headquarters, research and development, or distribution.

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
- Treat all categorical and coded values exactly as written in the JSONL data.
"""


BATCH_ANALYSIS_PROMPT = """
You are analyzing ONE BATCH of a larger JSONL knowledge base.

Your job:
- Use ONLY the JSONL rows in this batch.
- Do NOT use prior knowledge.
- Do NOT answer from memory.
- Analyze whether this batch contains information relevant to the question.
- Return ONLY valid JSON.

Important:
- If the question asks for a LIST, return all matching rows from this batch.
- If the question asks for a COUNT, return the matching rows from this batch and a batch_count.
- If the question asks for a RANKING, HIGHEST, LOWEST, TOTAL, or AGGREGATION, compute only the partial result for this batch.
- If no rows in this batch are relevant, return an empty relevant_rows list.
- Do not split labels like "Tier 1/2" or "Tier 2/3".

Return JSON in this exact structure:
{
  "batch_id": <number>,
  "question_type": "<list/count/lookup/summary/aggregation/mixed>",
  "relevant_rows": [
    {
      "row_id": <row_id>,
      "Company": "<company name>",
      "Category": "<category>",
      "Location": "<location>",
      "EV Supply Chain Role": "<role>",
      "Primary OEMs": "<primary oems>",
      "Supplier or Affiliation Type": "<type>",
      "Employment": "<employment>",
      "Product / Service": "<product/service>",
      "reason": "<why this row is relevant>"
    }
  ],
  "batch_count": <number>,
  "partial_aggregates": [
    {
      "group": "<county/category/role/oem/etc>",
      "value": <number or string>,
      "explanation": "<short explanation>"
    }
  ],
  "batch_notes": "<short notes or empty string>"
}
"""


FINAL_ANSWER_PROMPT = """
You are a precise data-analysis assistant.

You have been given:
1. The original user question.
2. Batch-level analyses produced by scanning every row of the KB in batches.

Your job:
- Produce the final answer using ONLY the batch analyses.
- Do NOT use prior knowledge.
- Do NOT guess.
- If the batch analyses do not contain enough information, answer exactly: Not found in the data.
- For COUNT or LIST questions: first list every matching company by name, then give the count.
- The count must equal the number of names listed.
- For aggregation questions, combine the partial aggregates carefully.
- Treat "Tier 1/2" and "Tier 2/3" as exact single categories.
- Preserve company names, numbers, and units exactly as given.

OUTPUT FORMAT:
Q: <brief restatement of the question>
Answer: <the answer, or "Not found in the data.">
Supporting companies/rows: <company names and row_id values used>
Notes: <only if there is a tie, ambiguity, batching issue, or definitional issue; else leave blank>
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


def extract_json_object(text):
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


def call_ollama(prompt, timeout, json_mode=False):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "seed": SEED,
            "num_ctx": NUM_CTX
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
        "load_duration": data.get("load_duration"),
        "prompt_eval_duration": data.get("prompt_eval_duration"),
        "eval_duration": data.get("eval_duration"),
    }


def analyze_batch(question, batch_id, batch_records):
    batch_jsonl = records_to_jsonl(batch_records)

    prompt = f"""
{BATCH_ANALYSIS_PROMPT}

{DATA_DICTIONARY}

QUESTION:
{question}

BATCH_ID:
{batch_id}

JSONL ROWS IN THIS BATCH:
{batch_jsonl}
""".strip()

    result = call_ollama(prompt, timeout=BATCH_TIMEOUT, json_mode=True)
    parsed = extract_json_object(result["response"])

    if parsed is None:
        parsed = {
            "batch_id": batch_id,
            "question_type": "parse_error",
            "relevant_rows": [],
            "batch_count": 0,
            "partial_aggregates": [],
            "batch_notes": "Model did not return parseable JSON.",
            "raw_response": result["response"]
        }

    if "batch_id" not in parsed:
        parsed["batch_id"] = batch_id

    return parsed, result


def synthesize_final_answer(question, batch_analyses):
    analyses_json = json.dumps(batch_analyses, ensure_ascii=False, indent=2)

    prompt = f"""
{FINAL_ANSWER_PROMPT}

ORIGINAL QUESTION:
{question}

BATCH ANALYSES FROM COMPLETE KB SCAN:
{analyses_json}
""".strip()

    result = call_ollama(prompt, timeout=FINAL_TIMEOUT, json_mode=False)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=25)
    args = parser.parse_args()

    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    records, columns = load_records(XLSX_FILE)
    questions = read_questions(QUESTIONS_FILE)[:args.limit]

    output_csv = f"{OUTPUT_DIR}/answers_qwen2.5_14b_batched.csv"
    audit_jsonl = f"{OUTPUT_DIR}/audit_qwen2.5_14b_batched.jsonl"

    print("Qwen2.5 batched evaluation")
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
                "num_ctx",
                "batch_errors",
                "final_prompt_eval_count",
                "final_eval_count",
                "final_total_duration",
            ]
        )
        writer.writeheader()

        for q_num, question in enumerate(questions, start=1):
            print("\n" + "=" * 90)
            print(f"Q{q_num}: {question}")
            print("=" * 90)

            batch_analyses = []
            batch_errors = 0

            for batch_id, batch_records in chunks(records, args.batch_size):
                print(f"Analyzing batch {batch_id} ({len(batch_records)} rows)...")

                try:
                    parsed, raw_result = analyze_batch(question, batch_id, batch_records)
                    batch_analyses.append(parsed)

                    audit_file.write(json.dumps({
                        "question_number": q_num,
                        "question": question,
                        "batch_id": batch_id,
                        "parsed": parsed,
                        "raw_result": raw_result
                    }, ensure_ascii=False) + "\n")
                    audit_file.flush()

                    print(f"  relevant_rows: {len(parsed.get('relevant_rows', []))}")
                    print(f"  batch_count: {parsed.get('batch_count')}")

                except Exception as e:
                    batch_errors += 1
                    error_obj = {
                        "batch_id": batch_id,
                        "question_type": "error",
                        "relevant_rows": [],
                        "batch_count": 0,
                        "partial_aggregates": [],
                        "batch_notes": f"ERROR: {e}"
                    }
                    batch_analyses.append(error_obj)

                    audit_file.write(json.dumps({
                        "question_number": q_num,
                        "question": question,
                        "batch_id": batch_id,
                        "error": str(e)
                    }, ensure_ascii=False) + "\n")
                    audit_file.flush()

                    print(f"  ERROR: {e}")

            print("Synthesizing final answer...")

            try:
                final_result = synthesize_final_answer(question, batch_analyses)
                final_answer = final_result["response"]

                print("Final answer preview:")
                print(final_answer[:1000])

                writer.writerow({
                    "question_number": q_num,
                    "question": question,
                    "answer": final_answer,
                    "model": MODEL,
                    "method": "Exhaustive Batched LLM Scanning with Final Answer Synthesis",
                    "batch_size": args.batch_size,
                    "num_batches": len(batch_analyses),
                    "num_ctx": NUM_CTX,
                    "batch_errors": batch_errors,
                    "final_prompt_eval_count": final_result["prompt_eval_count"],
                    "final_eval_count": final_result["eval_count"],
                    "final_total_duration": final_result["total_duration"],
                })
                csv_file.flush()

            except Exception as e:
                print(f"FINAL SYNTHESIS ERROR: {e}")
                writer.writerow({
                    "question_number": q_num,
                    "question": question,
                    "answer": f"ERROR: {e}",
                    "model": MODEL,
                    "method": "Exhaustive Batched LLM Scanning with Final Answer Synthesis",
                    "batch_size": args.batch_size,
                    "num_batches": len(batch_analyses),
                    "num_ctx": NUM_CTX,
                    "batch_errors": batch_errors,
                    "final_prompt_eval_count": None,
                    "final_eval_count": None,
                    "final_total_duration": None,
                })
                csv_file.flush()

    print("\nDone.")
    print(f"Saved answers to: {output_csv}")
    print(f"Saved audit log to: {audit_jsonl}")


if __name__ == "__main__":
    main()
