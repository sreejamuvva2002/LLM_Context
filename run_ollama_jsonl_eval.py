import argparse
import csv
import json
import time
from pathlib import Path

import pandas as pd
import requests


OLLAMA_URL = "http://localhost:11434/api/generate"

XLSX_FILE = "GNEM_Excel_Data.xlsx"
QUESTIONS_FILE = "questions.txt"
OUTPUT_DIR = "ollama_outputs"


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


ANSWER_PROMPT = """
You are a precise data-analysis assistant.

You have been given a JSONL knowledge base about electric-vehicle EV supply-chain companies and one question.

GROUNDING RULES:
- The JSONL knowledge base is the single source of truth.
- Use ONLY the JSONL knowledge base.
- Do NOT use prior knowledge or background knowledge.
- If your prior knowledge and the JSONL disagree, the JSONL wins.
- If a question cannot be answered from the JSONL, reply exactly: Not found in the data.
- Do not guess, infer, or fill gaps.
- Only combine values that are explicitly present in the JSONL.

HOW TO ANSWER:
- COUNT or LIST questions: FIRST list every matching company by name, THEN give the count.
- The count must equal the number of names you listed.
- When a question has multiple conditions, apply all filters together using AND.
- LOOKUP or SUMMARY questions: answer directly from the relevant row or rows.
- Reproduce names, numbers, and units exactly as they appear in the data.

OUTPUT FORMAT:
Q: <brief restatement of the question>
Answer: <the answer, or "Not found in the data.">
Supporting companies/rows: <the company names and row_id values you used>
Notes: <only if there is a tie, ambiguity, or definitional issue; else leave blank>
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


def excel_to_jsonl(xlsx_file):
    df = pd.read_excel(xlsx_file)
    df = df.dropna(how="all")
    df.columns = [str(c).strip() for c in df.columns]

    records = []

    for idx, row in df.iterrows():
        record = {"row_id": int(idx + 1)}

        for col in df.columns:
            record[col] = clean_value(row[col])

        records.append(record)

    jsonl_text = "\n".join(
        json.dumps(record, ensure_ascii=False)
        for record in records
    )

    return jsonl_text, records, list(df.columns)


def read_questions(path):
    questions = []

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            questions.append(line)

    return questions


def build_prompt(jsonl_kb, question):
    return f"""
{ANSWER_PROMPT}

{DATA_DICTIONARY}

JSONL KNOWLEDGE BASE:
{jsonl_kb}

QUESTION:
{question}
""".strip()


def call_ollama(model, prompt, num_ctx):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_ctx": num_ctx
        }
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=1200
    )

    response.raise_for_status()
    data = response.json()

    return {
        "answer": data.get("response", "").strip(),
        "prompt_eval_count": data.get("prompt_eval_count"),
        "eval_count": data.get("eval_count"),
        "total_duration": data.get("total_duration"),
        "load_duration": data.get("load_duration"),
        "prompt_eval_duration": data.get("prompt_eval_duration"),
        "eval_duration": data.get("eval_duration"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--num_ctx", type=int, default=65536)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    jsonl_kb, records, columns = excel_to_jsonl(XLSX_FILE)
    questions = read_questions(QUESTIONS_FILE)

    questions = questions[:args.limit]

    Path(f"{OUTPUT_DIR}/kb_full.jsonl").write_text(jsonl_kb, encoding="utf-8")

    approx_kb_tokens = len(jsonl_kb) // 4

    print("Excel converted to JSONL.")
    print(f"Rows: {len(records)}")
    print(f"Columns: {len(columns)}")
    print(f"Questions to run: {len(questions)}")
    print(f"Approx KB tokens: {approx_kb_tokens}")
    print(f"Model: {args.model}")
    print(f"num_ctx: {args.num_ctx}")
    print(f"JSONL saved to: {OUTPUT_DIR}/kb_full.jsonl")

    output_csv = f"{OUTPUT_DIR}/answers_{args.model.replace(':', '_').replace('/', '_')}.csv"

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question_number",
                "question",
                "answer",
                "model",
                "num_ctx",
                "prompt_eval_count",
                "eval_count",
                "total_duration",
                "load_duration",
                "prompt_eval_duration",
                "eval_duration",
            ],
        )

        writer.writeheader()

        for q_num, question in enumerate(questions, start=1):
            print("\n" + "-" * 80)
            print(f"Running Q{q_num}")
            print(question)

            prompt = build_prompt(jsonl_kb, question)

            start = time.time()

            try:
                result = call_ollama(args.model, prompt, args.num_ctx)
                elapsed = time.time() - start

                print(f"Finished Q{q_num} in {elapsed:.1f} seconds")
                print("\nAnswer preview:")
                print(result["answer"][:1000])

                writer.writerow({
                    "question_number": q_num,
                    "question": question,
                    "answer": result["answer"],
                    "model": args.model,
                    "num_ctx": args.num_ctx,
                    "prompt_eval_count": result["prompt_eval_count"],
                    "eval_count": result["eval_count"],
                    "total_duration": result["total_duration"],
                    "load_duration": result["load_duration"],
                    "prompt_eval_duration": result["prompt_eval_duration"],
                    "eval_duration": result["eval_duration"],
                })

            except Exception as e:
                print(f"ERROR on Q{q_num}: {e}")

                writer.writerow({
                    "question_number": q_num,
                    "question": question,
                    "answer": f"ERROR: {e}",
                    "model": args.model,
                    "num_ctx": args.num_ctx,
                    "prompt_eval_count": None,
                    "eval_count": None,
                    "total_duration": None,
                    "load_duration": None,
                    "prompt_eval_duration": None,
                    "eval_duration": None,
                })

    print("\nDone.")
    print(f"Output saved to: {output_csv}")


if __name__ == "__main__":
    main()
