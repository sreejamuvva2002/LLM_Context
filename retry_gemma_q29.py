import json
import time
from pathlib import Path

import pandas as pd
import requests

from run_ollama_jsonl_eval import (
    excel_to_jsonl,
    read_questions,
    build_prompt,
)

MODEL = "gemma3:27b"
NUM_CTX = 65536
QUESTION_NUMBER = 29
TIMEOUT_SECONDS = 3600

CSV_FILE = "ollama_outputs/answers_gemma3_27b.csv"
OLLAMA_URL = "http://localhost:11434/api/generate"


def call_ollama_long_timeout(model, prompt, num_ctx):
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
        timeout=TIMEOUT_SECONDS
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


jsonl_kb, records, columns = excel_to_jsonl("GNEM_Excel_Data.xlsx")
questions = read_questions("questions.txt")

question = questions[QUESTION_NUMBER - 1]

print(f"Retrying Gemma Q{QUESTION_NUMBER}")
print(question)
print(f"Rows in KB: {len(records)}")
print(f"Columns: {len(columns)}")
print(f"num_ctx: {NUM_CTX}")
print(f"Timeout: {TIMEOUT_SECONDS} seconds")

prompt = build_prompt(jsonl_kb, question)

start = time.time()
result = call_ollama_long_timeout(MODEL, prompt, NUM_CTX)
elapsed = time.time() - start

print(f"\nFinished retry in {elapsed:.1f} seconds")
print("\nAnswer preview:")
print(result["answer"][:1500])

df = pd.read_csv(CSV_FILE)

row_index = df.index[df["question_number"] == QUESTION_NUMBER]

if len(row_index) != 1:
    raise ValueError(f"Could not find exactly one row for question_number={QUESTION_NUMBER}")

idx = row_index[0]

df.loc[idx, "answer"] = result["answer"]
df.loc[idx, "prompt_eval_count"] = result["prompt_eval_count"]
df.loc[idx, "eval_count"] = result["eval_count"]
df.loc[idx, "total_duration"] = result["total_duration"]
df.loc[idx, "load_duration"] = result["load_duration"]
df.loc[idx, "prompt_eval_duration"] = result["prompt_eval_duration"]
df.loc[idx, "eval_duration"] = result["eval_duration"]

df.to_csv(CSV_FILE, index=False)

print(f"\nPatched Q{QUESTION_NUMBER} in {CSV_FILE}")
