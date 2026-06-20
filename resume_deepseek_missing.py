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

MODEL = "deepseek-r1:32b"
NUM_CTX = 65536
NUM_PREDICT = 2048
CSV_FILE = "ollama_outputs/answers_deepseek-r1_32b.csv"
OLLAMA_URL = "http://localhost:11434/api/generate"

FIELDNAMES = [
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
]

DEEPSEEK_RULE = """
IMPORTANT OUTPUT RULE FOR DEEPSEEK:
Return the final answer only.
Do not include reasoning, hidden chain-of-thought, analysis, or <think> tags.
Follow the required Q / Answer / Supporting companies / Notes format.
Keep the answer concise but complete.
"""


def call_ollama_stream(model, prompt):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
        },
    }

    full_answer = ""
    final_data = {}

    with requests.post(
        OLLAMA_URL,
        json=payload,
        stream=True,
        timeout=(10, 900),
    ) as response:
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue

            data = json.loads(line.decode("utf-8"))

            chunk = data.get("response", "")
            if chunk:
                print(chunk, end="", flush=True)
                full_answer += chunk

            if data.get("done", False):
                final_data = data
                break

    return {
        "answer": full_answer.strip(),
        "prompt_eval_count": final_data.get("prompt_eval_count"),
        "eval_count": final_data.get("eval_count"),
        "total_duration": final_data.get("total_duration"),
        "load_duration": final_data.get("load_duration"),
        "prompt_eval_duration": final_data.get("prompt_eval_duration"),
        "eval_duration": final_data.get("eval_duration"),
    }


def load_existing():
    path = Path(CSV_FILE)

    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=FIELDNAMES)

    for col in FIELDNAMES:
        if col not in df.columns:
            df[col] = None

    return df[FIELDNAMES]


def save_row(df, row):
    qn = row["question_number"]

    if qn in set(df["question_number"].dropna().astype(int)):
        idx = df.index[df["question_number"].astype(int) == qn][0]
        for key, value in row.items():
            df.loc[idx, key] = value
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df = df.sort_values("question_number").reset_index(drop=True)
    df.to_csv(CSV_FILE, index=False)
    return df


jsonl_kb, records, columns = excel_to_jsonl("GNEM_Excel_Data.xlsx")
questions = read_questions("questions.txt")

df = load_existing()

existing_qnums = set(df["question_number"].dropna().astype(int))
all_qnums = set(range(1, len(questions) + 1))

missing_qnums = sorted(all_qnums - existing_qnums)

error_qnums = sorted(
    df[
        df["answer"].astype(str).str.startswith("ERROR", na=False)
    ]["question_number"].astype(int).tolist()
)

pending_qnums = sorted(set(missing_qnums + error_qnums))

print("DeepSeek resume run")
print(f"Existing rows: {len(df)}")
print(f"Missing questions: {missing_qnums}")
print(f"Error questions: {error_qnums}")
print(f"Pending questions to run: {pending_qnums}")
print(f"Model: {MODEL}")
print(f"num_ctx: {NUM_CTX}")
print(f"num_predict: {NUM_PREDICT}")

if not pending_qnums:
    print("No missing or error questions found.")
    raise SystemExit

for qn in pending_qnums:
    question = questions[qn - 1]

    print("\n" + "=" * 100)
    print(f"Running Q{qn}: {question}")
    print("=" * 100)

    prompt = build_prompt(jsonl_kb, question) + "\n\n" + DEEPSEEK_RULE

    start = time.time()

    try:
        result = call_ollama_stream(MODEL, prompt)
        elapsed = time.time() - start

        print("\n\nFinished Q{} in {:.1f} seconds".format(qn, elapsed))

        answer = result["answer"]
        if not answer:
            answer = "ERROR: Empty response from model."

        row = {
            "question_number": qn,
            "question": question,
            "answer": answer,
            "model": MODEL,
            "num_ctx": NUM_CTX,
            "prompt_eval_count": result["prompt_eval_count"],
            "eval_count": result["eval_count"],
            "total_duration": result["total_duration"],
            "load_duration": result["load_duration"],
            "prompt_eval_duration": result["prompt_eval_duration"],
            "eval_duration": result["eval_duration"],
        }

    except Exception as e:
        elapsed = time.time() - start
        print(f"\nERROR on Q{qn} after {elapsed:.1f} seconds: {e}")

        row = {
            "question_number": qn,
            "question": question,
            "answer": f"ERROR: {e}",
            "model": MODEL,
            "num_ctx": NUM_CTX,
            "prompt_eval_count": None,
            "eval_count": None,
            "total_duration": None,
            "load_duration": None,
            "prompt_eval_duration": None,
            "eval_duration": None,
        }

    df = save_row(df, row)
    print(f"Saved progress to {CSV_FILE}")

print("\nDeepSeek resume complete.")
