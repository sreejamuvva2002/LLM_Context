#!/usr/bin/env python3
"""
Evaluate Georgia EV 42-question model responses from both experimental arms.

This is an evaluation-only script. It never feeds golden answers into a model.

Primary scoring is deterministic:
  - company/county entity precision, recall, F1 against human-validated answers
  - count correctness and signed count error
  - field-value coverage for requested fields
  - format compliance and row_id consistency
  - reliability/runtime/context diagnostics

Interpretive/risk-style questions are still scored deterministically as a
rough diagnostic, but are flagged for the second evaluation type described in
the report: optional external LLM/human judge review.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd


DEFAULT_KB = "Data/GNEM_Excel_Data.xlsx"
DEFAULT_GOLDEN = "Data/Human validated questions.xlsx"
DEFAULT_DIRECT_DIR = "results/direct_context/raw_json"
DEFAULT_BATCHWISE_DIR = "results/batchwise/raw_json"
DEFAULT_OUT = "evaluation"

SECTION_NAMES = ["Answer", "Evidence", "Count", "Supporting row_ids", "Notes"]

NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "twenty-one": 21,
    "twenty two": 22,
    "twenty-two": 22,
    "twenty three": 23,
    "twenty-three": 23,
    "twenty four": 24,
    "twenty-four": 24,
    "twenty five": 25,
    "twenty-five": 25,
    "twenty six": 26,
    "twenty-six": 26,
    "twenty seven": 27,
    "twenty-seven": 27,
    "twenty eight": 28,
    "twenty-eight": 28,
    "twenty nine": 29,
    "twenty-nine": 29,
    "thirty": 30,
}

COMPANY_SUFFIX_RE = re.compile(
    r"\b(inc|incorporated|llc|l\.l\.c|co|corp|corporation|ltd|company|"
    r"americas|america|usa|u\.s\.a|manufacturing|mfg|group|holdings?|gmbh)\b",
    re.I,
)

CANONICAL_SUFFIX_RE = re.compile(
    r"\b(inc|incorporated|llc|l\.l\.c|co|corp|corporation|ltd|company|holdings?|gmbh)\b",
    re.I,
)

UNKNOWN_COMPANY_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9&.'()/-]*(?:\s+[A-Z][A-Za-z0-9&.'()/-]*){0,7}\s+"
    r"(?:Inc\.?|LLC|L\.L\.C\.?|Corp\.?|Corporation|Co\.?|Company|Group|"
    r"Holdings?|Manufacturing|Systems|Industries|America|Americas|USA|U\.S\.A\.?))\b"
)

FIELD_KEYWORDS = [
    ("Category", ["tier", "category", "classified", "classification"]),
    ("EV Supply Chain Role", ["ev supply chain role", "role", "battery cell", "battery pack"]),
    ("Product / Service", ["product", "service", "produce", "producing", "manufacture"]),
    ("Primary OEMs", ["primary oem", "oem", "sole-sourced", "customer base", "rivian", "hyundai", "kia"]),
    ("Employment", ["employment", "employees", "worker", "workforce", "size"]),
    ("Primary Facility Type", ["facility type", "r&d", "research", "development", "distribution"]),
    ("Location", ["location", "area", "county", "where"]),
    ("Industry Group", ["industry group", "industry"]),
    ("EV / Battery Relevant", ["ev relevant", "battery relevant", "indirectly relevant"]),
]


def clean_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    if isinstance(value, str):
        value = value.strip()
        return value if value else None
    return value


def norm(text) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def norm_company(name) -> str:
    text = norm(name)
    text = COMPANY_SUFFIX_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def canonical_company_key(name) -> str:
    text = norm(name)
    text = CANONICAL_SUFFIX_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def number_value(token):
    if token is None:
        return None
    token = str(token).strip().lower().replace(",", "")
    if token.isdigit():
        return int(token)
    return NUMBER_WORDS.get(token)


def load_kb(path: str):
    df = pd.read_excel(path)
    df = df.dropna(how="all").reset_index(drop=True)
    df.columns = [str(c).strip() for c in df.columns]
    df.insert(0, "row_id", range(1, len(df) + 1))

    records = []
    raw_company_rows = defaultdict(list)
    counties = set()
    for _, row in df.iterrows():
        record = {col: clean_value(row[col]) for col in df.columns}
        records.append(record)
        company = str(record.get("Company") or "").strip()
        if company:
            raw_company_rows[company].append(record)
        location = str(record.get("Location") or "")
        if "," in location:
            county = location.split(",", 1)[1].strip()
            if county:
                counties.add(county)

    by_company = dict(raw_company_rows)

    primary_aliases = defaultdict(set)
    primary_alias_owners = defaultdict(set)
    loose_aliases = defaultdict(set)
    loose_alias_owners = defaultdict(set)
    for company in by_company:
        no_paren = re.sub(r"\([^)]*\)", "", company)
        for alias in {norm(company), norm(no_paren)}:
            if len(alias) >= 3:
                primary_aliases[company].add(alias)
                primary_alias_owners[alias].add(company)
        for alias in {norm_company(company), norm_company(no_paren)}:
            if len(alias) >= 4 and " " in alias:
                loose_aliases[company].add(alias)
                loose_alias_owners[alias].add(company)

    company_aliases = []
    for company in by_company:
        variants = set(primary_aliases[company])
        for alias in loose_aliases[company]:
            if len(loose_alias_owners[alias]) == 1 and not (primary_alias_owners[alias] - {company}):
                variants.add(alias)
        company_aliases.append((company, sorted(variants, key=len, reverse=True)))

    county_aliases = [(county, norm(county)) for county in counties]
    return records, by_company, company_aliases, county_aliases


def extract_companies(text: str, company_aliases):
    hay = norm(text)
    matches = []
    for company, aliases in company_aliases:
        for alias in aliases:
            if len(alias) < 3:
                continue
            pattern = rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])"
            for match in re.finditer(pattern, hay):
                matches.append((match.start(), match.end(), len(alias), company))

    hits = set()
    occupied = []
    for start, end, _, company in sorted(matches, key=lambda item: (item[0], -item[2])):
        if any(start < used_end and end > used_start for used_start, used_end in occupied):
            continue
        hits.add(company)
        occupied.append((start, end))
    return hits


def extract_counties(text: str, county_aliases):
    hay = f" {norm(text)} "
    hits = set()
    for county, alias in county_aliases:
        if alias and f" {alias} " in hay:
            hits.add(county)
    return hits


def extract_unknown_company_mentions(text: str, company_aliases):
    known = extract_companies(text, company_aliases)
    known_norms = {norm_company(k) for k in known} | {norm(k) for k in known}
    all_aliases = [alias for _, aliases in company_aliases for alias in aliases]
    stoplist = {
        "answer", "evidence", "count", "supporting row ids", "notes",
        "industry group", "category", "primary oems", "primary facility type",
        "ev supply chain role", "product service",
    }
    unknown = set()
    for match in UNKNOWN_COMPANY_RE.finditer(str(text or "")):
        candidate = match.group(1).strip(" .,:;|")
        c_norm = norm_company(candidate)
        if len(c_norm) < 4:
            continue
        if c_norm in stoplist or norm(candidate) in stoplist:
            continue
        if c_norm not in known_norms and norm(candidate) not in known_norms:
            if not extract_companies(candidate, company_aliases):
                if any(c_norm and c_norm in alias for alias in all_aliases):
                    continue
                unknown.add(candidate)
    return sorted(unknown)


def extract_golden_companies(golden_text: str, company_aliases):
    """Extract the answer-side company set from human golden text.

    Golden answers often use lines like:
      Company | Field: value
      Role: x | Only company: Company
      County: Company

    We intentionally avoid extracting company names from output fields such as
    "Primary OEMs: Hyundai Motor Group", because those are values to display,
    not necessarily the selected answer entities.
    """
    hits = set()
    for raw_line in str(golden_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^[\-*•\s]+", "", line).strip()
        low = norm(line)
        if low.startswith(("there are", "there is", "top ")):
            continue
        if low in {"oem footprint", "oem supply chain"}:
            continue
        if "appears in both categories" in low or "counted once" in low:
            continue

        candidate = line
        if re.search(r"only company\s*:", line, flags=re.I):
            candidate = re.split(r"only company\s*:", line, flags=re.I, maxsplit=1)[1]
        elif "→" in line and re.search(r"only company", line, flags=re.I):
            candidate = line.split("→", 1)[1]
            candidate = re.sub(r"only company\s*:?", "", candidate, flags=re.I).strip()
        elif "|" in line:
            candidate = line.split("|", 1)[0]
        elif ":" in line and "county" in low:
            candidate = line.split(":", 1)[1]

        candidate = re.sub(r"\[[^\]]+\]", "", candidate).strip()
        hits.update(extract_companies(candidate, company_aliases))

    if not hits:
        hits = extract_companies(golden_text, company_aliases)
    return hits


def extract_count(text: str, prefer_count_label: bool = True):
    raw = str(text or "")
    if prefer_count_label:
        m = re.search(r"(?im)^\s*Count:\s*([0-9,]+|[a-z -]+)", raw)
        if m:
            value = number_value(m.group(1).strip().split()[0])
            if value is not None:
                return value

    lowered = norm(raw)
    if re.search(r"\b(not found in the data|no matching|no rows|none can be|there are no|there is no)\b", lowered):
        return 0

    number_pattern = r"([0-9,]+|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?|thirty)"
    patterns = [
        rf"\bthere\s+(?:are|were)\s+(?:only\s+)?{number_pattern}\b",
        rf"\bthere\s+is\s+(?:only\s+)?{number_pattern}\b",
        rf"\bhas\s+{number_pattern}\s+(?:operating\s+entries|entries|locations|sites)\b",
        rf"\btop\s+{number_pattern}\b",
        rf"\bthe\s+{number_pattern}\s+(?:companies|company|suppliers|areas|facilities)\b",
        rf"\b{number_pattern}\s+(?:distinct\s+)?(?:companies|company|records|suppliers|areas|facilities|entries|oems)\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, lowered)
        if m:
            for idx, group in enumerate(m.groups(), start=1):
                value = number_value(group)
                if value is not None and m.start(idx) >= 0:
                    before = lowered[max(0, m.start(idx) - 12):m.start(idx)]
                    if "tier" in before:
                        continue
                if value is not None:
                    return value
    return None


def extract_numbers(text: str):
    return {
        re.sub(r"\D", "", m.group(0))
        for m in re.finditer(r"\b\d[\d,]*\b", str(text or ""))
        if re.sub(r"\D", "", m.group(0))
    }


def prf(pred, gold):
    pred, gold = set(pred), set(gold)
    if not gold and not pred:
        return 1.0, 1.0, 1.0
    if not gold:
        return 0.0, 1.0, 0.0
    true_pos = len(pred & gold)
    precision = true_pos / len(pred) if pred else 0.0
    recall = true_pos / len(gold) if gold else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def classify_question(question: str, use_case_category: str):
    q = norm(question)
    interpretive_terms = [
        "risk", "dependency", "fragility", "could be", "suitable", "suggesting",
        "reflecting", "emerging", "transition readiness", "potential", "opportunity",
        "innovation", "dual platform", "strategic", "import substitution",
    ]
    judge_recommended = any(term in q for term in interpretive_terms)

    if any(term in q for term in ["highest", "largest", "top", "rank", "total employment", "combined employment"]):
        qtype = "aggregation_ranking"
    elif "how many" in q or "number of" in q:
        qtype = "count"
    elif any(term in q for term in ["show all", "list all", "identify all", "map all", "which companies", "find "]):
        qtype = "list_multi_filter"
    elif any(term in q for term in ["which company", "which county", "which area"]):
        qtype = "lookup"
    else:
        qtype = "list_or_lookup"

    if judge_recommended:
        qtype = f"{qtype}_judge_recommended"
    return qtype, judge_recommended


def requested_fields(question: str):
    q = norm(question)
    fields = []
    for field, keywords in FIELD_KEYWORDS:
        if any(keyword in q for keyword in keywords):
            fields.append(field)
    # List questions often require the core role even when phrased compactly.
    if "supplier" in q and "EV Supply Chain Role" not in fields:
        fields.append("EV Supply Chain Role")
    return fields


def value_match_score(value, answer_text: str):
    if value is None:
        return None
    value_text = str(value).strip()
    if not value_text:
        return None
    answer_norm = norm(answer_text)
    value_norm = norm(value_text)
    if not value_norm:
        return None
    if value_norm in answer_norm:
        return 1.0
    if isinstance(value, (int, float)) or value_text.replace(",", "").isdigit():
        return 1.0 if value_text.replace(",", "") in re.sub(r"\D", " ", str(answer_text)).split() else 0.0
    tokens = [tok for tok in value_norm.split() if len(tok) > 2]
    if not tokens:
        return None
    overlap = sum(1 for tok in tokens if tok in answer_norm.split())
    return overlap / len(tokens)


def field_checks_for_answer(gold_companies, response_text, fields, by_company):
    checks = []
    if not gold_companies or not fields:
        return checks
    for company in sorted(gold_companies):
        records = by_company.get(company, [])
        for field in fields:
            values = []
            for record in records:
                value = record.get(field)
                if value is not None and str(value).strip():
                    values.append(value)
            if not values:
                continue
            scores = [value_match_score(value, response_text) for value in values]
            scores = [score for score in scores if score is not None]
            best_score = max(scores) if scores else None
            checks.append({
                "company": company,
                "field": field,
                "expected_values": "; ".join(str(v) for v in values),
                "field_value_score": round(best_score, 3) if best_score is not None else "",
                "field_value_present": bool(best_score is not None and best_score >= 0.6),
            })
    return checks


def extract_supporting_row_ids(text: str):
    raw = str(text or "")
    m = re.search(r"(?im)^\s*Supporting row_ids:\s*([^\n]+)", raw)
    if not m:
        return []
    return [int(n) for n in re.findall(r"\b\d{1,3}\b", m.group(1))]


def has_inline_thinking(text: str):
    return bool(re.search(r"<think>|</think>|\blet'?s think\b|\bwe need to\b", str(text or ""), re.I))


def direct_records(direct_dir: Path):
    for path in sorted(direct_dir.glob("answers_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            yield {
                "arm": "direct_context",
                "source_file": str(path),
                "model_name": row.get("model_name") or row.get("model"),
                "question_number": int(row.get("question_number")),
                "question": row.get("question"),
                "response": row.get("model_response") or "",
                "status": row.get("status") or "",
                "error_message": row.get("error_message") or "",
                "prompt_eval_count": row.get("prompt_eval_count"),
                "eval_count": row.get("eval_count"),
                "num_ctx_requested": row.get("num_ctx_requested"),
                "runtime_seconds": ns_to_seconds(row.get("total_duration")),
                "map_failed_count": "",
                "candidate_count": "",
                "verified_count": "",
            }


def batchwise_records(batchwise_dir: Path):
    for path in sorted(batchwise_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            timing = row.get("timing") or {}
            diagnostics = row.get("diagnostics") or {}
            ollama = row.get("ollama") or {}
            verification = row.get("verification") or {}
            yield {
                "arm": "batchwise_map_reduce",
                "source_file": str(path),
                "model_name": row.get("model"),
                "question_number": int(row.get("q_id")),
                "question": row.get("question"),
                "response": row.get("reduce_answer") or "",
                "status": row.get("reduce_status") or "",
                "error_message": row.get("reduce_error") or "",
                "prompt_eval_count": ollama.get("reduce_prompt_eval_count"),
                "eval_count": ollama.get("reduce_eval_count"),
                "num_ctx_requested": (row.get("config") or {}).get("reduce_num_ctx"),
                "runtime_seconds": sum(float(timing.get(k) or 0) for k in ["map_seconds", "verify_seconds", "reduce_seconds"]),
                "map_failed_count": diagnostics.get("n_map_failed"),
                "candidate_count": len(row.get("candidate_record_ids") or []),
                "verified_count": len(verification.get("verified_row_ids") or row.get("verified_record_ids") or []),
            }


def ns_to_seconds(value):
    if value is None or value == "":
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value > 1_000_000:
        return value / 1_000_000_000
    return value


def load_golden(path: str, company_aliases, county_aliases):
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    rows = {}
    question_type_rows = []
    for _, row in df.iterrows():
        qn = int(row["Num"])
        question = str(row["Question"]).strip()
        golden = str(row["Human validated answers"]).strip()
        use_case = str(row.get("Use Case Category") or "").strip()
        qtype, judge_recommended = classify_question(question, use_case)
        fields = requested_fields(question)
        gold_count = extract_count(golden, prefer_count_label=False)
        gold_companies = extract_golden_companies(golden, company_aliases)
        gold_counties = extract_counties(golden, county_aliases)
        gold_numbers = extract_numbers(golden)
        if gold_count is not None:
            gold_numbers.discard(str(gold_count))
        rows[qn] = {
            "question_number": qn,
            "question": question,
            "use_case_category": use_case,
            "golden_answer": golden,
            "question_type": qtype,
            "judge_recommended": judge_recommended,
            "requested_fields": fields,
            "gold_count": gold_count,
            "gold_companies": sorted(gold_companies),
            "gold_counties": sorted(gold_counties),
            "gold_numbers": sorted(gold_numbers, key=lambda x: (len(x), x)),
        }
        question_type_rows.append({
            "question_number": qn,
            "use_case_category": use_case,
            "question_type": qtype,
            "judge_recommended": judge_recommended,
            "requested_fields": "; ".join(fields),
            "gold_count": gold_count if gold_count is not None else "",
            "gold_company_count": len(gold_companies),
            "gold_county_count": len(gold_counties),
            "question": question,
        })
    return rows, pd.DataFrame(question_type_rows)


def score_answer(record, gold, by_company, company_aliases, county_aliases, valid_row_ids):
    response = record["response"]
    pred_companies = extract_companies(response, company_aliases)
    pred_counties = extract_counties(response, county_aliases)
    unknown_mentions = extract_unknown_company_mentions(response, company_aliases)
    pred_count = extract_count(response, prefer_count_label=True)
    supporting_ids = extract_supporting_row_ids(response)
    supporting_id_set = set(supporting_ids)

    gold_companies = set(gold["gold_companies"])
    gold_counties = set(gold["gold_counties"])

    company_precision, company_recall, company_f1 = prf(pred_companies, gold_companies)
    county_precision, county_recall, county_f1 = prf(pred_counties, gold_counties)

    if gold_companies:
        entity_precision, entity_recall, entity_f1 = company_precision, company_recall, company_f1
        entity_basis = "company"
    elif gold_counties:
        entity_precision, entity_recall, entity_f1 = county_precision, county_recall, county_f1
        entity_basis = "county"
    elif gold["gold_count"] == 0:
        entity_precision, entity_recall, entity_f1 = prf(pred_companies, set())
        entity_basis = "none_expected"
    else:
        entity_precision = entity_recall = entity_f1 = math.nan
        entity_basis = "not_applicable"

    gold_count = gold["gold_count"]
    count_error = None
    count_correct = None
    if gold_count is not None and pred_count is not None:
        count_error = pred_count - gold_count
        count_correct = count_error == 0

    gold_numbers = set(gold["gold_numbers"])
    pred_numbers = extract_numbers(response)
    numeric_recall = math.nan
    if gold_numbers:
        numeric_recall = len(gold_numbers & pred_numbers) / len(gold_numbers)

    field_checks = field_checks_for_answer(
        gold_companies=gold_companies,
        response_text=response,
        fields=gold["requested_fields"],
        by_company=by_company,
    )
    field_scores = [
        check["field_value_score"]
        for check in field_checks
        if check["field_value_score"] != ""
    ]
    field_value_accuracy = sum(float(v) >= 0.6 for v in field_scores) / len(field_scores) if field_scores else math.nan

    sections = {section: f"{section}:" in response for section in SECTION_NAMES}
    valid_supporting_ids = [rid for rid in supporting_ids if rid in valid_row_ids]
    bad_supporting_ids = [rid for rid in supporting_ids if rid not in valid_row_ids]
    duplicate_supporting_ids = len(supporting_ids) - len(set(supporting_ids))
    count_vs_supporting_ids_ok = None
    if pred_count is not None and supporting_ids:
        count_vs_supporting_ids_ok = pred_count == len(supporting_id_set)

    status_ok = str(record["status"]).lower() == "ok"
    answered = status_ok and bool(str(response).strip())
    format_core_ok = sections["Answer"] and sections["Evidence"] and sections["Count"] and sections["Supporting row_ids"]

    return {
        "arm": record["arm"],
        "model_name": record["model_name"],
        "question_number": record["question_number"],
        "question_type": gold["question_type"],
        "judge_recommended": gold["judge_recommended"],
        "use_case_category": gold["use_case_category"],
        "status": record["status"],
        "answered": answered,
        "error_message": record["error_message"],
        "gold_count": gold_count if gold_count is not None else "",
        "pred_count": pred_count if pred_count is not None else "",
        "count_correct": count_correct if count_correct is not None else "",
        "count_error": count_error if count_error is not None else "",
        "gold_company_count": len(gold_companies),
        "pred_company_count": len(pred_companies),
        "company_precision": round(company_precision, 4),
        "company_recall": round(company_recall, 4),
        "company_f1": round(company_f1, 4),
        "missing_companies": json.dumps(sorted(gold_companies - pred_companies), ensure_ascii=False),
        "extra_kb_companies": json.dumps(sorted(pred_companies - gold_companies), ensure_ascii=False),
        "gold_county_count": len(gold_counties),
        "pred_county_count": len(pred_counties),
        "county_precision": round(county_precision, 4),
        "county_recall": round(county_recall, 4),
        "county_f1": round(county_f1, 4),
        "entity_basis": entity_basis,
        "entity_precision": round(entity_precision, 4) if not math.isnan(entity_precision) else "",
        "entity_recall": round(entity_recall, 4) if not math.isnan(entity_recall) else "",
        "entity_f1": round(entity_f1, 4) if not math.isnan(entity_f1) else "",
        "gold_numbers": json.dumps(sorted(gold_numbers, key=lambda x: (len(x), x))),
        "numeric_recall": round(numeric_recall, 4) if not math.isnan(numeric_recall) else "",
        "field_value_accuracy": round(field_value_accuracy, 4) if not math.isnan(field_value_accuracy) else "",
        "unknown_company_mentions": json.dumps(unknown_mentions, ensure_ascii=False),
        "unknown_company_mention_count": len(unknown_mentions),
        "has_answer_section": sections["Answer"],
        "has_evidence_section": sections["Evidence"],
        "has_count_section": sections["Count"],
        "has_supporting_row_ids_section": sections["Supporting row_ids"],
        "has_notes_section": sections["Notes"],
        "format_core_ok": format_core_ok,
        "supporting_row_ids": json.dumps(supporting_ids),
        "supporting_row_id_count": len(supporting_id_set),
        "bad_supporting_row_ids": json.dumps(bad_supporting_ids),
        "duplicate_supporting_row_id_count": duplicate_supporting_ids,
        "count_vs_supporting_ids_ok": count_vs_supporting_ids_ok if count_vs_supporting_ids_ok is not None else "",
        "inline_thinking_leak": has_inline_thinking(response),
        "prompt_eval_count": record["prompt_eval_count"],
        "eval_count": record["eval_count"],
        "num_ctx_requested": record["num_ctx_requested"],
        "runtime_seconds": record["runtime_seconds"],
        "map_failed_count": record["map_failed_count"],
        "candidate_count": record["candidate_count"],
        "verified_count": record["verified_count"],
        "response_char_count": len(response),
        "source_file": record["source_file"],
    }, field_checks, unknown_mentions


def aggregate_scores(scores: pd.DataFrame):
    rows = []
    for (arm, model), sub in scores.groupby(["arm", "model_name"], dropna=False):
        answered = int(sub["answered"].sum())
        total = len(sub)
        entity_f1 = pd.to_numeric(sub["entity_f1"], errors="coerce")
        nonjudge = sub[sub["judge_recommended"] == False]  # noqa: E712
        judge = sub[sub["judge_recommended"] == True]  # noqa: E712
        count_scored = sub[sub["count_correct"] != ""]
        count_correct_rate = (
            count_scored["count_correct"].astype(bool).mean()
            if len(count_scored)
            else math.nan
        )
        field_accuracy = pd.to_numeric(sub["field_value_accuracy"], errors="coerce").mean()
        format_rate = sub["format_core_ok"].astype(bool).mean()
        hallucination_rate = (sub["unknown_company_mention_count"].astype(int) > 0).mean()
        reliability = answered / total if total else 0.0
        mean_f1 = entity_f1.mean()
        score_parts = [
            0.45 * (mean_f1 if not math.isnan(mean_f1) else 0),
            0.20 * (count_correct_rate if not math.isnan(count_correct_rate) else 0),
            0.15 * (field_accuracy if not math.isnan(field_accuracy) else 0),
            0.10 * (1 - hallucination_rate),
            0.05 * format_rate,
            0.05 * reliability,
        ]
        research_score = sum(score_parts)

        rows.append({
            "arm": arm,
            "model_name": model,
            "questions": total,
            "answered": answered,
            "reliability": round(reliability, 4),
            "mean_entity_f1": round(mean_f1, 4) if not math.isnan(mean_f1) else "",
            "mean_entity_precision": round(pd.to_numeric(sub["entity_precision"], errors="coerce").mean(), 4),
            "mean_entity_recall": round(pd.to_numeric(sub["entity_recall"], errors="coerce").mean(), 4),
            "nonjudge_mean_entity_f1": round(pd.to_numeric(nonjudge["entity_f1"], errors="coerce").mean(), 4) if len(nonjudge) else "",
            "judge_recommended_mean_entity_f1": round(pd.to_numeric(judge["entity_f1"], errors="coerce").mean(), 4) if len(judge) else "",
            "count_questions_scored": len(count_scored),
            "count_accuracy_rate": round(count_correct_rate, 4) if not math.isnan(count_correct_rate) else "",
            "mean_abs_count_error": round(pd.to_numeric(count_scored["count_error"], errors="coerce").abs().mean(), 4) if len(count_scored) else "",
            "field_value_accuracy": round(field_accuracy, 4) if not math.isnan(field_accuracy) else "",
            "format_core_ok_rate": round(format_rate, 4),
            "answers_with_unknown_company_mentions_rate": round(hallucination_rate, 4),
            "avg_unknown_company_mentions": round(sub["unknown_company_mention_count"].astype(int).mean(), 4),
            "inline_thinking_leak_count": int(sub["inline_thinking_leak"].astype(bool).sum()),
            "avg_runtime_seconds": round(pd.to_numeric(sub["runtime_seconds"], errors="coerce").mean(), 4),
            "total_runtime_seconds": round(pd.to_numeric(sub["runtime_seconds"], errors="coerce").sum(), 4),
            "prompt_eval_min": pd.to_numeric(sub["prompt_eval_count"], errors="coerce").min(),
            "prompt_eval_median": pd.to_numeric(sub["prompt_eval_count"], errors="coerce").median(),
            "prompt_eval_max": pd.to_numeric(sub["prompt_eval_count"], errors="coerce").max(),
            "research_score_deterministic": round(research_score, 4),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["rank_by_research_score"] = df["research_score_deterministic"].rank(ascending=False, method="min").astype(int)
        df = df.sort_values(["rank_by_research_score", "arm", "model_name"])
    return df


def aggregate_questions(scores: pd.DataFrame):
    rows = []
    for qn, sub in scores.groupby("question_number"):
        entity_f1 = pd.to_numeric(sub["entity_f1"], errors="coerce")
        count_scored = sub[sub["count_correct"] != ""]
        count_matches = int(count_scored["count_correct"].astype(bool).sum()) if len(count_scored) else 0
        rows.append({
            "question_number": qn,
            "question_type": sub["question_type"].iloc[0],
            "judge_recommended": bool(sub["judge_recommended"].iloc[0]),
            "gold_count": sub["gold_count"].iloc[0],
            "gold_company_count": sub["gold_company_count"].iloc[0],
            "mean_entity_f1": round(entity_f1.mean(), 4) if len(entity_f1.dropna()) else "",
            "models_with_count_match": count_matches,
            "models_scored_for_count": len(count_scored),
            "best_model_by_entity_f1": best_model(sub),
            "question": sub["source_question"].iloc[0] if "source_question" in sub.columns else "",
        })
    return pd.DataFrame(rows).sort_values(["mean_entity_f1", "question_number"], na_position="last")


def best_model(sub: pd.DataFrame):
    tmp = sub.copy()
    tmp["_f1"] = pd.to_numeric(tmp["entity_f1"], errors="coerce")
    tmp = tmp.sort_values(["_f1", "count_correct"], ascending=[False, False])
    if tmp.empty or pd.isna(tmp.iloc[0]["_f1"]):
        return ""
    return f"{tmp.iloc[0]['arm']}::{tmp.iloc[0]['model_name']}"


def cap_cell(value, limit=32767):
    if isinstance(value, str) and len(value) > limit:
        return value[: limit - 30] + "\n...[truncated for Excel]"
    return value


def write_outputs(out_dir: Path, scores, model_summary, question_summary, question_types, field_checks, unknown_mentions):
    out_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out_dir / "scores_by_answer.csv", index=False)
    model_summary.to_csv(out_dir / "scores_by_model.csv", index=False)
    question_summary.to_csv(out_dir / "scores_by_question.csv", index=False)
    question_types.to_csv(out_dir / "question_types.csv", index=False)
    field_checks.to_csv(out_dir / "field_value_checks.csv", index=False)
    unknown_mentions.to_csv(out_dir / "unknown_company_mentions.csv", index=False)

    workbook = out_dir / "evaluation_summary_42q.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        for sheet, df in [
            ("model_summary", model_summary),
            ("question_summary", question_summary),
            ("scores_by_answer", scores),
            ("question_types", question_types),
            ("field_value_checks", field_checks),
            ("unknown_company_mentions", unknown_mentions),
        ]:
            df.apply(lambda col: col.map(cap_cell)).to_excel(writer, sheet_name=sheet, index=False)
    return workbook


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate 42Q direct-context and batchwise model responses.")
    parser.add_argument("--kb", default=DEFAULT_KB)
    parser.add_argument("--golden", default=DEFAULT_GOLDEN)
    parser.add_argument("--direct-dir", default=DEFAULT_DIRECT_DIR)
    parser.add_argument("--batchwise-dir", default=DEFAULT_BATCHWISE_DIR)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--skip-direct", action="store_true")
    parser.add_argument("--skip-batchwise", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    records, by_company, company_aliases, county_aliases = load_kb(args.kb)
    valid_row_ids = {int(record["row_id"]) for record in records}
    golden, question_types = load_golden(args.golden, company_aliases, county_aliases)

    run_records = []
    if not args.skip_direct:
        run_records.extend(direct_records(Path(args.direct_dir)))
    if not args.skip_batchwise:
        run_records.extend(batchwise_records(Path(args.batchwise_dir)))
    if not run_records:
        raise SystemExit("No model response records found.")

    score_rows = []
    field_rows = []
    unknown_rows = []
    for record in run_records:
        qn = int(record["question_number"])
        if qn not in golden:
            continue
        score, checks, unknowns = score_answer(
            record=record,
            gold=golden[qn],
            by_company=by_company,
            company_aliases=company_aliases,
            county_aliases=county_aliases,
            valid_row_ids=valid_row_ids,
        )
        score["source_question"] = golden[qn]["question"]
        score_rows.append(score)
        for check in checks:
            check.update({
                "arm": record["arm"],
                "model_name": record["model_name"],
                "question_number": qn,
            })
            field_rows.append(check)
        for mention in unknowns:
            unknown_rows.append({
                "arm": record["arm"],
                "model_name": record["model_name"],
                "question_number": qn,
                "unknown_company_mention": mention,
            })

    scores = pd.DataFrame(score_rows)
    field_checks = pd.DataFrame(field_rows)
    unknown_mentions = pd.DataFrame(unknown_rows)
    if field_checks.empty:
        field_checks = pd.DataFrame(columns=["arm", "model_name", "question_number", "company", "field"])
    if unknown_mentions.empty:
        unknown_mentions = pd.DataFrame(columns=["arm", "model_name", "question_number", "unknown_company_mention"])

    model_summary = aggregate_scores(scores)
    question_summary = aggregate_questions(scores)
    workbook = write_outputs(
        out_dir=Path(args.out),
        scores=scores,
        model_summary=model_summary,
        question_summary=question_summary,
        question_types=question_types,
        field_checks=field_checks,
        unknown_mentions=unknown_mentions,
    )

    print("=== Model Summary ===")
    show_cols = [
        "rank_by_research_score", "arm", "model_name", "answered", "questions",
        "mean_entity_f1", "count_accuracy_rate", "field_value_accuracy",
        "format_core_ok_rate", "answers_with_unknown_company_mentions_rate",
        "research_score_deterministic",
    ]
    print(model_summary[show_cols].to_string(index=False))
    print(f"\nWrote {Path(args.out)}")
    print(f"Wrote {workbook}")
    print("\nNote: rows marked judge_recommended should be reviewed with the second evaluation type")
    print("(external LLM-as-judge or human rubric) before making final claims about interpretation questions.")


if __name__ == "__main__":
    main()
