#!/usr/bin/env python3
"""
Evaluate Georgia EV 42-question model responses -- V2 (research-ready).

This builds on shared helpers in scripts/eval_common.py (imported as `v1`) and
keeps every deterministic metric the audit found sound (entity precision/recall/
F1, count accuracy + signed error, field-value accuracy, format compliance,
supporting row_id validity, runtime/context diagnostics). It REPLACES the
unknown-company / hallucination detection, which in v1 scanned the entire answer
text and produced false positives from KB field values -- e.g. Industry Group
values like "Primary Metal Industries" and Product/Service substrings like
"Jackson America, Inc.*" that live in non-Company columns of the Evidence table.

V2 changes (all KB-driven and general -- no per-answer / per-question hardcoding):

  1. Evidence-table-first parsing. Company mentions are read from the `Company`
     column of the Evidence table only. This structurally excludes the Industry
     Group / Product-Service columns that caused the v1 false positives. The
     Answer prose is also scanned (company-suffix phrases only) so a hallucinated
     company named in prose but absent from Evidence is still caught.

  2. Mention classification taxonomy. Every candidate mention is classified as:
       known_kb_company                (resolves to a real KB company -> not flagged)
       industry_group_false_positive   (matches an Industry Group value)
       product_service_false_positive  (matches a Product / Service value)
       kb_supported_non_company_field  (matches another non-Company KB field value)
       misspelled_kb_company           (fuzzy match to a real KB company; model error)
       not_a_company_phrase            (generic/descriptive, not company-shaped)
       true_out_of_kb_hallucination    (company-shaped, absent from the KB)
     Misspellings are kept as model errors (never silently corrected) and linked
     to their nearest KB company with a similarity score.

  3. Row_id -> company consistency. For each Evidence row that cites a row_id,
     the KB company actually stored at that row_id is compared to the claimed
     company. Disagreements are recorded as row_id-company mismatches (v1 only
     checked that the row_id existed, not that it matched the claim).

  4. Corrected composite score. research_score_deterministic now uses the
     corrected true_hallucination_answer_rate. The v1 (regex-contaminated)
     hallucination rate and the composite it produces are preserved with a
     `_legacy` label so nothing is silently dropped.

Outputs (under --out):
  scores_by_model.csv, scores_by_answer.csv, scores_by_question.csv,
  mention_classification.csv, row_id_company_checks.csv, field_value_checks.csv,
  evaluation_summary_42q.xlsx
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_common as v1  # noqa: E402

norm = v1.norm
norm_company = v1.norm_company
canonical_company_key = v1.canonical_company_key

DEFAULT_OUT = "evaluation_v2"

# Fuzzy matching for "misspelled KB company" detection.
FUZZY_THRESHOLD = 0.82
FUZZY_MIN_LEN = 5

# A mention "looks like a company" if it carries a corporate-style suffix token.
COMPANY_SUFFIX_TOKENS = re.compile(
    r"\b(inc|incorporated|llc|l\.l\.c|co|corp|corporation|ltd|company|"
    r"holdings?|gmbh|group|industries|systems|manufacturing|mfg|"
    r"technologies|technology|plastics|steel|automotive|motors|materials)\b",
    re.I,
)

# Obvious non-entity cell contents that should never be treated as a company.
GENERIC_PHRASES = {
    "", "company", "various", "various companies", "multiple", "multiple companies",
    "suppliers", "companies", "see above", "see evidence", "n a", "na", "none",
    "not found", "not found in the data", "tier", "unknown", "other", "others",
    "the company", "various suppliers", "all suppliers", "supplier",
}

# Non-Company KB columns scanned (besides Industry Group and Product / Service,
# which get their own dedicated classification buckets).
NON_COMPANY_FIELDS = [
    "Primary OEMs",
    "Category",
    "Primary Facility Type",
    "Supplier or Affiliation Type",
    "EV Supply Chain Role",
    "Classification Method",
]


# --------------------------------------------------------------------------- #
# KB indexes used only by the corrected hallucination / row_id logic.
# --------------------------------------------------------------------------- #
def build_kb_indexes(records, by_company):
    kb_by_row_id = {}
    for rec in records:
        rid = rec.get("row_id")
        if rid is not None:
            kb_by_row_id[int(rid)] = rec

    def field_norm_set(field):
        vals = set()
        for rec in records:
            value = rec.get(field)
            if value is None:
                continue
            value_norm = norm(value)
            if value_norm:
                vals.add(value_norm)
        return vals

    industry_group_norms = field_norm_set("Industry Group")
    product_service_norms = sorted(field_norm_set("Product / Service"), key=len, reverse=True)
    other_field_norms = {field: field_norm_set(field) for field in NON_COMPANY_FIELDS}

    kb_canon = []
    for company in by_company:
        ck = canonical_company_key(company)
        if ck:
            kb_canon.append((ck, company))

    # Structural labels: KB column names and output section headers. These are
    # field/section labels (e.g. "Industry Group", "Supporting row_ids") that a
    # model may echo in prose; they must never be treated as company mentions.
    structural_labels = {norm(name) for name in v1.SECTION_NAMES}
    structural_labels |= {norm("Supporting row_ids"), norm("row_id")}
    for rec in records:
        structural_labels |= {norm(col) for col in rec.keys()}
        break

    return {
        "kb_by_row_id": kb_by_row_id,
        "industry_group_norms": industry_group_norms,
        "product_service_norms": product_service_norms,
        "other_field_norms": other_field_norms,
        "kb_canon": kb_canon,
        "structural_labels": structural_labels,
    }


# --------------------------------------------------------------------------- #
# Evidence-table parsing.
# --------------------------------------------------------------------------- #
def _split_row(line: str):
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _is_separator(cells):
    nonempty = [c.strip() for c in cells if c.strip()]
    return bool(nonempty) and all(re.fullmatch(r":?-{2,}:?", c) for c in nonempty)


def _section(text: str, name: str) -> str:
    """Return the body of a named output section (Answer/Evidence/Count/...)."""
    raw = str(text or "")
    m = re.search(rf"(?i)\b{name}\s*:", raw)
    if not m:
        return ""
    start = m.end()
    nxt = re.search(
        r"(?im)^\s*(Answer|Evidence|Count|Supporting row_ids|Notes)\s*:",
        raw[start:],
    )
    end = start + nxt.start() if nxt else len(raw)
    return raw[start:end]


def _clean_company_cell(cell: str) -> str:
    cell = re.sub(r"\*+|`+", "", str(cell or ""))
    return cell.strip(" .,:;|")


def parse_evidence_table(text: str):
    """Parse the Evidence markdown table into [{'row_id': int|None, 'company': str}].

    Reads the `row_id` and `Company` columns by header name, falling back to the
    documented positional layout (col 0 = row_id, col 1 = Company) when headers
    are missing. Only the Company column is returned -- Industry Group /
    Product-Service columns are deliberately ignored.
    """
    section = _section(text, "Evidence")
    if not section:
        return []

    header = None
    row_id_idx = None
    company_idx = None
    rows = []
    for line in section.splitlines():
        if "|" not in line:
            continue
        cells = _split_row(line)
        if _is_separator(cells):
            continue
        low = [re.sub(r"[^a-z0-9 ]+", " ", c.lower()).strip() for c in cells]

        is_header = header is None and any(
            ("company" in c) or (c in ("row id", "rowid", "id")) for c in low
        )
        if is_header:
            header = low
            for i, h in enumerate(low):
                if row_id_idx is None and (h in ("row id", "rowid", "id") or "row id" in h):
                    row_id_idx = i
                if company_idx is None and ("company" in h or h in ("supplier", "name")):
                    company_idx = i
            if row_id_idx is None:
                row_id_idx = 0
            if company_idx is None:
                company_idx = 1 if len(low) > 1 else 0
            continue

        if header is None:
            # No header row seen; accept positional layout only if it looks like
            # a data row (first cell numeric, i.e. a row_id).
            if cells and re.search(r"\d", cells[0]):
                header, row_id_idx, company_idx = [], 0, 1
            else:
                continue

        rid = None
        if row_id_idx is not None and row_id_idx < len(cells):
            mnum = re.search(r"\d{1,4}", cells[row_id_idx])
            if mnum:
                rid = int(mnum.group(0))
        company = _clean_company_cell(cells[company_idx]) if company_idx is not None and company_idx < len(cells) else ""
        if company:
            rows.append({"row_id": rid, "company": company})
    return rows


# --------------------------------------------------------------------------- #
# Mention classification.
# --------------------------------------------------------------------------- #
def _contains_phrase(value_norm: str, cand_norm: str) -> bool:
    return f" {cand_norm} " in f" {value_norm} "


def match_field_value(cand_norm: str, idx):
    """Return (classification, field, matched_value) if the candidate matches a
    KB non-company field value, else None."""
    if len(cand_norm) < 5:
        return None
    multi = len(cand_norm.split()) >= 2

    if cand_norm in idx["industry_group_norms"]:
        return ("industry_group_false_positive", "Industry Group", cand_norm)
    if multi:
        for value in idx["industry_group_norms"]:
            if _contains_phrase(value, cand_norm):
                return ("industry_group_false_positive", "Industry Group", value)

    for value in idx["product_service_norms"]:
        if cand_norm == value or (multi and _contains_phrase(value, cand_norm)):
            return ("product_service_false_positive", "Product / Service", value)

    for field, values in idx["other_field_norms"].items():
        if cand_norm in values:
            return ("kb_supported_non_company_field", field, cand_norm)
        if multi:
            for value in values:
                if _contains_phrase(value, cand_norm):
                    return ("kb_supported_non_company_field", field, value)
    return None


def best_fuzzy_company(candidate: str, idx):
    ck = canonical_company_key(candidate)
    if len(ck) < FUZZY_MIN_LEN:
        return None, 0.0
    best, best_score = None, 0.0
    for kb_ck, company in idx["kb_canon"]:
        score = SequenceMatcher(None, ck, kb_ck).ratio()
        if score > best_score:
            best, best_score = company, score
    return best, best_score


def kb_company_containing(candidate: str, idx):
    """Return a KB company whose canonical name contains the candidate as a
    contiguous token sub-sequence, else None.

    This recovers partial / fragmented mentions of real companies -- e.g.
    "Vose Co" extracted from the prose "Hollingsworth & Vose Co." -- so a
    fragment of a genuine KB company is not miscounted as a hallucination. A
    distinctive token (length >= 4) is required to avoid matching on bare
    suffix words like "co" or "inc".
    """
    tokens = canonical_company_key(candidate).split()
    if not tokens or not any(len(tok) >= 4 for tok in tokens):
        return None
    n = len(tokens)
    for kb_ck, company in idx["kb_canon"]:
        kb_tokens = kb_ck.split()
        for i in range(len(kb_tokens) - n + 1):
            if kb_tokens[i:i + n] == tokens:
                return company
    return None


def classify_mention(candidate: str, source: str, company_aliases, idx):
    cand = re.sub(r"\s+", " ", str(candidate or "")).strip().strip("*`\"' .,:;|")
    cand_norm = norm(cand)
    result = {
        "mention_text": cand,
        "source": source,
        "classification": None,
        "matched_kb_company": "",
        "similarity": "",
        "matched_field": "",
        "matched_value": "",
    }

    if (not cand_norm or cand_norm in GENERIC_PHRASES
            or cand_norm in idx.get("structural_labels", set())
            or canonical_company_key(cand) == ""):
        result["classification"] = "not_a_company_phrase"
        return result

    # 1. Known KB company (exact / alias).
    known = v1.extract_companies(cand, company_aliases)
    if known:
        result["classification"] = "known_kb_company"
        result["matched_kb_company"] = sorted(known)[0]
        result["similarity"] = 1.0
        return result

    # 2-4. KB non-company field values (Industry Group / Product-Service / other).
    field_match = match_field_value(cand_norm, idx)
    if field_match:
        result["classification"], result["matched_field"], result["matched_value"] = field_match
        return result

    # 5. Fuzzy match -> misspelled KB company (kept as a model error).
    best, score = best_fuzzy_company(cand, idx)
    if best is not None and score >= FUZZY_THRESHOLD:
        result["classification"] = "misspelled_kb_company"
        result["matched_kb_company"] = best
        result["similarity"] = round(score, 4)
        return result

    # 6. Partial / fragmented mention of a real KB company (e.g. a prose fragment
    #    "Vose Co" of "Hollingsworth & Vose Co.") -> treat as known, not a
    #    hallucination.
    partial = kb_company_containing(cand, idx)
    if partial is not None:
        result["classification"] = "known_kb_company"
        result["matched_kb_company"] = partial
        result["similarity"] = "partial"
        return result

    # 7-8. Company-shaped but absent -> hallucination; otherwise generic phrase.
    if COMPANY_SUFFIX_TOKENS.search(cand) or source == "evidence_company":
        result["classification"] = "true_out_of_kb_hallucination"
        if best is not None:
            result["matched_kb_company"] = best  # nearest KB company (below threshold)
            result["similarity"] = round(score, 4)
    else:
        result["classification"] = "not_a_company_phrase"
    return result


def classify_answer_mentions(response: str, company_aliases, idx):
    """Classify every distinct company mention in an answer (deduped by canonical key)."""
    mentions = {}
    order = []

    def add(candidate, source):
        ck = canonical_company_key(candidate) or norm(candidate)
        if not ck:
            return
        if ck in mentions:
            if source not in mentions[ck]["sources"]:
                mentions[ck]["sources"].append(source)
            return
        result = classify_mention(candidate, source, company_aliases, idx)
        result["sources"] = [source]
        mentions[ck] = result
        order.append(ck)

    for row in parse_evidence_table(response):
        add(row["company"], "evidence_company")

    answer_text = _section(response, "Answer") or str(response or "")[:800]
    for m in v1.UNKNOWN_COMPANY_RE.finditer(answer_text):
        add(m.group(1).strip(" .,:;|"), "answer_prose")

    return [mentions[k] for k in order]


# --------------------------------------------------------------------------- #
# Row_id -> company consistency.
# --------------------------------------------------------------------------- #
def _company_consistency(claimed: str, kb_company: str, company_aliases, idx):
    claimed_norm = norm(claimed)
    kb_norm = norm(kb_company)
    if not kb_norm:
        return "bad_row_id", ""
    if claimed_norm == kb_norm or norm_company(claimed) == norm_company(kb_company):
        return "match", "exact"
    if kb_company in v1.extract_companies(claimed, company_aliases):
        return "match", "alias"
    score = SequenceMatcher(
        None, canonical_company_key(claimed), canonical_company_key(kb_company)
    ).ratio()
    if score >= FUZZY_THRESHOLD:
        return "match", f"fuzzy:{round(score, 3)}"
    return "mismatch", ""


def row_id_company_checks(response: str, company_aliases, idx):
    checks = []
    kb_by_row_id = idx["kb_by_row_id"]
    for row in parse_evidence_table(response):
        rid = row["row_id"]
        company = row["company"]
        if rid is None or not company:
            continue
        rec = kb_by_row_id.get(rid)
        if rec is None:
            checks.append({
                "claimed_company": company, "cited_row_id": rid,
                "kb_company_at_row_id": "", "status": "bad_row_id", "match_type": "",
            })
            continue
        kb_company = str(rec.get("Company") or "").strip()
        status, match_type = _company_consistency(company, kb_company, company_aliases, idx)
        checks.append({
            "claimed_company": company, "cited_row_id": rid,
            "kb_company_at_row_id": kb_company, "status": status, "match_type": match_type,
        })
    return checks


# --------------------------------------------------------------------------- #
# Per-answer scoring (v1 base metrics + v2 corrected hallucination/row_id).
# --------------------------------------------------------------------------- #
def score_answer_v2(record, gold, kb, idx):
    by_company, company_aliases, county_aliases, valid_row_ids = kb
    base, field_checks, _legacy_unknowns = v1.score_answer(
        record, gold, by_company, company_aliases, county_aliases, valid_row_ids
    )
    response = record["response"]

    mentions = classify_answer_mentions(response, company_aliases, idx)
    counts = defaultdict(int)
    for mention in mentions:
        counts[mention["classification"]] += 1

    rid_checks = row_id_company_checks(response, company_aliases, idx)
    n_mismatch = sum(1 for c in rid_checks if c["status"] == "mismatch")
    n_bad = sum(1 for c in rid_checks if c["status"] == "bad_row_id")

    nonco_fp = (
        counts["industry_group_false_positive"]
        + counts["product_service_false_positive"]
        + counts["kb_supported_non_company_field"]
    )

    # Preserve v1 (legacy) hallucination columns under a clear label.
    base["legacy_unknown_company_mentions"] = base.pop("unknown_company_mentions")
    base["legacy_unknown_company_mention_count"] = base.pop("unknown_company_mention_count")

    base.update({
        "evidence_company_mentions": sum(1 for m in mentions if "evidence_company" in m["sources"]),
        "mention_known_kb_company_count": counts["known_kb_company"],
        "mention_misspelled_kb_company_count": counts["misspelled_kb_company"],
        "mention_industry_group_fp_count": counts["industry_group_false_positive"],
        "mention_product_service_fp_count": counts["product_service_false_positive"],
        "mention_kb_noncompany_field_count": counts["kb_supported_non_company_field"],
        "mention_not_a_company_phrase_count": counts["not_a_company_phrase"],
        "mention_true_hallucination_count": counts["true_out_of_kb_hallucination"],
        "kb_supported_non_company_fp_count": nonco_fp,
        "row_id_company_checked": len(rid_checks),
        "row_id_company_mismatch_count": n_mismatch,
        "row_id_bad_count": n_bad,
    })
    return base, field_checks, mentions, rid_checks


# --------------------------------------------------------------------------- #
# Aggregation.
# --------------------------------------------------------------------------- #
def aggregate_scores_v2(scores: pd.DataFrame):
    rows = []
    for (arm, model), sub in scores.groupby(["arm", "model_name"], dropna=False):
        total = len(sub)
        answered = int(sub["answered"].sum())
        reliability = answered / total if total else 0.0

        entity_f1 = pd.to_numeric(sub["entity_f1"], errors="coerce")
        mean_f1 = entity_f1.mean()
        count_scored = sub[sub["count_correct"] != ""]
        count_rate = count_scored["count_correct"].astype(bool).mean() if len(count_scored) else math.nan
        field_acc = pd.to_numeric(sub["field_value_accuracy"], errors="coerce").mean()
        format_rate = sub["format_core_ok"].astype(bool).mean()

        true_hall_rate = (sub["mention_true_hallucination_count"].astype(int) > 0).mean()
        misspell_rate = (sub["mention_misspelled_kb_company_count"].astype(int) > 0).mean()
        nonco_fp_rate = (sub["kb_supported_non_company_fp_count"].astype(int) > 0).mean()
        rid_mismatch_rate = (sub["row_id_company_mismatch_count"].astype(int) > 0).mean()
        legacy_hall_rate = (sub["legacy_unknown_company_mention_count"].astype(int) > 0).mean()

        def composite(hall_rate):
            return (
                0.45 * (mean_f1 if not math.isnan(mean_f1) else 0)
                + 0.20 * (count_rate if not math.isnan(count_rate) else 0)
                + 0.15 * (field_acc if not math.isnan(field_acc) else 0)
                + 0.10 * (1 - hall_rate)
                + 0.05 * format_rate
                + 0.05 * reliability
            )

        rows.append({
            "arm": arm,
            "model_name": model,
            "questions": total,
            "answered": answered,
            "reliability": round(reliability, 4),
            "mean_entity_f1": round(mean_f1, 4) if not math.isnan(mean_f1) else "",
            "mean_entity_precision": round(pd.to_numeric(sub["entity_precision"], errors="coerce").mean(), 4),
            "mean_entity_recall": round(pd.to_numeric(sub["entity_recall"], errors="coerce").mean(), 4),
            "count_questions_scored": len(count_scored),
            "count_accuracy_rate": round(count_rate, 4) if not math.isnan(count_rate) else "",
            "mean_abs_count_error": round(pd.to_numeric(count_scored["count_error"], errors="coerce").abs().mean(), 4) if len(count_scored) else "",
            "field_value_accuracy": round(field_acc, 4) if not math.isnan(field_acc) else "",
            "format_core_ok_rate": round(format_rate, 4),
            "true_hallucination_answer_rate": round(true_hall_rate, 4),
            "avg_true_hallucinations": round(sub["mention_true_hallucination_count"].astype(int).mean(), 4),
            "misspelled_kb_company_answer_rate": round(misspell_rate, 4),
            "avg_misspelled_kb_company": round(sub["mention_misspelled_kb_company_count"].astype(int).mean(), 4),
            "kb_supported_non_company_false_positive_rate": round(nonco_fp_rate, 4),
            "avg_kb_supported_non_company_fp": round(sub["kb_supported_non_company_fp_count"].astype(int).mean(), 4),
            "row_id_company_mismatch_rate": round(rid_mismatch_rate, 4),
            "avg_row_id_company_mismatch": round(sub["row_id_company_mismatch_count"].astype(int).mean(), 4),
            "total_row_id_company_checked": int(sub["row_id_company_checked"].astype(int).sum()),
            "total_row_id_company_mismatch": int(sub["row_id_company_mismatch_count"].astype(int).sum()),
            "legacy_answers_with_unknown_company_mentions_rate": round(legacy_hall_rate, 4),
            "inline_thinking_leak_count": int(sub["inline_thinking_leak"].astype(bool).sum()),
            "avg_runtime_seconds": round(pd.to_numeric(sub["runtime_seconds"], errors="coerce").mean(), 4),
            "total_runtime_seconds": round(pd.to_numeric(sub["runtime_seconds"], errors="coerce").sum(), 4),
            "prompt_eval_min": pd.to_numeric(sub["prompt_eval_count"], errors="coerce").min(),
            "prompt_eval_median": pd.to_numeric(sub["prompt_eval_count"], errors="coerce").median(),
            "prompt_eval_max": pd.to_numeric(sub["prompt_eval_count"], errors="coerce").max(),
            "research_score_deterministic": round(composite(true_hall_rate), 4),
            "research_score_deterministic_legacy": round(composite(legacy_hall_rate), 4),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["rank_by_research_score"] = df["research_score_deterministic"].rank(ascending=False, method="min").astype(int)
        df = df.sort_values(["rank_by_research_score", "arm", "model_name"])
    return df


# --------------------------------------------------------------------------- #
# Output.
# --------------------------------------------------------------------------- #
def write_outputs(out_dir: Path, scores, model_summary, question_summary,
                  question_types, field_checks, mention_rows, row_id_rows):
    out_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out_dir / "scores_by_answer.csv", index=False)
    model_summary.to_csv(out_dir / "scores_by_model.csv", index=False)
    question_summary.to_csv(out_dir / "scores_by_question.csv", index=False)
    question_types.to_csv(out_dir / "question_types.csv", index=False)
    field_checks.to_csv(out_dir / "field_value_checks.csv", index=False)
    mention_rows.to_csv(out_dir / "mention_classification.csv", index=False)
    row_id_rows.to_csv(out_dir / "row_id_company_checks.csv", index=False)

    workbook = out_dir / "evaluation_summary_42q.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        for sheet, df in [
            ("model_summary", model_summary),
            ("question_summary", question_summary),
            ("scores_by_answer", scores),
            ("question_types", question_types),
            ("field_value_checks", field_checks),
            ("mention_classification", mention_rows),
            ("row_id_company_checks", row_id_rows),
        ]:
            df.apply(lambda col: col.map(v1.cap_cell)).to_excel(writer, sheet_name=sheet, index=False)
    return workbook


# --------------------------------------------------------------------------- #
# Driver.
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate 42Q model responses (V2, research-ready).")
    parser.add_argument("--kb", default=v1.DEFAULT_KB)
    parser.add_argument("--golden", default=v1.DEFAULT_GOLDEN)
    parser.add_argument("--direct-dir", default=v1.DEFAULT_DIRECT_DIR)
    parser.add_argument("--batchwise-dir", default=v1.DEFAULT_BATCHWISE_DIR)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--models", default="", help="comma-separated model_name filter")
    parser.add_argument("--questions", default="", help="comma-separated question numbers to score")
    parser.add_argument("--limit", type=int, default=0, help="score only the first N question numbers")
    parser.add_argument("--skip-direct", action="store_true")
    parser.add_argument("--skip-batchwise", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    records, by_company, company_aliases, county_aliases = v1.load_kb(args.kb)
    valid_row_ids = {int(rec["row_id"]) for rec in records}
    idx = build_kb_indexes(records, by_company)
    kb = (by_company, company_aliases, county_aliases, valid_row_ids)
    golden, question_types = v1.load_golden(args.golden, company_aliases, county_aliases)

    run_records = []
    if not args.skip_direct:
        run_records.extend(v1.direct_records(Path(args.direct_dir)))
    if not args.skip_batchwise:
        run_records.extend(v1.batchwise_records(Path(args.batchwise_dir)))

    models_filter = {m.strip() for m in args.models.split(",") if m.strip()} or None
    if models_filter:
        run_records = [r for r in run_records if r["model_name"] in models_filter]

    questions_filter = {int(q) for q in args.questions.split(",") if q.strip()} or None
    if questions_filter:
        run_records = [r for r in run_records if int(r["question_number"]) in questions_filter]
    elif args.limit:
        allowed = sorted({int(r["question_number"]) for r in run_records})[: args.limit]
        allowed = set(allowed)
        run_records = [r for r in run_records if int(r["question_number"]) in allowed]

    if not run_records:
        raise SystemExit("No model response records found for the requested filters.")

    score_rows, field_rows, mention_rows, row_id_rows = [], [], [], []
    for record in run_records:
        qn = int(record["question_number"])
        if qn not in golden:
            continue
        score, checks, mentions, rid_checks = score_answer_v2(record, golden[qn], kb, idx)
        score["source_question"] = golden[qn]["question"]
        score_rows.append(score)
        for check in checks:
            check.update({"arm": record["arm"], "model_name": record["model_name"], "question_number": qn})
            field_rows.append(check)
        for mention in mentions:
            mention_rows.append({
                "arm": record["arm"], "model_name": record["model_name"], "question_number": qn,
                "mention_text": mention["mention_text"],
                "classification": mention["classification"],
                "sources": ",".join(mention["sources"]),
                "matched_kb_company": mention["matched_kb_company"],
                "similarity": mention["similarity"],
                "matched_field": mention["matched_field"],
                "matched_value": mention["matched_value"],
            })
        for check in rid_checks:
            row_id_rows.append({
                "arm": record["arm"], "model_name": record["model_name"], "question_number": qn,
                **check,
            })

    scores = pd.DataFrame(score_rows)
    field_checks = pd.DataFrame(field_rows) if field_rows else pd.DataFrame(
        columns=["arm", "model_name", "question_number", "company", "field"])
    mention_df = pd.DataFrame(mention_rows) if mention_rows else pd.DataFrame(
        columns=["arm", "model_name", "question_number", "mention_text", "classification"])
    row_id_df = pd.DataFrame(row_id_rows) if row_id_rows else pd.DataFrame(
        columns=["arm", "model_name", "question_number", "claimed_company", "cited_row_id",
                 "kb_company_at_row_id", "status", "match_type"])

    model_summary = aggregate_scores_v2(scores)
    question_summary = v1.aggregate_questions(scores)
    workbook = write_outputs(
        out_dir=Path(args.out), scores=scores, model_summary=model_summary,
        question_summary=question_summary, question_types=question_types,
        field_checks=field_checks, mention_rows=mention_df, row_id_rows=row_id_df,
    )

    print("=== Model Summary (V2) ===")
    show_cols = [
        "rank_by_research_score", "arm", "model_name", "answered", "questions",
        "mean_entity_f1", "count_accuracy_rate", "field_value_accuracy",
        "true_hallucination_answer_rate", "kb_supported_non_company_false_positive_rate",
        "row_id_company_mismatch_rate", "legacy_answers_with_unknown_company_mentions_rate",
        "research_score_deterministic", "research_score_deterministic_legacy",
    ]
    show_cols = [c for c in show_cols if c in model_summary.columns]
    print(model_summary[show_cols].to_string(index=False))
    print(f"\nWrote {Path(args.out)}")
    print(f"Wrote {workbook}")


if __name__ == "__main__":
    main()
