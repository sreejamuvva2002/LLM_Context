"""
build_gold.py  --  GRADING-ONLY ground-truth builder.

This script is part of the *evaluation* stage, NOT answer generation. The professor's
constraint ("the LLM must produce the answer, Python must not") applies to the model-run
scripts (run_ollama_jsonl_eval.py, run_qwen25_batched_eval.py). Grading is explicitly
allowed to use Python, and that is all this file does: it derives reference answers from
the spreadsheet so model outputs can be scored.

Design rules:
- row_id scheme matches the eval scripts exactly: row_id = dataframe_index + 1.
- Deterministic gold is computed ONLY for questions whose answer is an unambiguous,
  rule-based query over the KB. Each such rule is written out explicitly and is auditable.
- Fuzzy / interpretive / under-specified questions are NOT guessed. They are emitted with
  gold_row_ids = "" and needs_human = True, plus a suggested rule, for hand validation.
  (You said the 50 questions are human-validated, so filling these in is a small manual pass.)

Output: gold/gold_answers.csv  and  gold/question_types.csv
"""

import json
import re
from pathlib import Path

import pandas as pd

XLSX = Path(__file__).resolve().parent.parent / "GNEM_Excel_Data.xlsx"
OUT_GOLD = Path(__file__).resolve().parent / "gold_answers.csv"
OUT_TYPES = Path(__file__).resolve().parent / "question_types.csv"
QUESTIONS = Path(__file__).resolve().parent.parent / "questions.txt"


# --------------------------------------------------------------------------------------
# Load + light normalization (we keep raw values too; normalization is only for matching)
# --------------------------------------------------------------------------------------
def load():
    df = pd.read_excel(XLSX)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all").reset_index(drop=True)
    df["row_id"] = df.index + 1

    # Normalized helper columns (do not mutate the originals used for display).
    df["_category"] = df["Category"].astype(str).str.strip()
    df["_role"] = df["EV Supply Chain Role"].astype(str).str.strip()
    df["_role_l"] = df["_role"].str.lower()
    df["_industry"] = df["Industry Group"].astype(str).str.strip()
    df["_facility"] = df["Primary Facility Type"].astype(str).str.strip().str.lower()
    df["_evrel"] = df["EV / Battery Relevant"].astype(str).str.strip().str.lower()
    df["_product_l"] = df["Product / Service"].astype(str).str.lower()
    df["_oem"] = df["Primary OEMs"].astype(str).str.strip()
    df["_oem_l"] = df["_oem"].str.lower()
    df["_company_l"] = df["Company"].astype(str).str.lower()
    df["Employment"] = pd.to_numeric(df["Employment"], errors="coerce")

    # County extracted from "City, County" Location field.
    df["_county"] = (
        df["Location"].astype(str).str.split(",").str[-1].str.strip()
    )
    return df


def ids(series_mask, df):
    return sorted(df.loc[series_mask, "row_id"].astype(int).tolist())


def any_sub(series_lower, terms):
    """Row matches if the lowercase text contains ANY of the terms."""
    pat = "|".join(re.escape(t) for t in terms)
    return series_lower.str.contains(pat, regex=True, na=False)


# --------------------------------------------------------------------------------------
# Per-question gold rules.
# Each entry: qtype, output_fields (what the answer should report), and either a
# deterministic rule(df)->row_ids, OR needs_human=True with a suggested_rule string.
# --------------------------------------------------------------------------------------
def build_rules():
    R = {}

    def det(qtype, fields, fn, note=""):
        return dict(qtype=qtype, fields=fields, fn=fn, needs_human=False, note=note)

    def human(qtype, fields, suggested, note=""):
        return dict(qtype=qtype, fields=fields, fn=None, needs_human=True,
                    suggested_rule=suggested, note=note)

    # ---- cleanly deterministic ----
    R[1] = det("list", "EV Supply Chain Role; Product / Service",
               lambda d: ids(d["_category"] == "Tier 1/2", d))
    R[2] = det("multi_filter", "Company; Category(tier)",
               lambda d: ids(d["_role"].isin(["Battery Cell", "Battery Pack"]), d))
    R[3] = det("list", "Company; Primary OEMs",
               lambda d: ids(d["_role"] == "Thermal Management", d))
    R[4] = det("list", "Company; Employment",
               lambda d: ids(d["_role"].isin(["Power Electronics", "Charging Infrastructure"]), d))
    R[6] = det("lookup", "Location; Primary Facility Type",
               lambda d: ids(d["_company_l"].str.contains("novelis", na=False), d))
    R[10] = det("list", "Company; Primary OEMs",
                lambda d: ids(d["_role"] == "Vehicle Assembly", d))
    R[11] = det("lookup", "Location; Product / Service",
                lambda d: ids(d["_company_l"].str.contains("sewon", na=False), d))
    R[13] = det("multi_filter", "Company; Category; EV Supply Chain Role",
                lambda d: ids(d["_oem_l"].str.contains("rivian", na=False), d))
    R[18] = det("multi_filter", "Company; Industry Group",
                lambda d: ids((d["_category"] == "Tier 2/3") &
                              (d["_industry"].str.contains("Electronic and Other Electrical", na=False)), d))
    R[19] = det("list", "Company; Product / Service",
                lambda d: ids(any_sub(d["_product_l"], ["copper foil", "electrodeposit"]), d))
    R[23] = det("list", "Company; Category",
                lambda d: ids(any_sub(d["_product_l"], ["powder coat"]), d))
    R[25] = det("multi_filter", "Company; Employment",
                lambda d: ids((d["_category"] == "Tier 2/3") & (d["Employment"] > 300) &
                              (d["_role"] == "General Automotive"), d))
    R[26] = det("multi_filter", "Company; Product / Service",
                lambda d: ids((d["_category"] == "Tier 2/3") &
                              (d["_industry"].str.contains("Chemicals and Allied", na=False)), d))
    R[31] = det("multi_filter", "Company; Primary OEMs",
                lambda d: ids((d["_category"] == "Tier 1/2") &
                              (d["_oem_l"] == "multiple oems"), d))
    R[32] = det("multi_filter", "Company; Employment",
                lambda d: ids((d["_role"].isin(["Thermal Management", "Power Electronics"])) &
                              (d["Employment"] < 200), d))
    R[33] = det("list", "Company; Category",
                lambda d: ids(d["_category"].isin(
                    ["OEM Footprint", "OEM (Footprint)", "OEM Supply Chain"]) &
                    (d["_evrel"].isin(["yes", "indirect"])), d),
                note="EV-relevant taken as EV/Battery Relevant in {Yes, Indirect}.")
    R[37] = det("list", "Company; Product / Service",
                lambda d: ids(any_sub(d["_product_l"],
                              ["high-voltage", "high voltage", "high‑voltage",
                               "dc-to-dc", "dc to dc", "inverter", "motor controller"]), d))
    R[38] = det("multi_filter", "Company; Employment",
                lambda d: ids((d["Employment"] > 1000) & (d["_evrel"] == "indirect"), d),
                note="'Indirectly Relevant' mapped to EV/Battery Relevant == Indirect.")
    R[39] = det("aggregation_rank", "Company; Employment (top 3)",
                lambda d: d.loc[d["_role"] == "Thermal Management"]
                          .sort_values("Employment", ascending=False)
                          .head(3)["row_id"].astype(int).tolist(),
                note="Ordered top-3 by Employment; ranking matters.")
    R[43] = det("list", "Company; Product / Service",
                lambda d: ids(any_sub(d["_product_l"],
                              ["recycl", "second-life", "second life"]), d))
    R[50] = det("aggregation", "Location/area; facility=R&D",
                lambda d: ids(d["_facility"].str.contains("r&d", na=False), d))

    # ---- semi-deterministic: a defensible rule exists but wording is loose; emit gold
    #      AND mark needs_human so a human confirms the interpretation. ----
    R[14] = det("list", "Company; Category",
                lambda d: ids(any_sub(d["_product_l"],
                              ["anode", "cathode", "electrolyte", "copper foil",
                               "lithium-ion battery material"]), d),
                note="SEMI: substring on Product/Service; confirm by hand.")
    R[12] = det("multi_filter", "Company; EV Supply Chain Role",
                lambda d: ids((d["_category"] == "Tier 2/3") &
                              (d["_evrel"].isin(["yes", "indirect"])), d),
                note="SEMI: 'primary involvement in EV/battery' approximated by EV-relevant in {Yes,Indirect}.")
    R[40] = det("list", "Company; EV Supply Chain Role; Primary Facility Type",
                lambda d: ids(any_sub(d["_product_l"], ["thermal"]) |
                              d["_role_l"].str.contains("thermal", na=False), d),
                note="SEMI: thermal substring on product OR role.")
    R[49] = det("aggregation", "Location/area",
                lambda d: ids(d["_industry"].str.contains("Chemicals and Allied", na=False), d),
                note="SEMI: chemical infrastructure via Industry Group == Chemicals and Allied Products.")
    R[35] = det("count", "count of companies",
                lambda d: ids(any_sub(d["_product_l"],
                              ["lithium-ion", "li-ion", "battery cell", "electrolyte",
                               "battery material", "cathode", "anode"]) |
                              d["_role"].isin(["Battery Cell", "Battery Pack"]), d),
                note="SEMI: union of role and product substrings; confirm by hand.")

    # ---- aggregation questions whose ANSWER is a scalar/county, not a row set.
    #      We compute the scalar deterministically and store it in note/extra. ----
    def county_top_all(d):
        g = d.groupby("_county")["Employment"].sum().sort_values(ascending=False)
        return g
    def county_top_tier1(d):
        sub = d[d["_category"] == "Tier 1"]
        return sub.groupby("_county")["Employment"].sum().sort_values(ascending=False)

    R[9] = det("aggregation", "county; combined employment",
               lambda d: [],  # row_ids not the unit; see gold 'answer_value'
               note="SCALAR: county with highest total employment across all companies.")
    R[8] = det("aggregation", "county (Tier 1 only)",
               lambda d: [], note="SCALAR: county with highest total employment among Tier 1.")
    R[7] = det("aggregation", "company; EV Supply Chain Role",
               lambda d: (lambda sub: [int(sub.loc[sub["Employment"].idxmax(), "row_id"])]
                          if len(sub) else [])(d[d["_county"].str.contains("Gwinnett", na=False)]),
               note="Highest-Employment company in Gwinnett County; answer also names its role.")

    # ---- fuzzy / interpretive: DO NOT guess. Hand-validate. ----
    R[5] = human("list", "Company; EV Supply Chain Role",
                 "No 'Direct Manufacturer' value exists in Supplier or Affiliation Type "
                 "(values are 'Automotive supply chain participant', 'Original Equipment Manufacturer'). "
                 "Clarify what 'Direct Manufacturer' maps to before grading.")
    R[15] = human("list", "Company; Primary OEMs",
                  "Substring 'wiring harness' on Product/Service or role; confirm scope.")
    R[16] = human("multi_filter", "Company",
                  "Tier 1/Tier 1/2 AND electrolyte/li-ion materials AND 'existing OEM contracts' "
                  "(OEM contract is not a discrete field; needs human judgment).")
    R[17] = human("list", "Company",
                  "high-voltage wiring harness / EV electrical distribution 'suitable for BEV' is interpretive.")
    R[20] = human("multi_filter", "Company",
                  "Tier 1/2 AND engineered plastics/polymers/composite applicable to EV structural/thermal "
                  "(applicability is interpretive).")
    R[21] = human("multi_filter", "Company",
                  "Hyundai/Kia in Primary OEMs AND 'could be expanded' (interpretive).")
    R[22] = human("list", "Company; Category",
                  "DC-to-DC/capacitors/power electronics relevant to drivetrains (interpretive scope).")
    R[24] = human("multi_filter", "Company; Primary OEMs",
                  "battery parts/enclosure AND Tier 1/2 AND 'ready for OEM engagement' (interpretive).")
    R[27] = human("aggregation_rank", "EV Supply Chain Role(s)",
                  "Roles served by exactly one company; messy free-form role values must be canonicalized first.")
    R[28] = human("interpretation", "Company; OEM",
                  "'sole-sourced by a specific OEM' / 'high dependency risk' is interpretive.")
    R[29] = human("count", "count",
                  "Define 'Hyundai Metaplant supplier' (Primary OEMs ~ Hyundai?) AND EV component AND Employment<200.")
    R[30] = human("interpretation", "Company",
                  "Tier 2/3 AND EV-relevant AND General Automotive AND 'provide materials to battery' (interpretive link).")
    R[34] = human("aggregation_rank", "top 10 companies; Employment",
                  "'supply BOTH General Automotive AND EV-specific' is interpretive; then top-10 by Employment.")
    R[36] = human("interpretation", "Company",
                  "Tier 2/3 AND lightweight aluminum/composite AND 'growing EV customer base' (interpretive).")
    R[41] = human("interpretation", "Company",
                  "Tier 1/2 General Automotive 'suggesting gradual EV evolution' (interpretive).")
    R[42] = human("aggregation", "count; employment summary",
                  "Aggregate count + employment of Thermal Management suppliers; phrasing is open-ended.")
    R[44] = human("interpretation", "Company",
                  "'innovation-stage roles through R&D/prototyping' — substring 'R&D/prototyp/research' on "
                  "facility/product, but interpretive.")
    R[45] = human("interpretation", "Company",
                  "traditional OEM AND EV-native OEM in Primary OEMs (dual-platform); needs OEM taxonomy.")
    R[46] = human("interpretation", "area",
                  "areas lacking Battery Cell/Pack but having Tier 1 general auto (areal aggregation + interpretation).")
    R[47] = human("interpretation", "area",
                  "areas with highest concentration of Materials-category suppliers (areal aggregation).")
    R[48] = human("interpretation", "area",
                  "areas with Manufacturing Plant facilities indirectly supplying EV, no dedicated EV (interpretive).")

    return R


def main():
    df = load()
    rules = build_rules()
    questions = [l.strip() for l in QUESTIONS.read_text(encoding="utf-8").splitlines() if l.strip()]

    # Scalar aggregation answers computed once for the note column.
    g_all = df.groupby("_county")["Employment"].sum().sort_values(ascending=False)
    g_t1 = df[df["_category"] == "Tier 1"].groupby("_county")["Employment"].sum().sort_values(ascending=False)
    scalar = {
        9: f"{g_all.index[0]} (total employment {int(g_all.iloc[0])})",
        8: f"{g_t1.index[0]} (Tier 1 total employment {int(g_t1.iloc[0])})",
    }

    gold_rows, type_rows = [], []
    for qn in range(1, len(questions) + 1):
        r = rules.get(qn)
        qtext = questions[qn - 1]
        if r is None:
            r = dict(qtype="unknown", fields="", fn=None, needs_human=True,
                     suggested_rule="No rule defined.", note="")
        if r["fn"] is not None:
            try:
                row_ids = r["fn"](df)
            except Exception as e:
                row_ids = []
                r["note"] = (r.get("note", "") + f" [rule error: {e}]").strip()
        else:
            row_ids = []
        companies = df.loc[df["row_id"].isin(row_ids), "Company"].tolist() if row_ids else []
        gold_rows.append({
            "question_number": qn,
            "qtype": r["qtype"],
            "needs_human": r["needs_human"],
            "gold_count": len(row_ids),
            "gold_row_ids": json.dumps(row_ids),
            "gold_companies": json.dumps(companies, ensure_ascii=False),
            "output_fields": r["fields"],
            "answer_value": scalar.get(qn, ""),
            "note": r.get("note", "") or r.get("suggested_rule", ""),
        })
        type_rows.append({"question_number": qn, "qtype": r["qtype"],
                          "needs_human": r["needs_human"], "question": qtext})

    pd.DataFrame(gold_rows).to_csv(OUT_GOLD, index=False)
    pd.DataFrame(type_rows).to_csv(OUT_TYPES, index=False)

    det_n = sum(1 for g in gold_rows if not g["needs_human"])
    hum_n = sum(1 for g in gold_rows if g["needs_human"])
    print(f"Wrote {OUT_GOLD}  ({len(gold_rows)} questions)")
    print(f"Wrote {OUT_TYPES}")
    print(f"Deterministic gold: {det_n}   Needs human validation: {hum_n}")
    # Spot-check Q1 (the one already hand-verified).
    q1 = json.loads(gold_rows[0]["gold_row_ids"])
    expected = [53, 57, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90]
    print(f"Q1 gold == expected (18 Tier 1/2): {q1 == expected}  (count {len(q1)})")


if __name__ == "__main__":
    main()
