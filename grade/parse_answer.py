"""
parse_answer.py -- format-agnostic extraction of the entities a model named.

Full-context models emit prose / markdown; the batched model emits a rigid table.
To score them on one axis we extract, from any answer text:
  - the set of KB companies the answer mentions (primary signal), and
  - any explicit row_id integers it printed (secondary signal).

Company matching is anchored to the KB master list (we only ever credit/penalize
companies that actually exist in the KB), which also makes hallucination measurable:
a "company-like" mention that matches no KB row is out-of-KB.
"""

import re
import unicodedata

import pandas as pd

_SUFFIX = re.compile(
    r"\b(inc|inc\.|llc|l\.l\.c|co|co\.|corp|corp\.|ltd|ltd\.|company|"
    r"americas|america|usa|u\.s\.a|manufacturing|mfg|group|holdings?|gmbh)\b",
    re.I,
)


def norm(text):
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def norm_company(name):
    n = norm(name)
    n = _SUFFIX.sub(" ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def load_kb_companies(xlsx):
    df = pd.read_excel(xlsx)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all").reset_index(drop=True)
    df["row_id"] = df.index + 1
    companies = []
    for _, r in df.iterrows():
        raw = str(r["Company"]).strip()
        companies.append({
            "row_id": int(r["row_id"]),
            "raw": raw,
            "norm": norm_company(raw),
            "norm_full": norm(raw),
        })
    return companies


def extract_companies(answer_text, kb_companies, min_len=4):
    """Return set of KB row_ids whose company name is mentioned in the answer.

    Matching uses the suffix-stripped normalized name as a word-boundary-ish
    substring of the normalized answer. We require a reasonably distinctive name
    (>= min_len chars after stripping) to avoid spurious short matches.
    """
    hay = norm(answer_text)
    hay_padded = f" {hay} "
    hits = set()
    for c in kb_companies:
        key = c["norm"] if len(c["norm"]) >= min_len else c["norm_full"]
        if len(key) < min_len:
            continue
        if f" {key} " in hay_padded or hay_padded.find(f" {key} ") != -1 or key in hay:
            # prefer padded boundary match; fall back to plain containment for
            # multi-word names that already carry their own boundaries
            if f" {key} " in hay_padded or (" " in key and key in hay):
                hits.add(c["row_id"])
    return hits


def extract_row_ids(answer_text, valid_ids):
    """Extract integers that look like row_ids (only keep ones that are valid KB ids).

    This is a secondary signal; many answers print employment counts / years that
    are NOT row ids, so we intersect with the valid id set to stay conservative.
    """
    nums = set(int(n) for n in re.findall(r"\b\d{1,3}\b", str(answer_text)))
    return nums & set(valid_ids)


def is_error(answer_text):
    return str(answer_text).strip().upper().startswith("ERROR")
