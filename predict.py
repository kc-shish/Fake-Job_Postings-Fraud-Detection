from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Dict, Iterable, List, Union

import joblib
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "artifacts")

# ---------------------------------------------------------------------------
# Category normalisation maps (must mirror the notebook exactly)
# ---------------------------------------------------------------------------

EDUCATION_MAP = {
    # School level
    "High School or equivalent": "school",
    "Some High School Coursework": "school",
    "Vocational - HS Diploma": "school",
    # Undergraduate
    "Some College Coursework Completed": "undergraduate",
    "Associate Degree": "undergraduate",
    # Bachelor
    "Bachelor's Degree": "bachelor's degree",
    # Postgraduate
    "Master's Degree": "Master's Degree",
    "Doctorate": "phd completed",
    # Professional skill-based
    "Certification": "professional",
    "Professional": "professional",
    "Vocational": "professional",
    "Vocational - Degree": "professional",
    # Unknown
    "Unspecified": "unknown",
    "Unknown": "unknown",
}

TEXT_FIELDS = ["title", "company_profile", "description", "requirements", "benefits"]

RAW_FIELD_DEFAULTS = {
    "title": "",
    "company_profile": "",
    "description": "",
    "requirements": "",
    "benefits": "",
    "industry": "Unknown",
    "function": "",
    "employment_type": "Unknown",
    "required_experience": "Unknown",
    "required_education": "Unknown",
    "telecommuting": 0,
    "has_company_logo": 0,
    "has_questions": 0,
}


# ---------------------------------------------------------------------------
# Artifact / model loading (cached so Streamlit doesn't reload on every run)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_artifacts():
    """Load the fitted vectorizer, model, and expected feature-column order."""
    vectorizer = joblib.load(os.path.join(ARTIFACTS_DIR, "count_vectorizer.pkl"))
    model = joblib.load(os.path.join(ARTIFACTS_DIR, "fraud_detection_model.pkl"))
    feature_columns = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_columns.pkl"))
    return vectorizer, model, feature_columns


# ---------------------------------------------------------------------------
# Text cleaning / tokenisation
#
# NOTE: the original notebook used spaCy ("en_core_web_sm") for lemmatisation.
# This module instead uses a dependency-free tokenizer (regex + scikit-learn's
# built-in stopword list) so the app doesn't require downloading an external
# spaCy model at deploy time. If you retrain on the real dataset with spaCy
# lemmatisation, swap this function out for the notebook's preprocess_batch
# and retrain count_vectorizer.pkl / fraud_detection_model.pkl to match.
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"_+", " ", text)
    return text


def _preprocess_one(text: str) -> str:
    text = _clean_text(text)
    tokens = re.findall(r"[a-z']+", text)
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


def preprocess_texts(texts: Iterable[str]) -> List[str]:
    """Clean + tokenise a batch of raw text strings."""
    return [_preprocess_one(t) for t in texts]


# ---------------------------------------------------------------------------
# Raw-input normalisation (mirrors notebook's cleaning / feature-engineering)
# ---------------------------------------------------------------------------

def _normalise_record(raw: Dict) -> Dict:
    """Fill defaults + apply the same category remapping used in training."""
    rec = {**RAW_FIELD_DEFAULTS, **{k: v for k, v in raw.items() if v is not None}}

    # Text fields: never allow NaN/None to reach string concatenation.
    for col in TEXT_FIELDS + ["industry", "function"]:
        if rec.get(col) is None or (isinstance(rec[col], float) and pd.isna(rec[col])):
            rec[col] = "Unknown" if col == "industry" else ""

    # employment_type: "Other" was folded into "Unknown" during training.
    employment_type = rec.get("employment_type") or "Unknown"
    if employment_type == "Other":
        employment_type = "Unknown"
    rec["employment_type"] = employment_type

    # required_experience: used as-is (fillna -> "Unknown").
    rec["required_experience"] = rec.get("required_experience") or "Unknown"

    # required_education: "Unknown" -> "Unspecified" -> mapped grouping.
    req_edu = rec.get("required_education") or "Unknown"
    rec["required_education_mapped"] = EDUCATION_MAP.get(req_edu, "unknown")

    # Binary flags -> int 0/1.
    for col in ["telecommuting", "has_company_logo", "has_questions"]:
        rec[col] = int(bool(rec.get(col, 0)))

    return rec


def _build_text(rec: Dict) -> str:
    """Recreate df['text'] = title + company_profile + description + ..."""
    return " ".join(
        [
            rec["title"],
            rec["company_profile"],
            rec["description"],
            rec["requirements"],
            rec["benefits"],
            rec["industry"],
            rec["function"],
        ]
    )


def _build_numeric_row(rec: Dict, feature_columns: List[str]) -> List[int]:
    """
    Recreate the one-hot / binary columns in the exact order the model
    expects (feature_columns), e.g. 'employment_type_Full-time',
    'required_experience_Mid-Senior level', 'required_education_unknown'...
    """
    row = []
    for col in feature_columns:
        if col == "telecommuting":
            row.append(rec["telecommuting"])
        elif col == "has_company_logo":
            row.append(rec["has_company_logo"])
        elif col == "has_questions":
            row.append(rec["has_questions"])
        elif col.startswith("employment_type_"):
            value = col[len("employment_type_"):]
            row.append(int(rec["employment_type"] == value))
        elif col.startswith("required_experience_"):
            value = col[len("required_experience_"):]
            row.append(int(rec["required_experience"] == value))
        elif col.startswith("required_education_"):
            value = col[len("required_education_"):]
            row.append(int(rec["required_education_mapped"] == value))
        else:
            # Unknown/unused column defensively defaults to 0.
            row.append(0)
    return row


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_feature_matrix(records: Union[Dict, List[Dict]]):
    """
    Turn one or more raw job-posting dicts into the sparse feature matrix
    the model expects (same column order used during training).
    """
    if isinstance(records, dict):
        records = [records]

    vectorizer, _model, feature_columns = load_artifacts()

    normalised = [_normalise_record(r) for r in records]
    raw_texts = [_build_text(r) for r in normalised]
    lemmatised_texts = preprocess_texts(raw_texts)

    X_text = vectorizer.transform(lemmatised_texts)

    numeric_rows = [_build_numeric_row(r, feature_columns) for r in normalised]
    X_numeric = csr_matrix(pd.DataFrame(numeric_rows, columns=feature_columns).values)

    return hstack([X_text, X_numeric])


def predict_fraud(records: Union[Dict, List[Dict]]) -> List[Dict]:
    """
    Score one or more raw job postings.

    Parameters
    ----------
    records : dict or list[dict]
        Each dict may contain any of: title, company_profile, description,
        requirements, benefits, industry, function, employment_type,
        required_experience, required_education, telecommuting,
        has_company_logo, has_questions. Missing fields fall back to
        sensible defaults (mirroring how NaNs were handled in training).

    Returns
    -------
    list[dict]
        One result per input record:
        {"is_fraudulent": bool, "fraud_probability": float, "label": str}
    """
    single_input = isinstance(records, dict)
    if single_input:
        records = [records]

    _vectorizer, model, _feature_columns = load_artifacts()
    X = build_feature_matrix(records)

    preds = model.predict(X)
    probas = model.predict_proba(X)[:, 1]

    results = [
        {
            "is_fraudulent": bool(pred),
            "fraud_probability": float(proba),
            "label": "Fraudulent" if pred else "Legitimate",
        }
        for pred, proba in zip(preds, probas)
    ]
    return results


if __name__ == "__main__":
    # Simple smoke test / usage example.
    example_job = {
        "title": "Work From Home Data Entry - No Experience Needed!",
        "company_profile": "",
        "description": "Earn $5000 a week working from home. Just send your bank details to get started immediately.",
        "requirements": "No experience required. Must have a computer.",
        "benefits": "",
        "industry": "Unknown",
        "function": "",
        "employment_type": "Full-time",
        "required_experience": "Not Applicable",
        "required_education": "Unspecified",
        "telecommuting": 1,
        "has_company_logo": 0,
        "has_questions": 0,
    }
    print(predict_fraud(example_job))