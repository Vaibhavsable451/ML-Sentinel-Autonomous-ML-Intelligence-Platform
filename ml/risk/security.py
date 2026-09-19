"""ML Security module: input validation, PII handling, artifact integrity, dependency scanning.

Lives under ml/risk because its output feeds the risk engine's security sub-score.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"\b(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
SSN_LIKE_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,16}\b")


@dataclass
class PIIFinding:
    field: str
    pattern: str
    sample_masked: str


def detect_pii_in_text(value: str) -> list[str]:
    hits = []
    if EMAIL_RE.search(value):
        hits.append("email")
    if PHONE_RE.search(value):
        hits.append("phone")
    if SSN_LIKE_RE.search(value):
        hits.append("ssn_like")
    if CREDIT_CARD_RE.search(value.replace(" ", "")):
        hits.append("credit_card_like")
    return hits


def mask_value(value: str) -> str:
    value = EMAIL_RE.sub(lambda m: m.group(0)[0] + "***@***", value)
    value = SSN_LIKE_RE.sub("***-**-****", value)
    value = PHONE_RE.sub("***-***-****", value)
    return value


def scan_dataframe_for_pii(df, sample_size: int = 200) -> list[PIIFinding]:
    findings = []
    text_cols = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype).startswith("string")]
    for col in dict.fromkeys(text_cols):
        sample = df[col].dropna().astype(str).head(sample_size)
        hit_types = set()
        example = None
        for v in sample:
            hits = detect_pii_in_text(v)
            if hits:
                hit_types.update(hits)
                example = example or v
        if hit_types:
            findings.append(PIIFinding(field=col, pattern=",".join(sorted(hit_types)), sample_masked=mask_value(example or "")))
    return findings


# ---- input schema validation (API-boundary security) ----

class SchemaValidationError(Exception):
    pass


def validate_input_schema(payload: dict, expected_schema: dict) -> None:
    """expected_schema: {field_name: python_type}. Raises SchemaValidationError on mismatch."""
    missing = [f for f in expected_schema if f not in payload]
    if missing:
        raise SchemaValidationError(f"missing required fields: {missing}")
    for field_name, expected_type in expected_schema.items():
        val = payload[field_name]
        if not isinstance(val, expected_type):
            raise SchemaValidationError(f"field '{field_name}' expected {expected_type}, got {type(val)}")
    # basic payload-size / injection guard
    serialized = json.dumps(payload)
    if len(serialized) > 50_000:
        raise SchemaValidationError("payload too large")
    if re.search(r"(<script|DROP TABLE|;--|\$where)", serialized, re.IGNORECASE):
        raise SchemaValidationError("payload contains suspicious content")


# ---- model artifact integrity ----

def compute_artifact_checksum(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_artifact_checksum(path: str | Path, expected_sha256: str) -> bool:
    return compute_artifact_checksum(path) == expected_sha256


# ---- dependency vulnerability scan (uses pip-audit if available) ----

@dataclass
class DependencyScanResult:
    ran: bool
    vulnerable_packages: list = field(default_factory=list)
    raw_output: str = ""


def scan_dependencies(requirements_path: str = "requirements.txt") -> DependencyScanResult:
    try:
        result = subprocess.run(
            ["pip-audit", "-r", requirements_path, "-f", "json"],
            capture_output=True, text=True, timeout=60, check=False,
        )
        data = json.loads(result.stdout or "[]")
        vulnerable = [d["name"] for d in data.get("dependencies", []) if d.get("vulns")]
        return DependencyScanResult(ran=True, vulnerable_packages=vulnerable, raw_output=result.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return DependencyScanResult(ran=False, raw_output="pip-audit not available in this environment; run in CI.")


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("data/train_raw.csv")
    findings = scan_dataframe_for_pii(df)
    for f in findings:
        print(f)

    try:
        validate_input_schema({"age": 30, "income": 50000}, {"age": int, "income": (int, float)})
        print("schema OK")
    except SchemaValidationError as e:
        print("schema error:", e)

    try:
        validate_input_schema({"age": "<script>bad</script>"}, {"age": str})
    except SchemaValidationError as e:
        print("caught malicious payload:", e)
