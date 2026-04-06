from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path("/home/edogan/Downloads/ertugrul/myenv/medical-rag")
REPORTS_CSV = PROJECT_ROOT / "indiana_reports.csv"
PROJECTIONS_CSV = PROJECT_ROOT / "indiana_projections.csv"
IMAGES_DIR = PROJECT_ROOT / "images/images_normalized"

OUTPUT_JSONL = PROJECT_ROOT / "data/indiana_manifest.jsonl"
OUTPUT_CSV = PROJECT_ROOT / "data/indiana_manifest.csv"

TEST_RATIO = 0.20
SEED = 42


def safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def clean_text(text: str) -> str:
    text = safe_str(text)
    text = re.sub(r"\bXXXX\b", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    return text.strip()


def split_terms(text: str) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    return [term.strip() for term in text.split(";") if term.strip()]


def build_report(findings: str, impression: str) -> str:
    parts = []
    if findings:
        parts.append(f"Findings: {findings}")
    if impression:
        parts.append(f"Impression: {impression}")
    return " ".join(parts).strip()


def assign_splits(uids: list[int], test_ratio: float, seed: int) -> dict[int, str]:
    unique_uids = sorted(set(uids))
    rng = random.Random(seed)
    rng.shuffle(unique_uids)

    n_test = int(len(unique_uids) * test_ratio)
    test_uids = set(unique_uids[:n_test])

    return {uid: ("test" if uid in test_uids else "train") for uid in unique_uids}


def main() -> None:
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    reports = pd.read_csv(REPORTS_CSV)
    projections = pd.read_csv(PROJECTIONS_CSV)

    reports.columns = [c.strip() for c in reports.columns]
    projections.columns = [c.strip() for c in projections.columns]

    projections["projection"] = projections["projection"].astype(str).str.strip().str.lower()

    frontal = (
        projections[projections["projection"] == "frontal"]
        .drop_duplicates(subset=["uid"])
        .copy()
    )
    lateral = (
        projections[projections["projection"] == "lateral"]
        .drop_duplicates(subset=["uid"])
        .copy()
    )

    frontal = frontal.rename(columns={"filename": "frontal_filename"})
    lateral = lateral.rename(columns={"filename": "lateral_filename"})

    df = reports.merge(
        frontal[["uid", "frontal_filename"]],
        on="uid",
        how="left"
    ).merge(
        lateral[["uid", "lateral_filename"]],
        on="uid",
        how="left"
    )

    records = []

    for _, row in df.iterrows():
        uid = int(row["uid"])

        findings = clean_text(row.get("findings", ""))
        impression = clean_text(row.get("impression", ""))
        indication = clean_text(row.get("indication", ""))
        comparison = clean_text(row.get("comparison", ""))

        mesh_terms = split_terms(row.get("MeSH", ""))
        problem_terms = split_terms(row.get("Problems", ""))

        frontal_filename = safe_str(row.get("frontal_filename", ""))
        lateral_filename = safe_str(row.get("lateral_filename", ""))

        frontal_abs = IMAGES_DIR / frontal_filename if frontal_filename else None
        lateral_abs = IMAGES_DIR / lateral_filename if lateral_filename else None

        frontal_path = f"images/images_normalized/{frontal_filename}" if frontal_filename else ""
        lateral_path = f"images/images_normalized/{lateral_filename}" if lateral_filename else ""

        if frontal_abs and not frontal_abs.exists():
            frontal_path = ""

        if lateral_abs and not lateral_abs.exists():
            lateral_path = ""

        primary_image_path = frontal_path if frontal_path else lateral_path
        report = build_report(findings, impression)

        record = {
            "uid": uid,
            "primary_image_path": primary_image_path,
            "frontal_image_path": frontal_path,
            "lateral_image_path": lateral_path,
            "findings": findings,
            "impression": impression,
            "report": report,
            "mesh_terms": mesh_terms,
            "problem_terms": problem_terms,
            "indication": indication,
            "comparison": comparison,
        }
        records.append(record)

    manifest_df = pd.DataFrame(records)

    manifest_df["has_report"] = manifest_df["report"].str.len() > 0
    manifest_df["has_image"] = manifest_df["primary_image_path"].str.len() > 0

    clean_df = manifest_df[(manifest_df["has_report"]) & (manifest_df["has_image"])].copy()

    split_map = assign_splits(clean_df["uid"].tolist(), TEST_RATIO, SEED)
    clean_df["split"] = clean_df["uid"].map(split_map)

    final_columns = [
        "uid",
        "primary_image_path",
        "frontal_image_path",
        "lateral_image_path",
        "findings",
        "impression",
        "report",
        "mesh_terms",
        "problem_terms",
        "indication",
        "comparison",
        "split",
    ]
    clean_df = clean_df[final_columns]

    clean_df.to_csv(OUTPUT_CSV, index=False)

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for rec in clean_df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Total reports rows       : {len(reports)}")
    print(f"Total projection rows    : {len(projections)}")
    print(f"Manifest rows (all)      : {len(manifest_df)}")
    print(f"Clean rows (img+report)  : {len(clean_df)}")
    print(f"Train rows               : {(clean_df['split'] == 'train').sum()}")
    print(f"Test rows                : {(clean_df['split'] == 'test').sum()}")
    print(f"Saved CSV   -> {OUTPUT_CSV}")
    print(f"Saved JSONL -> {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()