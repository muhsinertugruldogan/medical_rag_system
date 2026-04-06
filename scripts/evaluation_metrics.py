from __future__ import annotations
import nltk

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")
    nltk.download("omw-1.4")
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

import requests
from bert_score import score as bertscore_score
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

BASE_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = BASE_DIR / "data" / "indiana_manifest.jsonl"
OUTPUT_CSV = BASE_DIR / "evaluation_metrics_results.csv"
OUTPUT_JSON = BASE_DIR / "evaluation_metrics_summary.json"

API_URL = "http://localhost:8000/query"
REQUEST_TIMEOUT = 300

QUESTION = "What are the main radiographic findings in this chest X-ray?"
REFERENCE_FIELD = "impression"   
MAX_SAMPLES = 50          


def load_test_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("split") == "test" and rec.get("primary_image_path") and rec.get(REFERENCE_FIELD):
                records.append(rec)
    return records


def extract_eval_text(answer_text: str) -> str:
    
    lines = [line.strip() for line in answer_text.splitlines() if line.strip()]

    answer_line = next((l for l in lines if l.lower().startswith("answer:")), "")
    evidence_line = next((l for l in lines if l.lower().startswith("evidence summary:")), "")

    answer_clean = answer_line[len("Answer:"):].strip() if answer_line else ""
    evidence_clean = evidence_line[len("Evidence summary:"):].strip() if evidence_line else ""

    combined = " ".join([x for x in [answer_clean, evidence_clean] if x]).strip()
    return combined if combined else answer_text.strip()


def call_api(question: str, image_path: str) -> Dict[str, Any]:
    payload = {
        "question": question,
        "image_path": image_path,
    }
    response = requests.post(API_URL, json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def compute_bleu(pred: str, ref: str) -> float:
    smoothie = SmoothingFunction().method1
    return sentence_bleu(
        [ref.split()],
        pred.split(),
        smoothing_function=smoothie,
    )


def compute_meteor(pred: str, ref: str) -> float:
    return meteor_score([ref.split()], pred.split())


def main() -> None:
    records = load_test_records(MANIFEST_PATH)
    if MAX_SAMPLES is not None:
        records = records[:MAX_SAMPLES]

    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    rows: List[Dict[str, Any]] = []
    preds_for_bert: List[str] = []
    refs_for_bert: List[str] = []

    for idx, rec in enumerate(records, start=1):
        uid = rec["uid"]
        image_path = rec["primary_image_path"]
        reference = str(rec[REFERENCE_FIELD]).strip()

        error_msg = ""
        raw_answer = ""
        pred_text = ""

        bleu = None
        rouge1 = None
        rouge2 = None
        rougeL = None
        meteor = None

        try:
            data = call_api(QUESTION, image_path=image_path)
            raw_answer = str(data.get("answer", "")).strip()
            pred_text = extract_eval_text(raw_answer)

            bleu = compute_bleu(pred_text, reference)

            rouge_scores = rouge.score(reference, pred_text)
            rouge1 = rouge_scores["rouge1"].fmeasure
            rouge2 = rouge_scores["rouge2"].fmeasure
            rougeL = rouge_scores["rougeL"].fmeasure

            meteor = compute_meteor(pred_text, reference)

            preds_for_bert.append(pred_text)
            refs_for_bert.append(reference)

        except Exception as e:
            error_msg = str(e)

        rows.append(
            {
                "uid": uid,
                "question": QUESTION,
                "image_path": image_path,
                "reference": reference,
                "raw_answer": raw_answer,
                "prediction_for_eval": pred_text,
                "bleu": bleu,
                "rouge1_f": rouge1,
                "rouge2_f": rouge2,
                "rougeL_f": rougeL,
                "meteor": meteor,
                "bertscore_f1": None,   # sonra dolduracağız
                "error": error_msg,
            }
        )

        print(f"[{idx}/{len(records)}] done")

    # BERTScore
    valid_indices = [i for i, r in enumerate(rows) if not r["error"]]
    if preds_for_bert and refs_for_bert:
        _, _, f1 = bertscore_score(preds_for_bert, refs_for_bert, lang="en", verbose=True)

        bert_idx = 0
        for i in valid_indices:
            rows[i]["bertscore_f1"] = float(f1[bert_idx])
            bert_idx += 1

    # CSV
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "uid",
                "question",
                "image_path",
                "reference",
                "raw_answer",
                "prediction_for_eval",
                "bleu",
                "rouge1_f",
                "rouge2_f",
                "rougeL_f",
                "meteor",
                "bertscore_f1",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    valid_rows = [r for r in rows if not r["error"]]

    def safe_mean(key: str):
        vals = [float(r[key]) for r in valid_rows if r[key] is not None and r[key] != ""]
        return mean(vals) if vals else None

    summary = {
        "num_total": len(rows),
        "num_success": len(valid_rows),
        "num_failed": len(rows) - len(valid_rows),
        "question": QUESTION,
        "reference_field": REFERENCE_FIELD,
        "avg_bleu": safe_mean("bleu"),
        "avg_rouge1_f": safe_mean("rouge1_f"),
        "avg_rouge2_f": safe_mean("rouge2_f"),
        "avg_rougeL_f": safe_mean("rougeL_f"),
        "avg_meteor": safe_mean("meteor"),
        "avg_bertscore_f1": safe_mean("bertscore_f1"),
    }

    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved CSV   -> {OUTPUT_CSV}")
    print(f"Saved JSON  -> {OUTPUT_JSON}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()