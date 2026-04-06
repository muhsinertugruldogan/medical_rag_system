from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, List, Optional

import requests

BASE_DIR = Path(__file__).resolve().parent.parent

QUERIES_CSV = BASE_DIR / "queries.csv"
OUTPUT_CSV = BASE_DIR / "performance_results_api.csv"
MANIFEST_PATH = BASE_DIR / "data" / "indiana_manifest.jsonl"

API_URL = "http://localhost:8000/query"
REQUEST_TIMEOUT = 300  # seconds


def load_queries(path: Path) -> List[Dict[str, str]]:
    queries: List[Dict[str, str]] = []

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query = str(row.get("query", "")).strip()
            if not query:
                continue

            qlen = str(row.get("length", "")).strip().lower()
            if qlen not in {"short", "medium", "long"}:
                wc = len(query.split())
                if wc < 8:
                    qlen = "short"
                elif wc < 18:
                    qlen = "medium"
                else:
                    qlen = "long"

            queries.append(
                {
                    "query": query,
                    "length": qlen,
                }
            )

    return queries


def load_test_image_paths(manifest_path: Path) -> List[str]:
    image_paths: List[str] = []

    import json

    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            rec = json.loads(line)
            if rec.get("split") == "test":
                image_path = str(rec.get("primary_image_path", "")).strip()
                if image_path:
                    image_paths.append(image_path)

    if not image_paths:
        raise ValueError("No test image paths found in manifest.")

    return image_paths


def safe_get_latency(data: Dict, key: str) -> Optional[float]:
    try:
        return float(data.get("latency_ms", {}).get(key))
    except Exception:
        return None


def main() -> None:
    random.seed(42)

    queries = load_queries(QUERIES_CSV)
    image_paths = load_test_image_paths(MANIFEST_PATH)

    rows: List[Dict[str, object]] = []

    for idx, item in enumerate(queries, start=1):
        query = item["query"]
        query_length = item["length"]

        # Karışık set: yarısı image'lı
        has_image = (idx % 2 == 0)
        image_path = random.choice(image_paths) if has_image else ""

        payload = {"question": query}
        if has_image:
            payload["image_path"] = image_path

        error_msg = ""
        answer = ""
        top_source_uids = ""
        retrieval_ms = ""
        generation_ms = ""
        total_ms = ""

        try:
            response = requests.post(
                API_URL,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code != 200:
                error_msg = f"HTTP {response.status_code}: {response.text[:300]}"
            else:
                data = response.json()

                answer = str(data.get("answer", "")).replace("\n", " | ")

                sources = data.get("sources", []) or []
                top_source_uids = ",".join(
                    str(src.get("uid", "")) for src in sources[:3]
                )

                retrieval_val = safe_get_latency(data, "retrieval_ms")
                generation_val = safe_get_latency(data, "generation_ms")
                total_val = safe_get_latency(data, "total_ms")

                retrieval_ms = retrieval_val if retrieval_val is not None else ""
                generation_ms = generation_val if generation_val is not None else ""
                total_ms = total_val if total_val is not None else ""

        except Exception as e:
            error_msg = str(e)

        rows.append(
            {
                "Query_ID": idx,
                "Query": query,
                "Has_Image": has_image,
                "Query_Length": query_length,
                "Image_Path": image_path,
                "Retrieval_Time_MS": retrieval_ms,
                "Generation_Time_MS": generation_ms,
                "Total_Time_MS": total_ms,
                "Top_Source_UIDs": top_source_uids,
                "Answer": answer,
                "Error": error_msg,
            }
        )

        print(f"[{idx}/{len(queries)}] done")

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Query_ID",
                "Query",
                "Has_Image",
                "Query_Length",
                "Image_Path",
                "Retrieval_Time_MS",
                "Generation_Time_MS",
                "Total_Time_MS",
                "Top_Source_UIDs",
                "Answer",
                "Error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()