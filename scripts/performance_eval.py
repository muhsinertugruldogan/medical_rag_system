from __future__ import annotations

import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from retrieval.config import (
    MANIFEST_PATH,
    PERSIST_DIRECTORY,
    TEXT_COLLECTION_NAME,
    IMAGE_COLLECTION_NAME,
    BIOMEDCLIP_MODEL_NAME,
)
from retrieval.embedder import TextEmbedder
from retrieval.image_embedder import BiomedClipImageEmbedder
from retrieval.vectordb import ChromaTextStore
from retrieval.reranker import MedicalCrossEncoderReranker
from generation.qwen_answer_generator import QwenAnswerGenerator

QUERIES_CSV = BASE_DIR / "queries.csv"
OUTPUT_CSV = BASE_DIR / "performance_results.csv"

TEXT_TOP_K = 10
IMAGE_TOP_K = 6
RERANK_TOP_K = 5


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


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
                word_count = len(query.split())
                if word_count < 8:
                    qlen = "short"
                elif word_count < 18:
                    qlen = "medium"
                else:
                    qlen = "long"

            queries.append({"query": query, "length": qlen})
    return queries


def parse_results(results: Dict[str, Any], source_type: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0] if results.get("distances") else [None] * len(docs)

    for rank, (doc_id, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists), start=1):
        out.append(
            {
                "id": doc_id,
                "uid": str(meta.get("uid", "")),
                "report": doc,
                "impression": str(meta.get("impression", "")),
                "distance": dist,
                "source_type": source_type,
                "rank": rank,
            }
        )
    return out


def pool_candidates(
    text_candidates: List[Dict[str, Any]],
    image_candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    pooled: Dict[str, Dict[str, Any]] = {}

    for cand in text_candidates + image_candidates:
        uid = cand["uid"]

        if uid not in pooled:
            pooled[uid] = {
                "uid": uid,
                "report": cand["report"],
                "impression": cand["impression"],
                "from_text": False,
                "from_image": False,
                "text_distance": None,
                "image_distance": None,
                "text_rank": None,
                "image_rank": None,
            }

        if cand["source_type"] == "text":
            pooled[uid]["from_text"] = True
            pooled[uid]["text_distance"] = cand["distance"]
            pooled[uid]["text_rank"] = cand["rank"]

        if cand["source_type"] == "image":
            pooled[uid]["from_image"] = True
            pooled[uid]["image_distance"] = cand["distance"]
            pooled[uid]["image_rank"] = cand["rank"]

    return list(pooled.values())


def choose_record(records: List[Dict[str, Any]], has_image: bool) -> Dict[str, Any]:
    if has_image:
        candidates = [
            r for r in records
            if r.get("split") == "test" and str(r.get("primary_image_path", "")).strip()
        ]
    else:
        candidates = [
            r for r in records
            if r.get("split") == "test" and str(r.get("report", "")).strip()
        ]

    if not candidates:
        raise ValueError("No suitable test records found.")
    return random.choice(candidates)


def main() -> None:
    random.seed(42)

    records = load_manifest(MANIFEST_PATH)
    queries = load_queries(QUERIES_CSV)

    text_embedder = TextEmbedder(model_name=BIOMEDCLIP_MODEL_NAME)
    image_embedder = BiomedClipImageEmbedder(model_name=BIOMEDCLIP_MODEL_NAME)

    text_store = ChromaTextStore(
        persist_directory=PERSIST_DIRECTORY,
        collection_name=TEXT_COLLECTION_NAME,
    )
    image_store = ChromaTextStore(
        persist_directory=PERSIST_DIRECTORY,
        collection_name=IMAGE_COLLECTION_NAME,
    )

    reranker = MedicalCrossEncoderReranker(
        model_name="ncbi/MedCPT-Cross-Encoder",
        max_length=512,
        batch_size=16,
    )

    generator = QwenAnswerGenerator(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        device_map="auto",
        torch_dtype="auto",
        max_new_tokens=100,
    )

    rows: List[Dict[str, Any]] = []

    for idx, item in enumerate(queries, start=1):
        query = item["query"]
        query_length = item["length"]

        has_image = (idx % 2 == 0)

        rec = choose_record(records, has_image=has_image)
        image_path = str(rec.get("primary_image_path", "")).strip() if has_image else ""

        t0 = time.perf_counter()

        text_query_embedding = text_embedder.encode_query(query)
        t1 = time.perf_counter()

        text_results = text_store.query(
            query_embedding=text_query_embedding,
            n_results=TEXT_TOP_K,
        )

        if has_image and image_path:
            image_query_embedding = image_embedder.encode_image(image_path)
            image_results = image_store.query(
                query_embedding=image_query_embedding,
                n_results=IMAGE_TOP_K,
            )

            text_candidates = parse_results(text_results, source_type="text")
            image_candidates = parse_results(image_results, source_type="image")
            pooled = pool_candidates(text_candidates, image_candidates)
        else:
            pooled = parse_results(text_results, source_type="text")

        t2 = time.perf_counter()

        reranked = reranker.rerank(
            question=query,
            candidates=pooled,
            top_k=RERANK_TOP_K,
        )

        answer = generator.generate_answer(
            question=query,
            retrieved_contexts=reranked,
            image_findings=None,
        )
        t3 = time.perf_counter()

        retrieval_ms = round((t2 - t1) * 1000, 2)
        generation_ms = round((t3 - t2) * 1000, 2)
        total_ms = round((t3 - t0) * 1000, 2)

        top_source_uids = ",".join([str(x.get("uid", "")) for x in reranked[:3]])

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
                "Answer": answer.replace("\n", " | "),
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
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()