from __future__ import annotations

import json
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

TEXT_TOP_K = 10
IMAGE_TOP_K = 8
RERANK_TOP_K = 6


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def pick_test_record(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    for rec in records:
        if rec.get("split") == "test" and str(rec.get("primary_image_path", "")).strip():
            return rec
    raise ValueError("No suitable test record found.")


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


def main() -> None:
    records = load_manifest(MANIFEST_PATH)
    rec = pick_test_record(records)

    question = "Is there evidence of acute cardiopulmonary abnormality in this chest X-ray?"

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
        max_new_tokens=100,
    )

    t0 = time.perf_counter()

    text_query_embedding = text_embedder.encode_query(question)
    t1 = time.perf_counter()

    image_query_embedding = image_embedder.encode_image(rec["primary_image_path"])
    t2 = time.perf_counter()

    text_results = text_store.query(
        query_embedding=text_query_embedding,
        n_results=TEXT_TOP_K,
    )
    image_results = image_store.query(
        query_embedding=image_query_embedding,
        n_results=IMAGE_TOP_K,
    )
    t3 = time.perf_counter()

    text_candidates = parse_results(text_results, source_type="text")
    image_candidates = parse_results(image_results, source_type="image")
    pooled = pool_candidates(text_candidates, image_candidates)

    reranked = reranker.rerank(
        question=question,
        candidates=pooled,
        top_k=RERANK_TOP_K,
    )
    t4 = time.perf_counter()

    answer = generator.generate_answer(
        question=question,
        retrieved_contexts=reranked,
        image_findings=None,
    )
    t5 = time.perf_counter()

    output = {
        "query_uid": rec["uid"],
        "question": question,
        "image_path": rec["primary_image_path"],
        "final_sources": [
            {
                "uid": c["uid"],
                "impression": c["impression"],
                "from_text": c["from_text"],
                "from_image": c["from_image"],
                "text_rank": c.get("text_rank"),
                "image_rank": c.get("image_rank"),
                # "rerank_score_raw": c.get("rerank_score_raw"),
                "rerank_score": c.get("rerank_score"),
            }
            for c in reranked
        ],
        "answer": answer,
        "timing": {
            "text_embedding_ms": round((t1 - t0) * 1000, 2),
            "image_embedding_ms": round((t2 - t1) * 1000, 2),
            "retrieval_ms": round((t3 - t2) * 1000, 2),
            "reranking_ms": round((t4 - t3) * 1000, 2),
            "generation_ms": round((t5 - t4) * 1000, 2),
            "total_ms": round((t5 - t0) * 1000, 2),
        },
    }

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()