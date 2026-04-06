from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from retrieval.config import (
    MANIFEST_PATH,
    PERSIST_DIRECTORY,
    TEXT_COLLECTION_NAME,
    BIOMEDCLIP_MODEL_NAME,
)
from retrieval.embedder import TextEmbedder
from retrieval.vectordb import ChromaTextStore

TOP_K = 5


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_test_examples(records: List[Dict[str, Any]], n: int = 5) -> List[Dict[str, Any]]:
    out = []
    for rec in records:
        if rec.get("split") == "test" and str(rec.get("report", "")).strip():
            out.append(rec)
        if len(out) >= n:
            break
    return out


def pretty_print_result(query_rec: Dict[str, Any], results: Dict[str, Any]) -> None:
    print("=" * 100)
    print(f"QUERY UID: {query_rec['uid']}")
    print(f"QUERY REPORT: {query_rec['report']}")
    print("-" * 100)

    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0] if "distances" in results else [None] * len(ids)

    for rank, (doc_id, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists), start=1):
        print(f"[{rank}] {doc_id} | uid={meta.get('uid')} | distance={dist}")
        print(f"Impression: {meta.get('impression', '')}")
        print(f"Report: {doc}")
        print("-" * 100)


def main() -> None:
    records = load_manifest(MANIFEST_PATH)
    test_examples = get_test_examples(records, n=5)

    embedder = TextEmbedder(model_name=BIOMEDCLIP_MODEL_NAME)
    store = ChromaTextStore(
        persist_directory=PERSIST_DIRECTORY,
        collection_name=TEXT_COLLECTION_NAME,
    )

    print(f"Text collection count: {store.count()}")

    for rec in test_examples:
        query_embedding = embedder.encode_query(rec["report"])
        results = store.query(
            query_embedding=query_embedding,
            n_results=TOP_K,
        )
        pretty_print_result(rec, results)


if __name__ == "__main__":
    main()