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

BATCH_SIZE = 32
RESET_COLLECTION = True


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def prepare_train_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for rec in records:
        if rec.get("split") != "train":
            continue
        if not str(rec.get("report", "")).strip():
            continue
        if not str(rec.get("primary_image_path", "")).strip():
            continue
        out.append(rec)
    return out


def chunked(seq: List[Any], size: int) -> List[List[Any]]:
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def main() -> None:
    print(f"Loading manifest from: {MANIFEST_PATH}")
    print(f"Text collection: {TEXT_COLLECTION_NAME}")
    print(f"Text model: {BIOMEDCLIP_MODEL_NAME}")

    records = load_manifest(MANIFEST_PATH)
    train_records = prepare_train_records(records)

    print(f"Total records : {len(records)}")
    print(f"Train records : {len(train_records)}")

    embedder = TextEmbedder(model_name=BIOMEDCLIP_MODEL_NAME)
    store = ChromaTextStore(
        persist_directory=PERSIST_DIRECTORY,
        collection_name=TEXT_COLLECTION_NAME,
    )

    if RESET_COLLECTION:
        print("Resetting text collection...")
        store.reset_collection()

    total_added = 0

    for batch in chunked(train_records, BATCH_SIZE):
        ids = [f"uid_{rec['uid']}" for rec in batch]
        documents = [rec["report"] for rec in batch]

        metadatas = []
        for rec in batch:
            metadatas.append(
                {
                    "uid": int(rec["uid"]),
                    "split": str(rec["split"]),
                    "primary_image_path": str(rec["primary_image_path"]),
                    "frontal_image_path": str(rec["frontal_image_path"]),
                    "lateral_image_path": str(rec["lateral_image_path"]),
                    "impression": str(rec["impression"]),
                    "indication": str(rec["indication"]),
                }
            )

        embeddings = embedder.encode(documents, batch_size=16)
        store.add_documents(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        total_added += len(batch)
        print(f"Indexed {total_added}/{len(train_records)}")

    print(f"Done. Text collection count: {store.count()}")


if __name__ == "__main__":
    main()