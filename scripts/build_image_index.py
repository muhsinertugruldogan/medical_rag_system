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
    IMAGE_COLLECTION_NAME,
    IMAGE_EMBED_MODEL_NAME,
)
from retrieval.image_embedder import BiomedClipImageEmbedder
from retrieval.vectordb import ChromaTextStore

RESET_COLLECTION = True


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    records = []
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
        if not str(rec.get("primary_image_path", "")).strip():
            continue
        out.append(rec)
    return out


def main() -> None:
    print(f"Loading manifest from: {MANIFEST_PATH}")
    print(f"Image collection: {IMAGE_COLLECTION_NAME}")
    print(f"Image model: {IMAGE_EMBED_MODEL_NAME}")

    records = load_manifest(MANIFEST_PATH)
    train_records = prepare_train_records(records)

    print(f"Total records : {len(records)}")
    print(f"Train records : {len(train_records)}")

    embedder = BiomedClipImageEmbedder(model_name=IMAGE_EMBED_MODEL_NAME)
    store = ChromaTextStore(
        persist_directory=PERSIST_DIRECTORY,
        collection_name=IMAGE_COLLECTION_NAME,
    )

    if RESET_COLLECTION:
        print("Resetting image collection...")
        store.reset_collection()

    total_added = 0

    for rec in train_records:
        try:
            emb = embedder.encode_image(rec["primary_image_path"])
        except Exception as e:
            print(f"Skipping uid={rec['uid']} due to image embedding error: {e}")
            continue

        store.add_documents(
            ids=[f"img_uid_{rec['uid']}"],
            documents=[rec["report"]],
            embeddings=[emb],
            metadatas=[{
                "uid": int(rec["uid"]),
                "split": str(rec["split"]),
                "primary_image_path": str(rec["primary_image_path"]),
                "frontal_image_path": str(rec["frontal_image_path"]),
                "lateral_image_path": str(rec["lateral_image_path"]),
                "impression": str(rec["impression"]),
                "indication": str(rec["indication"]),
            }],
        )

        total_added += 1
        if total_added % 100 == 0:
            print(f"Indexed {total_added}/{len(train_records)}")

    print(f"Done. Image collection count: {store.count()}")


if __name__ == "__main__":
    main()