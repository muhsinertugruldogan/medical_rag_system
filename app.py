from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from retrieval.config import (
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

# Retrieval / rerank params
TEXT_TOP_K = 10
IMAGE_TOP_K = 8
RERANK_TOP_K = 6

app = FastAPI(
    title="Medical RAG API",
    version="0.1.0",
    description="Multi-modal Medical RAG system for chest X-ray question answering.",
)


class QueryRequest(BaseModel):
    question: str = Field(..., description="Medical question to answer")
    image_path: Optional[str] = Field(
        default=None,
        description="Optional local image path for multimodal query",
    )


class SourceItem(BaseModel):
    uid: Optional[str] = None
    impression: Optional[str] = None
    from_text: Optional[bool] = None
    from_image: Optional[bool] = None
    text_rank: Optional[int] = None
    image_rank: Optional[int] = None
    rerank_score: Optional[float] = None


class LatencyInfo(BaseModel):
    retrieval_ms: float
    generation_ms: float
    total_ms: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    latency_ms: LatencyInfo


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


# Load once at startup
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


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Medical RAG API is running."}


@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest) -> QueryResponse:
    question = payload.question.strip()
    image_path = payload.image_path.strip() if payload.image_path else None

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    t0 = time.perf_counter()

    try:
        text_query_embedding = text_embedder.encode_query(question)

        text_results = text_store.query(
            query_embedding=text_query_embedding,
            n_results=TEXT_TOP_K,
        )

        if image_path:
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

        reranked = reranker.rerank(
            question=question,
            candidates=pooled,
            top_k=RERANK_TOP_K,
        )

        t1 = time.perf_counter()

        answer = generator.generate_answer(
            question=question,
            retrieved_contexts=reranked,
            image_findings=None,
        )

        t2 = time.perf_counter()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")

    return QueryResponse(
        answer=answer,
        sources=[
            SourceItem(
                uid=c.get("uid"),
                impression=c.get("impression"),
                from_text=c.get("from_text"),
                from_image=c.get("from_image"),
                text_rank=c.get("text_rank"),
                image_rank=c.get("image_rank"),
                rerank_score=c.get("rerank_score"),
            )
            for c in reranked
        ],
        latency_ms=LatencyInfo(
            retrieval_ms=round((t1 - t0) * 1000, 2),
            generation_ms=round((t2 - t1) * 1000, 2),
            total_ms=round((t2 - t0) * 1000, 2),
        ),
    )