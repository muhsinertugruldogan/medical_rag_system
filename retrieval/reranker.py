from __future__ import annotations

import re
from typing import Dict, List, Optional

from sentence_transformers import CrossEncoder


class MedicalCrossEncoderReranker:
    """
    Medical cross-encoder reranker for chest X-ray report candidates.

    Principles:
    - Ranking depends ONLY on semantic relevance between question and report.
    - Retrieval metadata is preserved but does NOT affect ranking.
    - Duplicate reports are merged to avoid metadata loss.
    - Batch inference is used for efficiency.
    """

    def __init__(
        self,
        model_name: str = "ncbi/MedCPT-Cross-Encoder",
        max_length: int = 512,
        batch_size: int = 16,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size

        self.model = CrossEncoder(
            model_name,
            max_length=max_length,
            trust_remote_code=False,
        )

    @staticmethod
    def _normalize_report_text(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _safe_min(a: Optional[int], b: Optional[int]) -> Optional[int]:
        if a is None:
            return b
        if b is None:
            return a
        return min(a, b)

    def _merge_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """
        Merge duplicates by normalized report text.
        Metadata is preserved, but ranking will not use it.
        """
        merged: Dict[str, Dict] = {}

        for cand in candidates:
            report = cand.get("report", "")
            if not isinstance(report, str) or not report.strip():
                continue

            key = self._normalize_report_text(report)

            if key not in merged:
                merged[key] = {
                    **cand,
                    "from_text": bool(cand.get("from_text", False)),
                    "from_image": bool(cand.get("from_image", False)),
                    "text_rank": cand.get("text_rank"),
                    "image_rank": cand.get("image_rank"),
                    "retrieval_count": 1,
                }
                continue

            existing = merged[key]

            existing["from_text"] = existing["from_text"] or bool(cand.get("from_text", False))
            existing["from_image"] = existing["from_image"] or bool(cand.get("from_image", False))
            existing["text_rank"] = self._safe_min(existing.get("text_rank"), cand.get("text_rank"))
            existing["image_rank"] = self._safe_min(existing.get("image_rank"), cand.get("image_rank"))
            existing["retrieval_count"] = int(existing.get("retrieval_count", 1)) + 1

            for field in ("study_id", "doc_id", "uid", "source", "image_path", "impression"):
                if existing.get(field) is None and cand.get(field) is not None:
                    existing[field] = cand[field]

        return list(merged.values())

    @staticmethod
    def _build_pair(question: str, candidate: Dict) -> List[str]:
        """
        Pure semantic input for cross-encoder:
        [question, report]
        No metadata injection.
        """
        return [question.strip(), candidate["report"].strip()]

    def rerank(
        self,
        question: str,
        candidates: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Returns top_k reranked candidates.

        Added fields:
        - rerank_score_raw: raw model score
        - rerank_score: same as raw score for downstream compatibility

        Important:
        - Ranking is based ONLY on semantic score.
        - Metadata does not affect score or sort order.
        """
        if not isinstance(question, str) or not question.strip():
            return candidates[:top_k]

        merged_candidates = self._merge_candidates(candidates)
        if not merged_candidates:
            return []

        pairs = [self._build_pair(question, cand) for cand in merged_candidates]

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        scored_candidates: List[Dict] = []
        for cand, score in zip(merged_candidates, scores):
            score = float(score)

            enriched = dict(cand)
            enriched["rerank_score_raw"] = score
            enriched["rerank_score"] = score
            scored_candidates.append(enriched)

        scored_candidates.sort(
            key=lambda x: x["rerank_score_raw"],
            reverse=True,
        )

        return scored_candidates[:top_k]