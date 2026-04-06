from __future__ import annotations

from typing import List

import torch
from open_clip import create_model_from_pretrained, get_tokenizer


class TextEmbedder:
    def __init__(
        self,
        model_name: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model, _ = create_model_from_pretrained(model_name)
        self.tokenizer = get_tokenizer(model_name)

        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            tokens = self.tokenizer(batch, context_length=256).to(self.device)

            with torch.no_grad():
                text_features = self.model.encode_text(tokens)
                text_features = torch.nn.functional.normalize(text_features, dim=-1)

            all_embeddings.extend(text_features.detach().cpu().tolist())

        return all_embeddings

    def encode_query(self, text: str) -> List[float]:
        return self.encode([text], batch_size=1)[0]