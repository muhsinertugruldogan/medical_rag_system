from __future__ import annotations

from pathlib import Path
from typing import List

import torch
from PIL import Image
from open_clip import create_model_from_pretrained


class BiomedClipImageEmbedder:
    def __init__(
        self,
        model_name: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model, self.preprocess = create_model_from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def encode_image(self, image_path: str) -> List[float]:
        image_file = Path(image_path)
        if not image_file.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_file).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            image_features = torch.nn.functional.normalize(image_features, dim=-1)

        return image_features[0].detach().cpu().tolist()