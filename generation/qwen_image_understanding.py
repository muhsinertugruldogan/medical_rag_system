from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


class QwenImageUnderstanding:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device_map: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 256,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

        # base_prompt = (
        #     "You are reading a chest X-ray. "
        #     "Describe only the medically relevant visible findings in concise radiology style. "
        #     "Do not give treatment. "
        #     "If the image quality or view limits certainty, say so. "
        #     "Output 3 sections:\n"
        #     "1. Findings\n"
        #     "2. Impression\n"
        #     "3. Key terms"
        # )
        
    def _build_prompt(self, user_question: Optional[str] = None) -> str:
        base_prompt = (
            "You are a radiology assistant. "
            "Read this single frontal chest X-ray and produce a short radiology-style report. "
            "Do not explain what a chest X-ray is. "
            "Do not describe the imaging modality. "
            "Do not give advice, recommendations, or treatment. "
            "Do not mention consultation with a doctor. "
            "Only report visible findings. "
            "If no acute abnormality is visible, say so.\n\n"
            "Return exactly in this format:\n"
            "Findings: <one or two short sentences>\n"
            "Impression: <one short sentence>\n"
            "Key terms: <comma-separated keywords>"
        )

        if user_question and user_question.strip():
            return f"{base_prompt}\n\nTask: {user_question.strip()}"
        return base_prompt

    def generate_findings(
        self,
        image_path: str,
        user_question: Optional[str] = None,
    ) -> str:
        image_file = Path(image_path)
        if not image_file.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_file).convert("RGB")
        prompt = self._build_prompt(user_question=user_question)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )

        inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

        return output_text.strip()