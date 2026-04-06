from __future__ import annotations

from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenAnswerGenerator:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device_map: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 100,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

    def _build_messages(
        self,
        question: str,
        retrieved_contexts: List[Dict[str, str]],
        image_findings: str | None = None,
    ) -> List[Dict[str, str]]:
        context_blocks = []
        for i, ctx in enumerate(retrieved_contexts, start=1):
            uid = ctx.get("uid", "")
            impression = ctx.get("impression", "")
            report = ctx.get("report", "")
            context_blocks.append(
                f"[Source {i} | uid={uid}]\n"
                f"Impression: {impression}\n"
                f"Report: {report}"
            )

        joined_context = "\n\n".join(context_blocks) if context_blocks else "[No retrieved reports]"

        user_sections = [f"Question:\n{question.strip()}"]

        if image_findings and image_findings.strip():
            user_sections.append(f"Optional image-derived cues:\n{image_findings.strip()}")

        user_sections.append(f"Retrieved reports:\n{joined_context}")

        user_sections.append(
            "Return EXACTLY these 3 lines and nothing else:\n"
            "Answer: <1-2 sentences. If treatment/management/follow-up is asked but not supported, say inside this line that treatment information is not available from the retrieved reports.>\n"
            "Evidence summary: <1 sentence based only on retrieved reports>\n"
            "Confidence: <low|medium|high>"
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a grounded medical RAG assistant for chest X-ray questions. "
                    "Answer using ONLY the retrieved reports as primary evidence. "
                    "If optional image-derived cues are provided, treat them as secondary support only. "
                    "If retrieved reports and image-derived cues conflict, prioritize the retrieved reports. "
                    "Do NOT add findings, diagnoses, treatments, follow-up plans, or recommendations unless they are explicitly supported by the retrieved reports. "
                    "If the user's question has multiple parts, answer the parts that are supported by the retrieved reports and explicitly say when a requested part is not available from the retrieved reports. "
                    "If the reports support only imaging findings but not treatment information, say so clearly instead of guessing. "
                    "Be concise, factual, and medically cautious. "
                    "Do NOT add disclaimers or extra text outside the required 3 lines."
                ),
            },
            {
                "role": "user",
                "content": "\n\n".join(user_sections),
            },
        ]
        return messages

    def _build_model_inputs(
        self,
        question: str,
        retrieved_contexts: List[Dict[str, str]],
        image_findings: str | None = None,
    ):
        messages = self._build_messages(
            question=question,
            retrieved_contexts=retrieved_contexts,
            image_findings=image_findings,
        )

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        return inputs

    @staticmethod
    def _postprocess_output(text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        answer_line = next((l for l in lines if l.lower().startswith("answer:")), None)
        evidence_line = next((l for l in lines if l.lower().startswith("evidence summary:")), None)
        confidence_line = next((l for l in lines if l.lower().startswith("confidence:")), None)

        if answer_line is None or answer_line.lower() == "answer:":
            answer_line = "Answer: Unable to answer cleanly from the retrieved reports."

        if evidence_line is None or evidence_line.lower() == "evidence summary:":
            evidence_line = "Evidence summary: The retrieved reports did not provide a clean extractable summary."

        if confidence_line is None or confidence_line.lower() == "confidence:":
            confidence_line = "Confidence: medium"

        if not answer_line.lower().startswith("answer:"):
            answer_line = f"Answer: {answer_line}"

        if not evidence_line.lower().startswith("evidence summary:"):
            evidence_line = f"Evidence summary: {evidence_line}"

        if not confidence_line.lower().startswith("confidence:"):
            confidence_line = f"Confidence: {confidence_line}"

        return "\n".join([answer_line, evidence_line, confidence_line])

    def generate_answer(
        self,
        question: str,
        retrieved_contexts: List[Dict[str, str]],
        image_findings: str | None = None,
    ) -> str:
        inputs = self._build_model_inputs(
            question=question,
            retrieved_contexts=retrieved_contexts,
            image_findings=image_findings,
        )

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return self._postprocess_output(text)