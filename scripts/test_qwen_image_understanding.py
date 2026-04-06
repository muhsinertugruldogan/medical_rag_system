from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from generation.qwen_image_understanding import QwenImageUnderstanding

BASE_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = BASE_DIR / "data" / "indiana_manifest.jsonl"


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def pick_test_record(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    for rec in records:
        if rec.get("split") == "test" and rec.get("primary_image_path"):
            return rec
    raise ValueError("No suitable test record found.")


def main() -> None:
    records = load_manifest(MANIFEST_PATH)
    rec = pick_test_record(records)

    model = QwenImageUnderstanding(
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        device_map="auto",
        torch_dtype="auto",
        max_new_tokens=256,
    )

    result = model.generate_findings(
        image_path=rec["primary_image_path"],
        user_question="Provide a concise radiology-style report for this frontal chest X-ray.",
    )

    print("=" * 100)
    print(f"UID: {rec['uid']}")
    print(f"IMAGE: {rec['primary_image_path']}")
    print("-" * 100)
    print(result)
    print("=" * 100)


if __name__ == "__main__":
    main()