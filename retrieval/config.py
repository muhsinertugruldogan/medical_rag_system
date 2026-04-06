from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MANIFEST_PATH = BASE_DIR / "data" / "indiana_manifest.jsonl"
PERSIST_DIRECTORY = str(BASE_DIR / "chroma_db")

BIOMEDCLIP_MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

TEXT_COLLECTION_NAME = "iu_xray_reports_biomedclip"
IMAGE_COLLECTION_NAME = "iu_xray_images_biomedclip"