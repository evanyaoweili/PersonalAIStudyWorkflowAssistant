import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    model_name: str = os.getenv("STUDY_ASSISTANT_MODEL", "claude-sonnet-5")
    study_materials_dir: str = os.getenv("STUDY_MATERIALS_DIR", "data/materials")
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "data/chroma")


settings = Settings()
