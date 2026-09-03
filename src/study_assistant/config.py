import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    model_name: str = os.getenv("STUDY_ASSISTANT_MODEL", "gpt-4.1-mini")
    embedding_model_name: str = os.getenv("STUDY_ASSISTANT_EMBEDDING_MODEL", "text-embedding-3-small")
    study_materials_dir: str = os.getenv("STUDY_MATERIALS_DIR", "data/materials")
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "data/chroma")
    course_data_path: str = os.getenv("COURSE_DATA_PATH", "data/course_data.json")


settings = Settings()
