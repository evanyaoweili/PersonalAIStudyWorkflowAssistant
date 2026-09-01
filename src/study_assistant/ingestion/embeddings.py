from langchain_openai import OpenAIEmbeddings

from study_assistant.config import settings


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=settings.embedding_model_name, api_key=settings.openai_api_key)
