from langchain_huggingface import HuggingFaceEmbeddings

# Local, free embedding model - avoids requiring a separate embeddings API key
# since Anthropic does not offer one.
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=_MODEL_NAME)
