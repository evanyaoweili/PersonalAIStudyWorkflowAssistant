from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from study_assistant.config import settings


def get_vectorstore(embeddings: Embeddings) -> Chroma:
    return Chroma(
        collection_name="study_materials",
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )


def index_documents(docs: list[Document], embeddings: Embeddings) -> Chroma:
    store = get_vectorstore(embeddings)
    if docs:
        store.add_documents(docs)
    return store
