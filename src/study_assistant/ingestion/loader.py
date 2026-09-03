from functools import partial
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": partial(TextLoader, encoding="utf-8", autodetect_encoding=True),
    ".md": partial(TextLoader, encoding="utf-8", autodetect_encoding=True),
}


def load_documents(materials_dir: str) -> list[Document]:
    docs: list[Document] = []
    for path in Path(materials_dir).rglob("*"):
        if not path.is_file():
            continue
        loader_cls = LOADERS.get(path.suffix.lower())
        if loader_cls is None:
            continue
        docs.extend(loader_cls(str(path)).load())
    return docs


def split_documents(docs: list[Document], chunk_size: int = 1000, chunk_overlap: int = 150) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)
