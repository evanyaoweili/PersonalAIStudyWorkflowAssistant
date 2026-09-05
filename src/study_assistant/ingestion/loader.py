import re
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

_CHECKPOINT_RE = re.compile(r"(?:checkpoint|assignment)\s+(\d+\.\d+)", re.IGNORECASE)


def _tag_metadata(doc: Document, path: Path) -> None:
    """Attach lightweight source metadata (checkpoint number, document type)
    so retrieval can disambiguate e.g. a '3.1' question from '2.1' content."""
    name = path.stem
    match = _CHECKPOINT_RE.search(name)
    if match:
        doc.metadata["checkpoint"] = match.group(1)
    if "assignment brief" in name.lower():
        doc.metadata["doc_type"] = "assignment_brief"
    elif match:
        doc.metadata["doc_type"] = "design_submission"
    else:
        doc.metadata["doc_type"] = "reference"


def load_documents(materials_dir: str) -> list[Document]:
    docs: list[Document] = []
    for path in Path(materials_dir).rglob("*"):
        if not path.is_file():
            continue
        loader_cls = LOADERS.get(path.suffix.lower())
        if loader_cls is None:
            continue
        loaded = loader_cls(str(path)).load()
        for doc in loaded:
            _tag_metadata(doc, path)
        docs.extend(loaded)
    return docs


def split_documents(docs: list[Document], chunk_size: int = 1000, chunk_overlap: int = 150) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)
