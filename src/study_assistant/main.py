import argparse
import logging

from langchain_core.globals import set_debug
from langchain_core.messages import HumanMessage

from study_assistant.agents.graph import build_agent
from study_assistant.config import settings
from study_assistant.ingestion.embeddings import get_embeddings
from study_assistant.ingestion.loader import load_documents, split_documents
from study_assistant.ingestion.vectorstore import index_documents
from study_assistant.observability import configure_logging

logger = logging.getLogger(__name__)


def cmd_ingest(_args: argparse.Namespace) -> None:
    docs = load_documents(settings.study_materials_dir)
    chunks = split_documents(docs)
    index_documents(chunks, get_embeddings())
    print(f"Ingested {len(docs)} document(s) into {len(chunks)} chunk(s).")


def cmd_chat(_args: argparse.Namespace) -> None:
    agent = build_agent()
    print("Study assistant ready. Type 'exit' to quit.")
    while True:
        question = input("> ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        try:
            result = agent.invoke({"messages": [HumanMessage(content=question)]})
        except Exception:
            logger.exception("decision=error question=%r", question)
            print("Something went wrong answering that. Please try again.")
            continue
        print(result["messages"][-1].content)


def main() -> None:
    parser = argparse.ArgumentParser(prog="study-assistant")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging (Python logging + LangChain chain/tool-call tracing).",
    )
    subparsers = parser.add_subparsers(required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest study materials into the vector store")
    ingest_parser.set_defaults(func=cmd_ingest)

    chat_parser = subparsers.add_parser("chat", help="Chat with the study assistant agent")
    chat_parser.set_defaults(func=cmd_chat)

    args = parser.parse_args()

    configure_logging()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
        set_debug(True)

    args.func(args)


if __name__ == "__main__":
    main()
