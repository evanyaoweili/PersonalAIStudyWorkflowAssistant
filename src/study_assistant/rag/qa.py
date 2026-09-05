import logging
import re

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from study_assistant.config import settings

logger = logging.getLogger(__name__)

_CHECKPOINT_RE = re.compile(r"(?:checkpoint|assignment)\s+(\d+\.\d+)", re.IGNORECASE)

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a study assistant. Answer the question using only the "
            "provided context from the user's study materials. If the context "
            "doesn't contain enough information to answer confidently, say so "
            "explicitly and suggest the user check with their instructor or "
            "ingest more materials — never guess or invent an answer.\n\n"
            "Retrieval confidence for this query is '{confidence}'. If it is "
            "'low', make clear your answer may be incomplete and should be "
            "verified before being relied on.\n\nContext:\n{context}",
        ),
        ("human", "{question}"),
    ]
)

NO_EVIDENCE_MESSAGE = (
    "I don't have any ingested study materials to search yet, so I can't answer "
    "this from your own notes/PDFs. Run `study-assistant ingest` after adding "
    "files to data/materials/, or ask your instructor directly."
)


def answer_question(question: str, store: Chroma, k: int = 4) -> str:
    """Ground the answer in retrieved evidence; escalate instead of guessing
    when there's no evidence, and flag the answer when evidence is weak.

    If the question names a specific checkpoint/assignment number, retrieval
    is first restricted to chunks tagged with that number (see
    ingestion/loader.py's metadata tagging) so a "3.1" question can't
    accidentally be answered from "2.1" content just because the two are
    semantically similar (both discuss the same capstone project).
    """
    checkpoint_filter = None
    match = _CHECKPOINT_RE.search(question)
    if match:
        candidate_filter = {"checkpoint": match.group(1)}
        if store.similarity_search(question, k=1, filter=candidate_filter):
            checkpoint_filter = candidate_filter

    scored = store.similarity_search_with_score(question, k=k, filter=checkpoint_filter)
    if not scored:
        logger.info("escalation=no_evidence question=%r", question)
        return NO_EVIDENCE_MESSAGE

    best_distance = min(score for _, score in scored)
    confidence = "low" if best_distance > settings.rag_max_distance else "normal"
    logger.info(
        "decision=answered confidence=%s best_distance=%.4f threshold=%.4f "
        "checkpoint_filter=%s question=%r",
        confidence, best_distance, settings.rag_max_distance, checkpoint_filter, question,
    )

    # MMR (not plain top-k) for the context passages themselves, so several
    # near-duplicate chunks don't crowd out distinct information.
    fetch_k = max(4 * k, 20)
    docs = store.max_marginal_relevance_search(question, k=k, fetch_k=fetch_k, filter=checkpoint_filter)
    context = "\n\n".join(doc.page_content for doc in docs)

    llm = ChatOpenAI(model=settings.model_name, api_key=settings.openai_api_key)
    chain = QA_PROMPT | llm
    response = chain.invoke({"context": context, "question": question, "confidence": confidence})
    return response.content
