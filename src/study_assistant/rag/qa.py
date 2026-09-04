import logging

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from study_assistant.config import settings

logger = logging.getLogger(__name__)

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
    when there's no evidence, and flag the answer when evidence is weak."""
    results = store.similarity_search_with_score(question, k=k)

    if not results:
        logger.info("escalation=no_evidence question=%r", question)
        return NO_EVIDENCE_MESSAGE

    best_distance = min(score for _, score in results)
    confidence = "low" if best_distance > settings.rag_max_distance else "normal"
    logger.info(
        "decision=answered confidence=%s best_distance=%.4f threshold=%.4f question=%r",
        confidence, best_distance, settings.rag_max_distance, question,
    )

    context = "\n\n".join(doc.page_content for doc, _ in results)
    llm = ChatOpenAI(model=settings.model_name, api_key=settings.openai_api_key)
    chain = QA_PROMPT | llm
    response = chain.invoke({"context": context, "question": question, "confidence": confidence})
    return response.content
