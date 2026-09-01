from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from study_assistant.config import settings

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a study assistant. Answer the question using only the "
            "provided context from the user's study materials. If the answer "
            "isn't in the context, say you don't know.\n\nContext:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def answer_question(question: str, store: Chroma, k: int = 4) -> str:
    results = store.similarity_search(question, k=k)
    context = "\n\n".join(doc.page_content for doc in results)

    llm = ChatOpenAI(model=settings.model_name, api_key=settings.openai_api_key)
    chain = QA_PROMPT | llm
    response = chain.invoke({"context": context, "question": question})
    return response.content
