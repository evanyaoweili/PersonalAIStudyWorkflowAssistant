from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph

from study_assistant.agents.tools import ALL_TOOLS
from study_assistant.config import settings

SYSTEM_PROMPT = (
    "You are a Personal AI Study Workflow Assistant. You help the user "
    "understand their study materials and plan their study time. Use the "
    "ask_study_materials tool for questions about ingested notes/PDFs, the "
    "plan_study_schedule tool to build study schedules, and the "
    "get_course_info / get_person_info tools for questions about the course, "
    "its assignments, or its students and teachers."
)


def build_agent() -> CompiledStateGraph:
    llm = ChatOpenAI(model=settings.model_name, api_key=settings.openai_api_key)
    return create_agent(llm, tools=ALL_TOOLS, system_prompt=SYSTEM_PROMPT)
