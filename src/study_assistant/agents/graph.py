from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from study_assistant.agents.tools import ALL_TOOLS
from study_assistant.config import settings

SYSTEM_PROMPT = (
    "You are a Personal AI Study Workflow Assistant. You help the user "
    "understand their study materials and plan their study time. Use the "
    "ask_study_materials tool for questions about ingested notes/PDFs, and "
    "the plan_study_schedule tool to build study schedules."
)


def build_agent() -> CompiledStateGraph:
    llm = ChatOpenAI(model=settings.model_name, api_key=settings.openai_api_key)
    return create_react_agent(llm, tools=ALL_TOOLS, prompt=SYSTEM_PROMPT)
