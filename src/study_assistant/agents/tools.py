from datetime import date

from langchain_core.tools import tool

from study_assistant.ingestion.embeddings import get_embeddings
from study_assistant.ingestion.vectorstore import get_vectorstore
from study_assistant.planning.models import StudyTask
from study_assistant.planning.scheduler import build_schedule
from study_assistant.rag.qa import answer_question


@tool
def ask_study_materials(question: str) -> str:
    """Answer a question using the user's ingested study materials (notes, PDFs, etc.)."""
    store = get_vectorstore(get_embeddings())
    return answer_question(question, store)


@tool
def plan_study_schedule(tasks: list[StudyTask], hours_per_day: float) -> str:
    """Build a day-by-day study schedule for a list of tasks (topic, deadline,
    estimated_hours, priority), given how many hours are available per day."""
    blocks = build_schedule(tasks, start_day=date.today(), hours_per_day=hours_per_day)
    if not blocks:
        return "No schedule could be built - check task deadlines and hours."
    lines = [f"{b.day.isoformat()}: {b.task.topic} - {b.hours:.1f}h" for b in blocks]
    return "\n".join(lines)


ALL_TOOLS = [ask_study_materials, plan_study_schedule]
