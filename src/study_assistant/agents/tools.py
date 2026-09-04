import logging
from datetime import date

from langchain_core.tools import tool

from study_assistant.course_data import get_course_info as _get_course_info
from study_assistant.course_data import get_person_info as _get_person_info
from study_assistant.course_data import list_people as _list_people
from study_assistant.ingestion.embeddings import get_embeddings
from study_assistant.ingestion.vectorstore import get_vectorstore
from study_assistant.planning.models import StudyTask
from study_assistant.planning.scheduler import build_schedule
from study_assistant.rag.qa import answer_question

logger = logging.getLogger(__name__)


@tool
def ask_study_materials(question: str) -> str:
    """Answer a question using the user's ingested study materials (notes, PDFs, etc.)."""
    store = get_vectorstore(get_embeddings())
    return answer_question(question, store)


@tool
def get_course_info() -> str:
    """Look up the current course's name, term, and list of assignments."""
    info = _get_course_info()
    if not info.get("course"):
        return "No course data available."
    lines = [f"Course: {info['course']}", f"Term: {info['term']}", "Assignments:"]
    lines += [f"- {a}" for a in info["assignments"]]
    return "\n".join(lines)


@tool
def list_roster() -> str:
    """List the names of all known students and teachers in the course."""
    roster = _list_people()
    lines = ["Students:"]
    lines += [f"- {name}" for name in roster["students"]]
    lines.append("Teachers:")
    lines += [f"- {name}" for name in roster["teachers"]]
    return "\n".join(lines)


@tool
def get_person_info(name: str) -> str:
    """Look up a student's or teacher's introduction/bio by name from the course roster."""
    person = _get_person_info(name)
    if person is None:
        logger.info("escalation=person_not_found requested_name=%r", name)
        roster = _list_people()
        return (
            f"No record found for '{name}'. "
            f"Known students: {roster['students']}. Known teachers: {roster['teachers']}."
        )
    return f"{person['name']} ({person['role']}): {person.get('introduction', 'no bio on file')}"


@tool
def plan_study_schedule(tasks: list[StudyTask], hours_per_day: float) -> str:
    """Build a day-by-day study schedule for a list of tasks (topic, deadline,
    estimated_hours, priority), given how many hours are available per day."""
    blocks = build_schedule(tasks, start_day=date.today(), hours_per_day=hours_per_day)
    if not blocks:
        return "No schedule could be built - check task deadlines and hours."
    lines = [f"{b.day.isoformat()}: {b.task.topic} - {b.hours:.1f}h" for b in blocks]
    return "\n".join(lines)


ALL_TOOLS = [ask_study_materials, plan_study_schedule, get_course_info, get_person_info, list_roster]
