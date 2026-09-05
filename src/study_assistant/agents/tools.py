import logging
from datetime import date, datetime

from langchain_core.tools import tool

from study_assistant.course_data import get_course_info as _get_course_info
from study_assistant.course_data import get_person_info as _get_person_info
from study_assistant.course_data import list_people as _list_people
from study_assistant.course_progress import get_completed as _get_completed
from study_assistant.course_progress import mark_completed as _mark_completed
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


def _format_assignment(a) -> str:
    if not isinstance(a, dict):
        return f"- {a}"
    name = a.get("assignment", "Untitled assignment")
    details = []
    if a.get("due_date"):
        details.append(f"due {a['due_date']}")
    if a.get("points") is not None:
        details.append(f"{a['points']} pt" + ("s" if a["points"] != 1 else ""))
    suffix = f" ({', '.join(details)})" if details else ""
    return f"- {name}{suffix}"


@tool
def get_course_info() -> str:
    """Look up the current course's name, term, and list of assignments."""
    info = _get_course_info()
    if not info.get("course"):
        return "No course data available."
    lines = [f"Course: {info['course']}", f"Term: {info['term']}", "Assignments:"]
    lines += [_format_assignment(a) for a in info["assignments"]]
    return "\n".join(lines)


@tool
def list_roster(group: str = "all") -> str:
    """List known students and/or teachers. group is 'students', 'teachers',
    or 'all' (default) to list both."""
    roster = _list_people()
    lines = []
    if group in ("all", "students"):
        lines.append("Students:")
        lines += [f"- {name}" for name in roster["students"]]
    if group in ("all", "teachers"):
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
def whats_next() -> str:
    """Recommend the highest-priority thing to work on next for the course:
    the nearest-due assignment/checkpoint that hasn't been marked completed."""
    info = _get_course_info()
    assignments = info.get("assignments", [])
    completed = set(_get_completed())

    pending = []
    for a in assignments:
        name = a.get("assignment") if isinstance(a, dict) else a
        if not name or name in completed:
            continue
        due_str = a.get("due_date") if isinstance(a, dict) else None
        try:
            due_date = datetime.strptime(due_str, "%d-%b-%y").date() if due_str else None
        except ValueError:
            due_date = None
        pending.append((due_date, name, a))

    if not pending:
        return "Nothing pending — every known assignment is marked completed (or there's no assignment data)."

    pending.sort(key=lambda p: (p[0] is None, p[0]))
    due_date, name, a = pending[0]
    lines = [f"Next up: {name}"]
    if isinstance(a, dict):
        if a.get("due_date"):
            lines.append(f"Due: {a['due_date']}")
        if a.get("points") is not None:
            lines.append(f"Points: {a['points']}")
    if len(pending) > 1:
        lines.append(f"({len(pending) - 1} more pending assignment(s) after this one.)")
    return "\n".join(lines)


@tool
def mark_assignment_completed(assignment_name: str) -> str:
    """Mark a course assignment/checkpoint as completed (matched by partial,
    case-insensitive name) so whats_next stops recommending it."""
    info = _get_course_info()
    names = [a.get("assignment", "") if isinstance(a, dict) else a for a in info.get("assignments", [])]
    match = next((n for n in names if assignment_name.lower() in n.lower()), None)
    if match is None:
        logger.info("escalation=assignment_not_found requested=%r", assignment_name)
        return f"No assignment matching '{assignment_name}' found. Known assignments: {names}"
    _mark_completed(match)
    logger.info("decision=assignment_marked_completed assignment=%r", match)
    return f"Marked '{match}' as completed."


@tool
def plan_study_schedule(tasks: list[StudyTask], hours_per_day: float) -> str:
    """Build a day-by-day study schedule for a list of tasks (topic, deadline,
    estimated_hours, priority), given how many hours are available per day."""
    blocks = build_schedule(tasks, start_day=date.today(), hours_per_day=hours_per_day)
    if not blocks:
        return "No schedule could be built - check task deadlines and hours."
    lines = [f"{b.day.isoformat()}: {b.task.topic} - {b.hours:.1f}h" for b in blocks]
    return "\n".join(lines)
