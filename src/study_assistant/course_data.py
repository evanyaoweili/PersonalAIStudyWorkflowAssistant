import json
from functools import lru_cache
from pathlib import Path

from study_assistant.config import settings


@lru_cache(maxsize=1)
def _load() -> dict:
    path = Path(settings.course_data_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def get_course_info() -> dict:
    data = _load()
    return {
        "course": data.get("course"),
        "term": data.get("term"),
        "assignments": data.get("assignments", []),
    }


def list_people() -> dict:
    data = _load()
    return {
        "students": list(data.get("students", {}).keys()),
        "teachers": list(data.get("teachers", {}).keys()),
    }


def get_person_info(name: str) -> dict | None:
    data = _load()
    for role in ("students", "teachers"):
        for person_name, info in data.get(role, {}).items():
            if person_name.lower() == name.lower():
                return {"name": person_name, "role": role[:-1], **info}
    return None
