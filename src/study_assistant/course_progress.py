import json
from pathlib import Path

from study_assistant.config import settings


def _load() -> dict:
    path = Path(settings.course_progress_path)
    if not path.exists():
        return {"completed_assignments": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _save(data: dict) -> None:
    path = Path(settings.course_progress_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def mark_completed(assignment_name: str) -> None:
    data = _load()
    if assignment_name not in data["completed_assignments"]:
        data["completed_assignments"].append(assignment_name)
        _save(data)


def get_completed() -> list[str]:
    return _load()["completed_assignments"]
