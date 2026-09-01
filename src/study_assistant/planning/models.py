from datetime import date

from pydantic import BaseModel


class StudyTask(BaseModel):
    topic: str
    deadline: date
    estimated_hours: float
    priority: int = 1  # higher = more important


class StudyBlock(BaseModel):
    task: StudyTask
    day: date
    hours: float
