from typing import Literal, TypedDict

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

Intent = Literal[
    "study_materials",
    "course_info",
    "roster",
    "person_info",
    "next_priority",
    "mark_completed",
    "scheduling",
    "general",
]


class PlannerDecision(BaseModel):
    """Structured output of the Planner Agent: what the user needs and the
    detail (question text / person name / assignment name) to act on it."""

    intent: Intent
    detail: str = Field(
        description="The specific question, person name, or assignment name to "
        "look up, extracted from the user's message. Empty string if not applicable."
    )


class ReasoningCandidate(BaseModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class ReasoningCandidates(BaseModel):
    candidates: list[ReasoningCandidate]


class ScheduleTaskExtract(BaseModel):
    topic: str
    deadline: str = Field(description="ISO date, YYYY-MM-DD")
    estimated_hours: float
    priority: int = 1


class ScheduleExtract(BaseModel):
    tasks: list[ScheduleTaskExtract]
    hours_per_day: float | None = None


class PipelineState(TypedDict, total=False):
    messages: list[BaseMessage]
    goal: str
    intent: Intent
    detail: str
    evidence: str
    candidates: list[ReasoningCandidate]
    chosen: ReasoningCandidate
    grounded: bool
    retry_count: int
    final_answer: str
