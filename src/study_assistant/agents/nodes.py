"""The five agents from the Capstone Checkpoint 5.1 multi-agent design:
Planner, Retrieval, Reasoning, Evaluation, Response.

Sequential core: init -> planner -> retrieval -> reasoning -> evaluation.
Graph-based branch: evaluation routes to either "response" or "prepare_retry".
Targeted feedback loop: prepare_retry -> retrieval (not back to planner),
capped at one retry so an unsupported answer can't loop forever.

Each of the five agents is a plain function over PipelineState. Retrieval
doesn't reimplement anything -- it invokes the existing LangChain tools in
tools.py (still fully guardrailed/logged), just under pipeline control
instead of an LLM's own tool-calling choice.
"""

import logging
import re

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from study_assistant.agents.state import (
    Intent,
    PipelineState,
    PlannerDecision,
    ReasoningCandidate,
    ReasoningCandidates,
    ScheduleExtract,
)
from study_assistant.agents.tools import (
    ask_study_materials,
    get_course_info,
    get_person_info,
    list_roster,
    mark_assignment_completed,
    plan_study_schedule,
    whats_next,
)
from study_assistant.config import settings

logger = logging.getLogger(__name__)

APP_DESCRIPTION = (
    "a Personal AI Study Workflow Assistant. It can: answer questions about "
    "the user's ingested study materials, look up course info/assignments, "
    "look up students/teachers, say what to work on next, mark an assignment "
    "completed, and build a study schedule."
)

_CHECKPOINT_MENTION_RE = re.compile(r"(?:checkpoint|assignment)\s+\d+\.\d+", re.IGNORECASE)

_UNGROUNDED_MARKERS = (
    "I don't have any ingested study materials",
    "No record found for",
    "Nothing pending",
    "No assignment matching",
    "No course data available",
)


def _looks_ungrounded(evidence: str) -> bool:
    return any(m in evidence for m in _UNGROUNDED_MARKERS)


def _llm() -> ChatOpenAI:
    return ChatOpenAI(model=settings.model_name, api_key=settings.openai_api_key)


# ---------------------------------------------------------------------------
# init -- pull the current request out of the message history
# ---------------------------------------------------------------------------
def init_node(state: PipelineState) -> dict:
    messages = state["messages"]
    goal = messages[-1].content if messages else ""
    return {"goal": goal, "retry_count": 0}


# ---------------------------------------------------------------------------
# Planner Agent
# ---------------------------------------------------------------------------
PLANNER_PROMPT = (
    "You are the Planner for a study assistant pipeline. Classify the user's "
    "latest message into exactly one intent and extract the detail needed to "
    "act on it.\n\n"
    "Intents:\n"
    "- study_materials: a question about ingested notes/PDFs/course content\n"
    "- course_info: asking about the course name, term, or assignment list\n"
    "- roster: asking to list students and/or teachers\n"
    "- person_info: asking about a specific named student or teacher\n"
    "- next_priority: asking what to work on next / what's the priority\n"
    "- mark_completed: saying they finished a specific assignment/checkpoint\n"
    "- scheduling: asking for a study schedule/plan across tasks\n"
    "- general: anything else (greetings, out-of-scope requests, etc.)\n\n"
    "detail = the question text (study_materials), person name (person_info), "
    "or assignment name (mark_completed). For roster: exactly 'students' if "
    "only students were asked for, 'teachers' if only teachers were asked "
    "for, or '' if both/unspecified. Empty string for other intents "
    "otherwise.\n\n"
    "If the latest message refers to something earlier in the conversation "
    "(e.g. 'it', 'that one'), resolve the reference using the conversation "
    "history so `detail` is self-contained."
)


def planner_node(state: PipelineState) -> dict:
    messages = state["messages"]
    structured = _llm().with_structured_output(PlannerDecision)
    decision: PlannerDecision = structured.invoke([("system", PLANNER_PROMPT), *messages])
    logger.info("decision=planned intent=%s detail=%r", decision.intent, decision.detail)
    return {"intent": decision.intent, "detail": decision.detail}


# ---------------------------------------------------------------------------
# Retrieval Agent -- invokes the existing tools, doesn't reimplement them
# ---------------------------------------------------------------------------
def _try_build_schedule(goal: str) -> str:
    structured = _llm().with_structured_output(ScheduleExtract)
    extract: ScheduleExtract = structured.invoke(
        [
            (
                "system",
                "Extract a list of study tasks (topic, deadline as an ISO date "
                "YYYY-MM-DD, estimated_hours, priority 1-5) and hours_per_day "
                "from the user's message. If information is missing or "
                "unclear, return an empty tasks list.",
            ),
            ("human", goal),
        ]
    )
    if not extract.tasks or extract.hours_per_day is None:
        return (
            "NEEDS_MORE_INFO: To build a schedule, tell me your tasks (topic, "
            "deadline, estimated hours, optional priority) and how many hours "
            "you have available per day."
        )
    tasks_payload = [
        {
            "topic": t.topic,
            "deadline": t.deadline,
            "estimated_hours": t.estimated_hours,
            "priority": t.priority,
        }
        for t in extract.tasks
    ]
    return plan_study_schedule.invoke({"tasks": tasks_payload, "hours_per_day": extract.hours_per_day})


_RETRIEVERS = {
    "study_materials": lambda detail, goal: ask_study_materials.invoke({"question": detail or goal}),
    "course_info": lambda detail, goal: get_course_info.invoke({}),
    "roster": lambda detail, goal: list_roster.invoke(
        {"group": detail if detail in ("students", "teachers") else "all"}
    ),
    "person_info": lambda detail, goal: get_person_info.invoke({"name": detail or goal}),
    "next_priority": lambda detail, goal: whats_next.invoke({}),
    "mark_completed": lambda detail, goal: mark_assignment_completed.invoke({"assignment_name": detail or goal}),
    "scheduling": lambda detail, goal: _try_build_schedule(goal),
    "general": lambda detail, goal: "",
}


def retrieval_node(state: PipelineState) -> dict:
    intent: Intent = state["intent"]
    evidence = _RETRIEVERS[intent](state.get("detail", ""), state["goal"])
    logger.info("decision=retrieved intent=%s evidence_preview=%r", intent, evidence[:200])
    return {"evidence": evidence}


# ---------------------------------------------------------------------------
# Reasoning Agent -- lightweight Tree-of-Thought, only where branching helps
# ---------------------------------------------------------------------------
# Deterministic lookups: one candidate, no point generating alternatives for
# a factual answer that's either right or missing.
_PASSTHROUGH_INTENTS = {"course_info", "roster", "person_info", "mark_completed"}

REASONING_PROMPT = (
    f"You are the Reasoning Agent for {APP_DESCRIPTION} Given the user's goal "
    "and the evidence retrieved for it, propose up to 2 short candidate "
    "answers reflecting different reasonable framings or priorities, each "
    "with your own confidence (0.0-1.0) based on how well the evidence "
    "supports it, and a one-sentence rationale. If the evidence doesn't "
    "support a confident answer, say so in the candidate itself and give it "
    "low confidence rather than guessing. If there is no evidence because "
    "the request is general/conversational (e.g. a greeting or a question "
    "about what you can do), answer directly and helpfully with confidence 1.0."
)


def reasoning_node(state: PipelineState) -> dict:
    intent = state["intent"]
    evidence = state["evidence"]
    goal = state["goal"]

    if evidence.startswith("NEEDS_MORE_INFO:"):
        candidate = ReasoningCandidate(
            answer=evidence.removeprefix("NEEDS_MORE_INFO:").strip(),
            confidence=1.0,
            rationale="scheduling details missing from the request",
        )
        return {"candidates": [candidate], "chosen": candidate}

    if intent in _PASSTHROUGH_INTENTS:
        confidence = 0.0 if _looks_ungrounded(evidence) else 1.0
        candidate = ReasoningCandidate(answer=evidence, confidence=confidence, rationale="direct lookup")
        return {"candidates": [candidate], "chosen": candidate}

    structured = _llm().with_structured_output(ReasoningCandidates)
    result: ReasoningCandidates = structured.invoke(
        [
            ("system", REASONING_PROMPT),
            ("human", f"Goal: {goal}\n\nEvidence:\n{evidence or '(none retrieved)'}"),
        ]
    )
    candidates = result.candidates or [
        ReasoningCandidate(
            answer=evidence or "I don't have enough information to answer that.",
            confidence=0.0,
            rationale="no candidates produced",
        )
    ]
    chosen = max(candidates, key=lambda c: c.confidence)
    logger.info(
        "decision=reasoned intent=%s num_candidates=%d chosen_confidence=%.2f",
        intent, len(candidates), chosen.confidence,
    )
    return {"candidates": candidates, "chosen": chosen}


# ---------------------------------------------------------------------------
# Evaluation Agent -- grounding/completeness gate, routes back or forward
# ---------------------------------------------------------------------------
def evaluation_node(state: PipelineState) -> dict:
    if state["intent"] == "general":
        return {"grounded": True}

    chosen = state["chosen"]
    grounded = chosen.confidence >= 0.5 and not _looks_ungrounded(state["evidence"])
    logger.info(
        "decision=evaluated grounded=%s confidence=%.2f retry_count=%d intent=%s",
        grounded, chosen.confidence, state.get("retry_count", 0), state["intent"],
    )
    return {"grounded": grounded}


_RETRYABLE_INTENTS = {"study_materials", "next_priority"}


def route_after_evaluation(state: PipelineState) -> str:
    if state["grounded"]:
        return "response"
    if state.get("retry_count", 0) < 1 and state["intent"] in _RETRYABLE_INTENTS:
        return "retry"
    return "response"


def prepare_retry_node(state: PipelineState) -> dict:
    """The targeted feedback loop: send the request back to Retrieval (not
    the whole pipeline) with the checkpoint-number mention stripped, so
    rag/qa.py's checkpoint filter widens to an unfiltered/MMR-wide search."""
    detail = state.get("detail", "")
    broadened = _CHECKPOINT_MENTION_RE.sub("", detail).strip() or state["goal"]
    logger.info("decision=retry broadened_detail=%r", broadened)
    return {"detail": broadened, "retry_count": state.get("retry_count", 0) + 1}


# ---------------------------------------------------------------------------
# Response Agent
# ---------------------------------------------------------------------------
_DIRECT_PASSTHROUGH_INTENTS = _PASSTHROUGH_INTENTS | {"scheduling"}

RESPONSE_PROMPT = (
    f"You are the Response Agent for {APP_DESCRIPTION} Turn the approved "
    "result into a clear, direct answer for the learner. Do not mention "
    "internal agent names, confidence scores, or this pipeline. You have no "
    "way to submit assignments, modify course records, or send messages on "
    "the user's behalf -- if asked to do something like that, explain it's "
    "outside what you're able to do."
)

NOT_GROUNDED_MESSAGE = (
    "I don't have enough reliable information to answer that confidently. "
    "Please check with your instructor, or ingest more materials with "
    "`study-assistant ingest`."
)


def response_node(state: PipelineState) -> dict:
    intent = state["intent"]
    chosen = state["chosen"]
    grounded = state["grounded"]

    if not grounded:
        final = NOT_GROUNDED_MESSAGE
    elif intent in _DIRECT_PASSTHROUGH_INTENTS:
        # Deterministic lookups already read cleanly -- an extra LLM call to
        # "polish" them would just add latency/cost for no benefit.
        final = chosen.answer
    else:
        response = _llm().invoke(
            [
                ("system", RESPONSE_PROMPT),
                ("human", f"Goal: {state['goal']}\n\nApproved result:\n{chosen.answer}"),
            ]
        )
        final = response.content

    messages = state["messages"] + [AIMessage(content=final)]
    return {"final_answer": final, "messages": messages}
