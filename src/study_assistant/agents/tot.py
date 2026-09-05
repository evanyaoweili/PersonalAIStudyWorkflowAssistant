"""Tree-of-Thought reasoning for prioritization decisions, per the Capstone
Checkpoint 4.1 (updated) design: a thought = one proposed next assignment to
work on; a branch = one strategy for proposing it; depth 1 generates and
scores candidates, depth 2 elaborates the surviving beam into an actionable
recommendation.

Scoped deliberately narrow, per the doc's own guidance: "A simple question
such as finding a due date does not require ToT... deciding what the learner
should work on next may involve several possible reasoning paths." This
module is only used for the next_priority intent; direct factual lookups
bypass it entirely (see nodes.py's _PASSTHROUGH_INTENTS).

Branching factor and beam width are small on purpose (the doc: "probably
around three candidate choices per step... a depth limit of approximately
three"), and score using real data already in this project rather than an
LLM's own self-rated confidence: urgency (due date), prerequisite readiness
(is the preceding checkpoint marked complete), and evidence quality (how
well the checkpoint's requirements are actually grounded in ingested
materials) -- the same three tensions the doc's worked example calls out
(nearest deadline vs. preparation vs. unfinished prerequisite material).
"""

import logging
import re
from datetime import date, datetime

from study_assistant.agents.state import ReasoningCandidate
from study_assistant.course_data import get_course_info
from study_assistant.course_progress import get_completed
from study_assistant.ingestion.embeddings import get_embeddings
from study_assistant.ingestion.vectorstore import get_vectorstore

logger = logging.getLogger(__name__)

BRANCH_WIDTH = 3  # thought generator: up to 3 candidate strategies
BEAM_WIDTH = 2  # keep the top 2 after depth-1 scoring, prune the rest

_CHECKPOINT_NUM_RE = re.compile(r"(\d+\.\d+)")


def _extract_checkpoint(name: str) -> str | None:
    match = _CHECKPOINT_NUM_RE.search(name or "")
    return match.group(1) if match else None


def _parse_due(due_str: str | None) -> date | None:
    if not due_str:
        return None
    try:
        return datetime.strptime(due_str, "%d-%b-%y").date()
    except ValueError:
        return None


def _pending_assignments() -> list[dict]:
    info = get_course_info()
    completed_names = set(get_completed())
    pending = []
    for a in info.get("assignments", []):
        name = a.get("assignment") if isinstance(a, dict) else a
        if not name or name in completed_names:
            continue
        pending.append(
            {
                "name": name,
                "due_date": _parse_due(a.get("due_date") if isinstance(a, dict) else None),
                "checkpoint": _extract_checkpoint(name),
            }
        )
    return pending


def _evidence_score(checkpoint: str | None) -> float:
    """0..1, higher = the checkpoint's requirements are well-grounded in
    ingested materials. Chroma L2 distance, lower = more similar; ~0.8-1.4 is
    the observed range in this project's data (see rag/qa.py), mapped to
    1.0-0.0."""
    if not checkpoint:
        return 0.5
    try:
        store = get_vectorstore(get_embeddings())
        results = store.similarity_search_with_score(
            f"checkpoint {checkpoint} requirements", k=1, filter={"checkpoint": checkpoint}
        )
    except Exception:
        logger.exception("tot evidence_score lookup failed for checkpoint=%s", checkpoint)
        return 0.5
    if not results:
        return 0.0
    _, distance = results[0]
    return max(0.0, min(1.0, (1.4 - distance) / 0.6))


def _prerequisite_score(checkpoint: str | None, completed_checkpoints: set[str]) -> float:
    """1.0 if there's no meaningful prerequisite gap, else 0.5. A checkpoint's
    prerequisite is specifically the immediately preceding one (e.g. 3.1's is
    2.1, not just "some" earlier checkpoint) -- otherwise finishing 1.1 alone
    would wrongly grant every later checkpoint full prerequisite credit."""
    if not checkpoint:
        return 1.0
    try:
        major = float(checkpoint)
    except ValueError:
        return 1.0
    if major <= 1.1:
        return 1.0
    prior_major = major - 1.0
    has_immediate_prior_done = any(
        _is_float(cp) and abs(float(cp) - prior_major) < 0.01 for cp in completed_checkpoints
    )
    return 1.0 if has_immediate_prior_done else 0.5


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _urgency_scores(pending: list[dict]) -> dict[str, float]:
    """Urgency ranked relative to the current pending set, not an absolute
    calendar window -- so the nearest-due item always scores highest (1.0)
    and the furthest scores lowest (0.0), whether everything is comfortably
    in the future or (very commonly, for a procrastinating student) already
    overdue. An absolute day-count formula saturates at a flat 1.0 for any
    overdue item, silently losing all ordering among the very items where
    ranking matters most."""
    dated = [p for p in pending if p["due_date"]]
    if not dated:
        return {p["name"]: 0.3 for p in pending}
    due_min = min(p["due_date"] for p in dated)
    due_max = max(p["due_date"] for p in dated)
    span = (due_max - due_min).days or 1
    scores = {p["name"]: 0.3 for p in pending}
    for p in dated:
        scores[p["name"]] = 1.0 - (p["due_date"] - due_min).days / span
    return scores


def _generate_candidates(pending: list[dict]) -> list[dict]:
    """Thought generator: up to BRANCH_WIDTH candidates from distinct
    strategies (urgency / prerequisite order / evidence readiness),
    deduplicated if two strategies agree on the same assignment."""
    if not pending:
        return []

    strategies: dict[str, dict] = {}
    with_due = [p for p in pending if p["due_date"]]
    if with_due:
        strategies["urgency"] = min(with_due, key=lambda p: p["due_date"])

    with_cp = [p for p in pending if p["checkpoint"]]
    if with_cp:
        strategies["prerequisite_order"] = min(with_cp, key=lambda p: float(p["checkpoint"]))

    strategies["readiness"] = max(pending, key=lambda p: _evidence_score(p["checkpoint"]))

    by_name: dict[str, dict] = {}
    for strategy, candidate in strategies.items():
        entry = by_name.setdefault(candidate["name"], {"candidate": candidate, "strategies": []})
        entry["strategies"].append(strategy)
    return list(by_name.values())[:BRANCH_WIDTH]


def _score(candidate: dict, completed_checkpoints: set[str], urgency_scores: dict[str, float]) -> float:
    urgency = urgency_scores.get(candidate["name"], 0.3)
    evidence = _evidence_score(candidate["checkpoint"])
    prereq = _prerequisite_score(candidate["checkpoint"], completed_checkpoints)
    return (urgency + evidence + prereq) / 3


def run_next_priority_tot(llm) -> dict:
    """Runs the beam search and returns {"candidates": [...], "chosen": ...}
    as ReasoningCandidate objects, so reasoning_node can plug this in without
    the rest of the pipeline (evaluation/response) needing to change."""
    pending = _pending_assignments()
    if not pending:
        candidate = ReasoningCandidate(
            answer="Nothing pending -- every known assignment is marked completed "
            "(or there's no assignment data).",
            confidence=1.0,
            rationale="no pending assignments to reason over",
        )
        return {"candidates": [candidate], "chosen": candidate}

    completed_checkpoints = {
        cp for cp in (_extract_checkpoint(n) for n in get_completed()) if cp
    }
    urgency_scores = _urgency_scores(pending)

    # Depth 1: generate + score candidates, prune to the beam.
    depth1 = _generate_candidates(pending)
    scored = sorted(
        (
            {**entry, "score": _score(entry["candidate"], completed_checkpoints, urgency_scores)}
            for entry in depth1
        ),
        key=lambda e: e["score"],
        reverse=True,
    )
    survivors = scored[:BEAM_WIDTH]
    logger.info(
        "decision=tot_depth1 branches=%d survivors=%s",
        len(scored),
        [(s["candidate"]["name"], round(s["score"], 2), s["strategies"]) for s in survivors],
    )

    # Depth 2: elaborate each surviving branch into an actionable thought.
    candidates = []
    for entry in survivors:
        c = entry["candidate"]
        due_str = c["due_date"].isoformat() if c["due_date"] else "no known due date"
        response = llm.invoke(
            [
                (
                    "system",
                    "You are the Reasoning Agent's thought elaborator. In one or two "
                    "sentences, explain why the learner should work on this item next, "
                    "referencing the reason(s) given. Be direct and actionable.",
                ),
                (
                    "human",
                    f"Item: {c['name']}\nDue: {due_str}\n"
                    f"Reasons this was proposed: {', '.join(entry['strategies'])}",
                ),
            ]
        )
        candidates.append(
            ReasoningCandidate(answer=response.content, confidence=entry["score"], rationale=", ".join(entry["strategies"]))
        )

    chosen = max(candidates, key=lambda c: c.confidence)
    logger.info(
        "decision=tot_depth2 chosen=%r confidence=%.2f",
        chosen.answer[:80], chosen.confidence,
    )
    return {"candidates": candidates, "chosen": chosen}
