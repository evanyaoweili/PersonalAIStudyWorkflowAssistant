from datetime import date, timedelta

from study_assistant.planning.models import StudyBlock, StudyTask


def build_schedule(
    tasks: list[StudyTask],
    start_day: date,
    hours_per_day: float,
) -> list[StudyBlock]:
    """Greedily allocate daily study hours to tasks, prioritizing the
    nearest deadline first and breaking ties by priority."""
    remaining = {t.topic: t.estimated_hours for t in tasks}
    blocks: list[StudyBlock] = []

    last_deadline = max((t.deadline for t in tasks), default=start_day)
    day = start_day
    while day <= last_deadline and any(h > 0 for h in remaining.values()):
        budget = hours_per_day
        due_today_or_later = sorted(
            (t for t in tasks if t.deadline >= day and remaining[t.topic] > 0),
            key=lambda t: (t.deadline, -t.priority),
        )
        for task in due_today_or_later:
            if budget <= 0:
                break
            allocation = min(budget, remaining[task.topic])
            if allocation <= 0:
                continue
            blocks.append(StudyBlock(task=task, day=day, hours=allocation))
            remaining[task.topic] -= allocation
            budget -= allocation
        day += timedelta(days=1)

    return blocks
