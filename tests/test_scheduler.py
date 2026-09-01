from datetime import date, timedelta

from study_assistant.planning.models import StudyTask
from study_assistant.planning.scheduler import build_schedule


def test_build_schedule_allocates_all_hours_before_deadline():
    start = date(2026, 1, 1)
    task = StudyTask(topic="Linear Algebra", deadline=start + timedelta(days=3), estimated_hours=6)

    blocks = build_schedule([task], start_day=start, hours_per_day=2)

    assert sum(b.hours for b in blocks) == 6
    assert all(b.day <= task.deadline for b in blocks)


def test_build_schedule_prioritizes_nearer_deadline():
    start = date(2026, 1, 1)
    urgent = StudyTask(topic="Urgent", deadline=start, estimated_hours=2)
    later = StudyTask(topic="Later", deadline=start + timedelta(days=5), estimated_hours=2)

    blocks = build_schedule([urgent, later], start_day=start, hours_per_day=2)

    assert blocks[0].task.topic == "Urgent"
