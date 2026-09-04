import logging

from study_assistant.config import settings

LOGGER_NAME = "study_assistant"


def configure_logging() -> None:
    """Always-on runtime log for guardrail/audit events (retrieval confidence,
    escalations, errors) — separate from --debug's verbose LangChain tracing."""
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(settings.guardrail_log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
