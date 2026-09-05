"""gui.py -- Gradio front-end for the study assistant agent.

Run with:
    study-assistant gui

Layout:

    +--------------------------+-----------------------------+
    |                          | [Trace]           [Log]     |
    |   chat with the agent    |                              |
    |                          |  live per-turn tool-call     |
    |                          |  trace / guardrail audit log |
    |  [ type here ]  [send]   |                              |
    +--------------------------+-----------------------------+

The Trace tab is built by streaming the LangGraph pipeline step by step, so
you can watch Planner -> Retrieval -> Reasoning -> Evaluation -> Response run
(with Evaluation's feedback loop back to Retrieval visible as a labeled
"retry" round) rather than just getting a final answer. The Log tab tails
the same guardrail/audit log written by observability.py (retrieval
confidence, escalations, errors) -- independent of --debug, which is much
noisier.
"""

from pathlib import Path

import gradio as gr
from langchain_core.messages import HumanMessage

from study_assistant.agents.graph import build_agent
from study_assistant.config import settings
from study_assistant.observability import configure_logging

_AGENT = None


def _get_agent():
    global _AGENT
    if _AGENT is None:
        _AGENT = build_agent()
    return _AGENT


def render_trace(events: list[str]) -> str:
    if not events:
        return "### Trace\n_Send a message to see Planner -> Retrieval -> Reasoning -> Evaluation -> Response run._"
    return "### Trace\n" + "\n\n".join(events)


def _preview(value, limit: int = 400) -> str:
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "..."


def render_log(n: int = 20) -> str:
    path = settings.guardrail_log_path
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return "### Guardrail log\n_No events logged yet -- send a message first._"
    tail = lines[-n:]
    if not tail:
        return "### Guardrail log\n_No events logged yet._"
    body = "".join(tail)
    return f"### Guardrail log\n_last {len(tail)} of {len(lines)} event(s), `{path}`_\n\n```\n{body}```"


def on_send(user_text: str, chat: list[dict], history: list):
    if not user_text.strip():
        yield "", chat, history, render_trace([]), render_log()
        return

    chat = chat + [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": ""},
    ]
    new_history = history + [HumanMessage(content=user_text)]
    events: list[str] = []
    seen: dict = {}
    agent = _get_agent()

    final_messages = new_history
    try:
        for state in agent.stream({"messages": new_history}, stream_mode="values"):
            retry = state.get("retry_count", 0)
            tag = f" (retry {retry})" if retry else ""

            if state.get("intent") is not None and seen.get("intent") != (state["intent"], state.get("detail"), retry):
                seen["intent"] = (state["intent"], state.get("detail"), retry)
                events.append(f"**Planner{tag}:** intent=`{state['intent']}`, detail=`{state.get('detail', '')}`")

            if state.get("evidence") is not None and seen.get("evidence") != (state["evidence"], retry):
                seen["evidence"] = (state["evidence"], retry)
                events.append(f"**Retrieval{tag}:**\n```\n{_preview(state['evidence'])}\n```")

            if state.get("chosen") is not None and seen.get("chosen") != (state["chosen"], retry):
                seen["chosen"] = (state["chosen"], retry)
                c = state["chosen"]
                events.append(f"**Reasoning{tag}:** confidence={c.confidence:.2f} -- {_preview(c.answer, 200)}")

            if state.get("grounded") is not None and seen.get("grounded") != (state["grounded"], retry):
                seen["grounded"] = (state["grounded"], retry)
                events.append(f"**Evaluation{tag}:** grounded=`{state['grounded']}`")

            if state.get("final_answer"):
                chat[-1]["content"] = state["final_answer"]

            final_messages = state["messages"]
            yield "", chat, final_messages, render_trace(events), render_log()
    except Exception as e:
        chat[-1]["content"] = "Something went wrong answering that. Please try again."
        events.append(f"**Error:** `{e}`")
        yield "", chat, history, render_trace(events), render_log()


def reset_conversation():
    return [], [], render_trace([]), render_log()


_CSS_PATH = str(Path(__file__).parent / "gui.css")


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Personal AI Study Workflow Assistant") as demo:
        gr.Markdown(
            "# Personal AI Study Workflow Assistant\n"
            "Chat with your study agent. **Trace** shows the five-agent pipeline "
            "(Planner, Retrieval, Reasoning, Evaluation, Response) run this turn; "
            "**Log** tails the guardrail/audit log (retrieval confidence, "
            "escalations, errors)."
        )
        history_state = gr.State([])

        with gr.Row():
            with gr.Column(scale=5):
                chatbot = gr.Chatbot(height=460, show_label=False)
                with gr.Row(elem_id="input-row"):
                    msg_box = gr.Textbox(
                        placeholder="Ask about your course, materials, or what to work on next...",
                        show_label=False,
                        scale=8,
                        autofocus=True,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                reset_btn = gr.Button("Reset conversation", size="sm")

            with gr.Column(scale=4):
                with gr.Tab("Trace"):
                    trace_md = gr.Markdown(render_trace([]))
                with gr.Tab("Log"):
                    log_md = gr.Markdown(render_log())

        outputs = [msg_box, chatbot, history_state, trace_md, log_md]
        inputs = [msg_box, chatbot, history_state]
        send_btn.click(on_send, inputs, outputs)
        msg_box.submit(on_send, inputs, outputs)
        reset_btn.click(reset_conversation, None, [chatbot, history_state, trace_md, log_md])

    return demo


def launch() -> None:
    configure_logging()
    build_app().launch(inbrowser=True, css_paths=[_CSS_PATH])


if __name__ == "__main__":
    launch()
