# Personal AI Study Workflow Assistant

CMU AI Agent Course capstone project — an agent that helps you understand your
study materials and plan your study time.

## Features

- **Material Q&A (RAG)** — ingest notes/PDFs and ask questions grounded in them.
- **Study planning** — build a day-by-day study schedule from a list of tasks,
  deadlines, and available hours per day.
- Built as a [LangGraph](https://github.com/langchain-ai/langgraph) ReAct agent
  with OpenAI as the underlying model, exposing the above as tools.

See [ARCHITECTURE.md](ARCHITECTURE.md) for how the pieces fit together — the agent's
tool-calling loop, the RAG pipeline, and why course roster data is served through tools
instead of being ingested into the vector store.

## Project layout

```
src/study_assistant/
  agents/       LangGraph agent + tool definitions
  ingestion/    document loading, chunking, embeddings, vector store
  rag/          retrieval-augmented Q&A over ingested materials
  planning/     study task/schedule models and scheduling logic
  config.py     environment-based settings
  main.py       CLI entry point (ingest / chat / gui)
  gui.py        optional Gradio chat UI with live Trace/Log panels
tests/          pytest suite
data/materials/ drop your study materials (PDFs, .txt, .md) here
```

## Setup

Run these from the project root (the folder this README is in).

**PowerShell** (VS Code's default terminal on Windows):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env
```

**Git Bash:**

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -e ".[dev]"
cp .env.example .env
```

Then:

1. Open `.env` and replace `OPENAI_API_KEY=your-api-key-here` with a real key from
   https://platform.openai.com/account/api-keys.
2. Drop study materials (`.pdf`, `.txt`, `.md`) into `data/materials/`.

`pip install -e ".[dev]"` installs the app plus test/lint tooling (pytest, ruff). There's
also an optional `docs` extra (`python-docx`, `matplotlib`) for generating Word-format
documentation — most people won't need it: `pip install -e ".[docs]"`.

### Verifying the install

```bash
study-assistant --help
```

If that prints usage instead of "command not found", the venv is active and the install
worked. If `study-assistant` isn't found, either the venv isn't activated in this shell
(re-run the `activate` line above) or you're in a different terminal/shell than the one
you ran setup in — each shell needs its own `activate`.

## Usage

```bash
# Ingest study materials into the vector store
study-assistant ingest

# Chat with the agent
study-assistant chat

# Launch the Gradio chat GUI (needs the "gui" extra, see below)
study-assistant gui
```

The GUI needs `gradio`, an optional extra so CLI-only users don't need to install it:
`pip install -e ".[gui]"`. It opens in your browser with a chat pane on the left and two
tabs on the right — **Trace** (the tool calls the agent makes this turn, live) and **Log**
(a tail of the guardrail/audit log from `observability.py`).

### Debugging

```bash
study-assistant --debug chat
```

`--debug` turns on LangChain's chain/tool-call tracing plus verbose HTTP logging, so you
can see exactly which tool the agent picked and why. There are also two VS Code "Run and
Debug" configurations (`.vscode/launch.json`) — "study-assistant: chat (debug)" and
"study-assistant: ingest (debug)" — for stepping through with breakpoints instead.

If you'd rather skip shell activation entirely, call the executable by its full path —
this works regardless of which shell or activation state you're in:

```powershell
.\.venv\Scripts\study-assistant.exe --debug chat   # PowerShell
```
```bash
.venv/Scripts/study-assistant.exe --debug chat     # Git Bash
```

## Tests

```bash
pytest
```
