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
  main.py       CLI entry point (ingest / chat)
tests/          pytest suite
data/materials/ drop your study materials (PDFs, .txt, .md) here
```

## Setup

1. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # Windows
   pip install -e ".[dev]"
   ```

2. Copy `.env.example` to `.env` and fill in your OpenAI API key:

   ```bash
   cp .env.example .env
   ```

3. Drop study materials (`.pdf`, `.txt`, `.md`) into `data/materials/`.

## Usage

```bash
# Ingest study materials into the vector store
study-assistant ingest

# Chat with the agent
study-assistant chat

#for debugging:
#Here's the full command line, both without needing prior activation:

#PowerShell (VS Code's default terminal on Windows):


cd "c:\SourceCode\CMUAiCourse\PersonalAIStudyWorkflowAssistant"
.\.venv\Scripts\study-assistant.exe --debug chat

#Git Bash:


cd "c:\SourceCode\CMUAiCourse\PersonalAIStudyWorkflowAssistant"
.venv/Scripts/study-assistant.exe --debug chat

#Calling the .exe by its full path this way skips activation entirely and PATH issues don't matter — it'll run regardless of which shell or activation state you're in. This is exactly what I used on my end to verify it works.

#Once it starts, type a question and paste back whatever appears (the --debug flag will print verbose LangChain/OpenAI trace output) — that'll show exactly where it's failing for you.
```

## Tests

```bash
pytest
```
