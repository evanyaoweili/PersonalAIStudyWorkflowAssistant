# Architecture

This document explains how the Personal AI Study Workflow Assistant works end to end:
what happens when you run each command, how the pieces fit together, and why a few
things are built the way they are.

For setup/usage instructions, see [README.md](README.md).

## What it is

A command-line agent, built as a [LangGraph](https://github.com/langchain-ai/langgraph)
multi-agent pipeline on top of OpenAI models, that helps you:

- **Ask questions about your own study materials** (PDFs, notes) — a retrieval-augmented
  generation (RAG) pipeline grounds answers in content you've ingested.
- **Look up course/roster info** (assignments, grading breakdown, students, teachers) —
  served from a structured JSON file via dedicated lookups, not RAG.
- **Build a study schedule** — a deterministic greedy scheduler, no LLM involved.

The pipeline is five specialized agents — Planner, Retrieval, Reasoning, Evaluation,
Response — instead of one LLM freely choosing tools turn by turn. See
[Multi-agent pipeline](#multi-agent-pipeline-capstone-checkpoint-51) below for why and how.

## High-level diagram

```mermaid
flowchart TB
    User(["User"]) -->|"study-assistant chat"| CLI[main.py CLI]
    User -->|"study-assistant ingest"| CLI

    CLI -->|ingest| Loader[ingestion/loader.py]
    Loader -->|"reads *.pdf / *.txt / *.md"| Materials[("data/materials/")]
    Loader --> Splitter[chunk + tag metadata]
    Splitter --> Embed1[ingestion/embeddings.py]
    Embed1 --> Chroma[("data/chroma/\nvector store")]

    CLI -->|chat| Init[agents/nodes.py\ninit]
    Init --> Planner[Planner]
    Planner --> Retrieval[Retrieval]
    Retrieval --> Reasoning[Reasoning]
    Reasoning --> Evaluation[Evaluation]
    Evaluation -->|grounded| Response[Response]
    Evaluation -->|"not grounded\n(max 1 retry)"| Retrieval

    Retrieval -->|"ask_study_materials"| RAG[rag/qa.py]
    RAG --> Chroma
    Retrieval -->|"get_course_info /\nget_person_info / whats_next"| CourseData[course_data.py]
    CourseData --> JSON[("data/course_data.json")]
    Retrieval -->|"plan_study_schedule"| Scheduler[planning/scheduler.py]

    Planner <-->|LLM call| LLM[["OpenAI Chat Model"]]
    Reasoning <-->|LLM call| LLM
    Response <-->|LLM call| LLM
```

## Components

| Module | Responsibility |
|---|---|
| [`main.py`](src/study_assistant/main.py) | `argparse` CLI: `ingest`, `chat`, and a global `--debug` flag. Thin — just wires args to the functions below. |
| [`config.py`](src/study_assistant/config.py) | Loads `.env` and exposes a single `settings` object (API key, model names, file paths). Everything else reads config from here, never from `os.environ` directly. |
| [`ingestion/loader.py`](src/study_assistant/ingestion/loader.py) | Walks `data/materials/`, loads `.pdf`/`.txt`/`.md` files, splits them into chunks (`RecursiveCharacterTextSplitter`). |
| [`ingestion/embeddings.py`](src/study_assistant/ingestion/embeddings.py) | Builds the `OpenAIEmbeddings` client used to embed both ingested chunks and incoming questions. |
| [`ingestion/vectorstore.py`](src/study_assistant/ingestion/vectorstore.py) | Wraps a local `Chroma` collection persisted at `data/chroma/`; `index_documents` adds embedded chunks to it. |
| [`rag/qa.py`](src/study_assistant/rag/qa.py) | Given a question: similarity-search the vector store for top-`k` chunks, stuff them into a prompt, ask the chat model, return the answer. |
| [`course_data.py`](src/study_assistant/course_data.py) | Loads `data/course_data.json` (course name/term, assignments, grading breakdown, student/teacher roster) and exposes lookup functions. See [Why course data isn't RAG-ingested](#why-course-data-isnt-rag-ingested) below. |
| [`course_progress.py`](src/study_assistant/course_progress.py) | Tracks which assignments the user has marked completed, persisted to `data/course_progress.json`. Backs the `whats_next` tool. |
| [`planning/models.py`](src/study_assistant/planning/models.py) | Pydantic models: `StudyTask` (input) and `StudyBlock` (a scheduled chunk of work). |
| [`planning/scheduler.py`](src/study_assistant/planning/scheduler.py) | Pure, deterministic greedy algorithm — no LLM call. Allocates daily hours to tasks, nearest deadline first, ties broken by priority. |
| [`agents/tools.py`](src/study_assistant/agents/tools.py) | Wraps the functions above as LangChain `@tool`s: `ask_study_materials`, `plan_study_schedule`, `get_course_info`, `get_person_info`, `list_roster`, `whats_next`, `mark_assignment_completed`. Invoked directly by the Retrieval Agent (`.invoke(...)`), not chosen by an LLM's own tool-calling loop. |
| [`agents/state.py`](src/study_assistant/agents/state.py) | `PipelineState` (the shared state threaded through the graph) and the structured-output Pydantic models (`PlannerDecision`, `ReasoningCandidate(s)`, `ScheduleExtract`). |
| [`agents/nodes.py`](src/study_assistant/agents/nodes.py) | The five agents themselves as plain functions over `PipelineState`, plus the evaluation-routing and retry logic. See [Multi-agent pipeline](#multi-agent-pipeline-capstone-checkpoint-51) below. |
| [`agents/graph.py`](src/study_assistant/agents/graph.py) | Wires `nodes.py`'s functions into a LangGraph `StateGraph` with the sequential core, the conditional evaluation branch, and the retry loop. |
| [`gui.py`](src/study_assistant/gui.py) | Optional Gradio chat UI (`study-assistant gui`, needs the `gui` extra). See [GUI](#gui) below. |

## GUI

`study-assistant gui` launches a Gradio app with a chat pane and two live tabs:

- **Trace** — built by calling `agent.stream(..., stream_mode="values")` instead of
  `.invoke()`. LangGraph yields the full pipeline state after every node, so each new
  value of `intent`/`evidence`/`chosen`/`grounded` is rendered as it appears — Planner,
  Retrieval, Reasoning, and Evaluation's output show up live, labeled `(retry N)` if
  Evaluation's feedback loop sends it back through Retrieval a second time.
- **Log** — tails `data/guardrail_events.log`, the same file `observability.py` writes to
  for the CLI. Independent of `--debug`, which is far noisier (raw HTTP + full LangChain
  trace) and not meant for this kind of at-a-glance view.

Conversation history is threaded across turns via a `gr.State` holding the LangChain
message list (not just chat display text), so follow-up questions ("how many points is
it worth?") resolve correctly against prior turns — the CLI's `cmd_chat`, by contrast,
sends only the latest message each turn and has no cross-turn memory.

This was adapted from a different project's Gradio demo
(`Module5/Local-Agent-Demo`), which is built on a from-scratch Ollama+MCP agent loop with
no LangChain involved at all. Only the UI shape (chat + live side panels) carried over;
the event-handling logic here is written from scratch against this project's LangGraph
agent and tools. That demo's Context/Memory/Tools panels weren't ported, since they map to
concepts (windowed short-term memory, MCP tool discovery, a separate long-term memory
store) that don't exist in this project's architecture — Trace and Log are what's
actually meaningful here.

## Two commands, two flows

### `study-assistant ingest`

```
load_documents(data/materials/) → split_documents() → get_embeddings() → index_documents()
```

One-shot, no LLM chat call. Reads whatever files are currently in `data/materials/`,
chunks them, embeds them, and writes/updates the persisted Chroma collection at
`data/chroma/`. Re-running it re-embeds and appends — it does not currently de-duplicate
against a previous run.

### `study-assistant chat`

```
input("> ") → agent.invoke({"messages": [...]}) → print(result)
```

Each turn runs the full five-agent pipeline once (see
[Multi-agent pipeline](#multi-agent-pipeline-capstone-checkpoint-51) below), then the
final assistant message is printed.

Run with `--debug` to see it directly — it turns on LangChain's `set_debug(True)`, which
prints every chain/LLM step (each node in the pipeline is a LangGraph chain), plus raw
HTTP-level logging for the OpenAI calls.

## Why course data isn't RAG-ingested

`data/course_data.json` deliberately lives outside `data/materials/` and is never
chunked/embedded. It's read on demand by `get_course_info`/`get_person_info` at tool-call
time. This matters for two reasons:

- **Freshness** — a JSON file read on every call always reflects the current file
  contents; RAG-ingested content is a stale snapshot from whenever you last ran `ingest`.
- **Structure** — this is small, structured, per-record data (a roster, a grading table).
  Answering "what's Alex Chen's background?" by exact key lookup is more reliable than
  hoping the right chunk gets retrieved by similarity search.

## Configuration

All settings live in `.env` (copy from `.env.example`) and are loaded once in
[`config.py`](src/study_assistant/config.py):

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — | required; used for both chat and embedding calls |
| `STUDY_ASSISTANT_MODEL` | `gpt-4.1-mini` | chat model for the agent and RAG answers |
| `STUDY_ASSISTANT_EMBEDDING_MODEL` | `text-embedding-3-small` | embedding model for ingestion and query embedding |
| `STUDY_MATERIALS_DIR` | `data/materials` | where `ingest` looks for source documents |
| `CHROMA_PERSIST_DIR` | `data/chroma` | where the vector store is persisted |
| `COURSE_DATA_PATH` | `data/course_data.json` | source for `get_course_info`/`get_person_info` |

## What's checked into git vs. generated locally

- `data/materials/*` and `data/chroma/` are gitignored — they're your personal study
  content and the derived vector index, regenerated locally via `ingest`.
- `data/course_data.json` is *not* gitignored (only `.gitkeep` under `data/materials/`
  is tracked today) — it's small structured sample/course data rather than a personal
  document dump.

## Tech stack

- **LangGraph** — a custom `StateGraph` (not the prebuilt `create_agent` loop) implementing the five-agent pipeline
- **OpenAI** — chat completions (`gpt-4.1-mini`) and embeddings (`text-embedding-3-small`)
- **Chroma** — local, file-persisted vector store
- **Pydantic** — validates tool arguments (`StudyTask`) and planning data models
- **argparse** — CLI parsing, no external CLI framework

## Safety guardrails

Implements the guardrails from the Capstone Checkpoint 6.1 Safety Guardrails and Human
Intervention Plan:

| Guardrail | Where | Behavior |
|---|---|---|
| Input validation | [`main.py`](src/study_assistant/main.py) `cmd_chat` | Empty/whitespace-only input is skipped rather than sent to the agent. |
| Evidence grounding | [`rag/qa.py`](src/study_assistant/rag/qa.py) | If retrieval returns zero chunks, the tool returns a fixed "no materials ingested" message and never calls the LLM — it can't invent an answer with no evidence. |
| Confidence-based escalation | [`rag/qa.py`](src/study_assistant/rag/qa.py) | The best retrieval distance is compared to `RAG_MAX_DISTANCE` (default `1.3`). Above it, the prompt tells the model confidence is `'low'` and to make clear the answer needs verification, instead of answering with false confidence. |
| Restricted tool access | [`agents/tools.py`](src/study_assistant/agents/tools.py), [`agents/nodes.py`](src/study_assistant/agents/nodes.py) `RESPONSE_PROMPT` | There are no tools to submit assignments, modify records, or send messages — only retrieval, lookup, and scheduling. The Response Agent's prompt states this explicitly so it doesn't claim otherwise. |
| Runtime/audit logging | [`observability.py`](src/study_assistant/observability.py) | An always-on (independent of `--debug`) logger writes retrieval confidence, escalation events, and unhandled errors to `GUARDRAIL_LOG_PATH` (default `data/guardrail_events.log`) — distinct from `--debug`'s verbose LangChain trace, meant for reviewing agent behavior over time. |
| Graceful error handling | [`main.py`](src/study_assistant/main.py) `cmd_chat` | An exception during `agent.invoke` is logged and the user gets a plain-language message, instead of a raw traceback killing the chat loop. |

Not implemented: automatic scope classification (rejecting off-topic requests before
they reach the agent) — currently the system prompt is the only scope control, which is
a soft measure, not a hard guardrail.

## "What's next" (Capstone Checkpoint 2.1)

Checkpoint 2.1's design doc narrows the project to answering one question reliably:
*"What should I work on next for my course?"* Two tools implement this directly:

- **`whats_next`** — reads assignments from `course_data.py`, excludes anything in
  `course_progress.py`'s completed set, parses `due_date` (`%d-%b-%y`), and returns the
  nearest-due pending item.
- **`mark_assignment_completed`** — case-insensitive partial-name match against known
  assignments, persists the canonical name to `data/course_progress.json`.

Deliberately out of scope from that same doc: a generic long-term memory tool
(`project_memory.json`/`prior_feedback.json`) and a reminder tool — both are real gaps,
but weren't added here since they weren't clearly CMU-course-scoped requests on their
own; add them as a separate, explicit ask if needed.

## Checkpoint-aware retrieval + MMR (Capstone Checkpoint 3.1)

Checkpoint 3.1's design doc calls out a specific retrieval failure mode: asking about one
checkpoint could retrieve another's content, since files about similar topics (e.g.
several capstone checkpoints) can be semantically close together. It also proposes MMR to
avoid several near-duplicate chunks crowding out distinct information. Both are
implemented in [`rag/qa.py`](src/study_assistant/rag/qa.py):

- **Metadata tagging** — [`loader.py`](src/study_assistant/ingestion/loader.py) tags each
  chunk with `checkpoint` (parsed from the filename, e.g. `"3.1"`) and `doc_type`
  (`assignment_brief` / `design_submission` / `reference`) at ingestion time.
- **Checkpoint-scoped retrieval** — if the question names a checkpoint/assignment number
  (`"checkpoint 3.1"`, case-insensitive), retrieval is first tried filtered to
  `{"checkpoint": "3.1"}`. If that filter returns anything, it's used exclusively; if not
  (e.g. the number doesn't match anything ingested), it falls back to an unfiltered
  search rather than hard-failing.
- **MMR for context** — the actual passages handed to the LLM come from
  `max_marginal_relevance_search` (not plain top-`k` similarity), so a set of
  near-duplicate chunks doesn't dominate the context at the expense of distinct
  information. The confidence-threshold guardrail still uses plain
  `similarity_search_with_score`, since MMR doesn't return comparable distances.

This is a real, reproducible risk in this exact dataset, not a hypothetical: asking
`"What are the requirements for checkpoint 3.1?"` with the filter disabled pulls
checkpoint 4.1 and 5.1 chunks into the top-4 (scores 1.02 and 1.11, close enough to the
top match's 0.99 to make the cut) — the filter excludes them.

## Multi-agent pipeline (Capstone Checkpoint 5.1)

Checkpoint 5.1's design doc replaces the single LLM-driven tool-calling loop described
above with five specialized agents, each independently checkable, coordinated by a
LangGraph `StateGraph` (not the prebuilt `create_agent`). Rationale from the doc:
*"Combining planning, retrieval, reasoning, verification, and writing in one prompt would
make mistakes harder to detect. Separating these responsibilities allows each result to
be checked before it reaches the learner."*

### The five agents ([`agents/nodes.py`](src/study_assistant/agents/nodes.py))

| Agent | Role | LLM call? |
|---|---|---|
| **Planner** | Classifies the request into one of 8 intents (`study_materials`, `course_info`, `roster`, `person_info`, `next_priority`, `mark_completed`, `scheduling`, `general`) and extracts a self-contained `detail` (resolving references like "it" against conversation history), via `with_structured_output(PlannerDecision)`. | Yes |
| **Retrieval** | Invokes the one existing `agents/tools.py` tool matching the intent (`.invoke(...)`, not LLM-chosen) and collects its output as `evidence`. Reuses every guardrail/log line already built into those tools. | No — plain function dispatch |
| **Reasoning** | For open-ended intents (`study_materials`, `next_priority`, `general`), proposes up to 2 candidate answers with self-rated confidence via `with_structured_output(ReasoningCandidates)` — a lightweight Tree-of-Thought, capped at 2 branches per the doc's own caution against "excessive branching wast[ing] time and tokens." For deterministic lookups (`course_info`, `roster`, `person_info`, `mark_completed`), skips branching entirely — one candidate, confidence 1.0 unless the evidence signals a not-found/empty result. | Only for open-ended intents |
| **Evaluation** | Gates on `chosen.confidence >= 0.5` and the evidence not matching a known not-found/no-evidence marker. `general` intent always passes (nothing to ground). Routes to Response if grounded, or back to Retrieval (capped at 1 retry) if not. | No |
| **Response** | Formats the approved result for the learner. Deterministic-lookup and scheduling intents pass the evidence straight through (no LLM call — nothing to add). Open-ended intents get one LLM polish call. If Evaluation never grounded the answer (even after the retry), returns a fixed "I don't have enough reliable information" message instead of guessing. | Only for open-ended intents |

### Coordination and the feedback loop ([`agents/graph.py`](src/study_assistant/agents/graph.py))

```
init → planner → retrieval → reasoning → evaluation ─┬─(grounded)──→ response → END
                     ↑                                └─(not grounded,
                     │                                    retry_count<1)
                     └──────────── prepare_retry ←────────┘
```

This matches the doc's hybrid strategy: a **sequential core** (Planner→Retrieval→
Reasoning→Evaluation→Response), a **targeted feedback loop** (Evaluation routes back to
Retrieval specifically, not a full restart from Planner), capped at one retry so an
unsupported answer can't loop forever. `prepare_retry_node` strips any checkpoint/
assignment number from `detail` before looping back, which forces `rag/qa.py`'s
checkpoint filter (see above) to widen to an unfiltered MMR search — a concrete, working
instance of "the evaluator sends the request back to Retrieval... instead of restarting
the entire workflow."

The doc's "graph-based reasoning" branch (small exploration inside the Reasoning stage)
is the capped 2-candidate proposal above, not a deep tree search — the doc itself frames
ToT as needing limited branch count and depth, and this project's evaluation showed
little value in branching factual lookups at all (see the passthrough-intent list above).

### Compatibility with the CLI and GUI

`PipelineState`'s `messages` field is only touched twice: `init_node` reads the latest
`HumanMessage`, and `response_node` appends the final `AIMessage` — so
`main.py`'s `agent.invoke({"messages": [...]})` / `result["messages"][-1].content` and
`gui.py`'s `agent.stream({"messages": history}, ...)` both work completely unchanged.
`gui.py`'s Trace tab *did* need updating, though — the old trace watched for
`AIMessage(tool_calls=...)`/`ToolMessage` pairs appearing in `messages` as a tool-calling
loop ran; this pipeline doesn't touch `messages` until the very end, so the Trace tab now
instead watches `intent`/`evidence`/`chosen`/`grounded` appear across pipeline steps.

### A known trade-off, not a bug

The Planner picks exactly **one** intent per turn. This means a request needing two of the
existing capabilities in one breath doesn't get both — e.g. after "what should I work on
next?" answers with a checkpoint name, asking "how many points is it worth?" gets
classified as `study_materials` (a natural reading of "how many points"), which searches
`data/materials/` rather than routing to `course_info`, where the point value actually
lives in `course_data.json`. It correctly says it doesn't know rather than guessing — it
just doesn't find the real answer sitting one intent over. This is the direct, honest cost
of the doc's own design choice (fixed specialized roles over one flexible agentic loop);
fixing it would mean letting Planner select multiple intents per turn, which reintroduces
exactly the harder-to-check complexity the design explicitly trades away.
