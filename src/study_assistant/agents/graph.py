"""Builds the multi-agent pipeline from the Capstone Checkpoint 5.1 design:
Planner -> Retrieval -> Reasoning -> Evaluation -> Response, with a targeted
feedback loop from Evaluation back to Retrieval. See nodes.py for the agents
themselves and ARCHITECTURE.md for the full design rationale.
"""

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from study_assistant.agents import nodes
from study_assistant.agents.state import PipelineState


def build_agent() -> CompiledStateGraph:
    graph = StateGraph(PipelineState)

    graph.add_node("init", nodes.init_node)
    graph.add_node("planner", nodes.planner_node)
    graph.add_node("retrieval", nodes.retrieval_node)
    graph.add_node("reasoning", nodes.reasoning_node)
    graph.add_node("evaluation", nodes.evaluation_node)
    graph.add_node("prepare_retry", nodes.prepare_retry_node)
    graph.add_node("response", nodes.response_node)

    graph.set_entry_point("init")
    graph.add_edge("init", "planner")
    graph.add_edge("planner", "retrieval")
    graph.add_edge("retrieval", "reasoning")
    graph.add_edge("reasoning", "evaluation")
    graph.add_conditional_edges(
        "evaluation",
        nodes.route_after_evaluation,
        {"retry": "prepare_retry", "response": "response"},
    )
    graph.add_edge("prepare_retry", "retrieval")
    graph.add_edge("response", END)

    return graph.compile()
