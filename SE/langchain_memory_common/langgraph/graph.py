##      Define the graph
from langgraph.graph import END, StateGraph
from langchain_memory_common.langgraph.state import AgentState
from langchain_memory_common.langgraph.nodes import run_agent, execute_tools, should_continue, call_model
from langgraph.checkpoint.memory import MemorySaver

# lang-graph structure
def create_graph_structure():
    workflow = StateGraph(AgentState)
    workflow.set_entry_point("agent")
    workflow.add_node("agent", run_agent)
    workflow.add_node("action", execute_tools)
    workflow.add_node("memory_node", call_model)
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "action",
            "end": "memory_node"
        }
    )
    workflow.add_edge('action', 'agent')
    workflow.add_edge("memory_node", END)
    return workflow

# app
def set_workflow_to_app():   # really this is the graph.
    # Add simple in-memory checkpointer
    memory = MemorySaver()
    return create_graph_structure().compile(checkpointer=memory)    # to be like an app in here