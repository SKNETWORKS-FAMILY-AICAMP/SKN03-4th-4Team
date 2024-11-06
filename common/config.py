from langchain_core.agents import AgentFinish

def should_continue(data):
    """Checks if the agent outcome indicates the end of the conversation."""
    if isinstance(data['agent_outcome'], AgentFinish):
        return "end"
    else:
        return "continue"