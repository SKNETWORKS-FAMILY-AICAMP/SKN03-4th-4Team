from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain_openai.chat_models import ChatOpenAI
from common.tool_module import tools  # tools 가져오기
import os

# 프롬프트 및 LLM 초기화
prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(model="gpt-4o-mini", streaming=True, api_key=os.getenv("OPENAI_API_KEY"))

# 에이전트 생성
agent_runnable = create_openai_functions_agent(llm, tools, prompt)

def run_agent(data):
    """Runs the agent and returns the outcome."""
    agent_outcome = agent_runnable.invoke(data)
    return {"agent_outcome": agent_outcome}