from langchain_core.agents import AgentFinish
from langgraph.prebuilt.tool_executor import ToolExecutor
from langchain_memory_common.langgraph.tools import get_langgraph_tools 
from langchain_memory_common.langgraph.agent import create_openai_langgraph_agent
from langchain_core.messages import trim_messages, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_memory_common.model import creat_chat_model
from langchain_memory_common.langgraph.state import AgentState

# 트리머 설정 함수
def set_trimmer():
    return trim_messages(strategy="last", max_tokens=50, token_counter=len)

# 모델을 호출하는 함수
def call_model(state: AgentState):
    # print(state)
    # 필요 시 메시지를 트리밍
    trimmed_messages = set_trimmer().invoke(state["chat_history"])
    # print("--------------")
    # print(trimmed_messages)
    # 시스템 프롬프트와 메시지 설정
    messages = [SystemMessage(content="You are a helpful assistant. Answer all questions to the best of your ability.")] + trimmed_messages
    # 모델 호출
    agent_outcome = creat_chat_model().invoke(messages)
    # 상태 업데이트
    state["agent_outcome"] = agent_outcome
    state["chat_history"].append({'role': 'assistant', 'content': agent_outcome.content})
    return state

# 에이전트를 실행하는 함수
def run_agent(state):
    agent_outcome = create_openai_langgraph_agent().invoke(state)
    state["agent_outcome"] = agent_outcome
    # 사용자 메시지를 메모리에 추가
    state["chat_history"].append({'role': 'user', 'content': state["input"]})
    return state

# 도구를 실행하는 함수
def execute_tools(state):
    agent_action = state['agent_outcome']
    output = ToolExecutor(get_langgraph_tools()).invoke(agent_action)
    state["intermediate_steps"] = [(agent_action, str(output))]
    return state

# 메시지 분기 함수
def should_continue(state):
    if isinstance(state['agent_outcome'], AgentFinish):
        return "end"
    else:
        return "continue"

# # 메모리 세이버 설정 (옵션)
# def save_memory(state):
#     # 메모리를 저장하는 로직을 구현
#     MemorySaver().save(state["messages"])
#     return state

# # 전체 흐름을 관리하는 함수 예시
# def main_flow(user_input):
#     # 초기 상태 설정
#     state = AgentState()
#     state["user_input"] = user_input
#     state["messages"] = []

#     # 에이전트 실행
#     state = run_agent(state)

#     # 계속 진행할지 결정
#     while should_continue(state) != "end":
#         # 도구 실행
#         state = execute_tools(state)
#         # 모델 호출
#         state = call_model(state)
#         # 에이전트 재실행
#         state = run_agent(state)

#     # 대화 종료 시 메모리 저장 (옵션)
#     state = save_memory(state)

#     # 최종 응답 반환
#     return state["agent_outcome"].return_values['output']

# # 사용 예시
# if __name__ == "__main__":
#     user_input = "안녕하세요, 오늘 날씨가 어떤가요?"
#     response = main_flow(user_input)
#     print("Assistant:", response)
