from dotenv import load_dotenv
import os

load_dotenv()

from datetime import datetime
import streamlit as st
from uuid import uuid4
from common.agent_module import run_agent
from common.config import should_continue
from common.tool_module import tools, jeju_recommendation
from langgraph.prebuilt.tool_executor import ToolExecutor


# ToolExecutor 생성
tool_executor = ToolExecutor(tools)

# URL과 제목을 추출하여 표시하는 함수
def display_sites(site_data):
    st.write("### 추천 블로그 목록:")
    for site in site_data:
        url = site['url']
        title = site['content'].split(":")[0]  # 제목만 추출
        st.markdown(f"- [{title}]({url})", unsafe_allow_html=True)

# Streamlit 세션 초기화
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Streamlit 사이드바 설정
st.sidebar.title("🍊 제주도 여행 가이드")
st.sidebar.write("채팅방 목록:")

for session_id, session_data in st.session_state.chat_sessions.items():
    if st.sidebar.button(session_data["title"], key=session_id):
        st.session_state.current_session_id = session_id
        st.session_state.chat_history = session_data["messages"]

# 새 채팅방 시작 버튼
if st.sidebar.button("새 채팅방 시작"):
    new_session_id = str(uuid4())
    st.session_state.chat_sessions[new_session_id] = {
        "title": f"채팅방 {len(st.session_state.chat_sessions) + 1}",
        "messages": [],
        "created_at": datetime.now()
    }
    st.session_state.current_session_id = new_session_id
    st.session_state.chat_history = []

# 현재 선택된 채팅방 가져오기
current_session = st.session_state.chat_sessions.get(st.session_state.current_session_id)

# 메인 화면 설정
st.title("🍊 제주도 여행 가이드 🛫")

# 채팅방이 선택되었을 경우
if current_session:
    st.write(f"**{current_session['title']}**")

    # 저장된 메시지 출력 (채팅 기록이 제대로 유지되도록)
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력 받기
    prompt = st.chat_input("제주도 여행 관련 질문을 입력해주세요 ~!")

    if prompt:
        # 사용자 메시지를 세션에 추가하고 화면에 표시
        user_message = {
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now()
        }
        st.session_state.chat_history.append(user_message)
        current_session["messages"].append(user_message)

        # 사용자 메시지 화면에 표시
        with st.chat_message("user"):
            st.markdown(prompt)

        # "제주"가 포함된 경우에만 `jeju_recommendation`을 호출하여 외부 지역 질문 차단
        if "제주" in prompt:
            recommendation_response = jeju_recommendation(prompt)
            
            # 제주도 외 지역 요청 시 안내 메시지 표시 및 에이전트 실행 건너뛰기
            if recommendation_response != "제주도 여행 추천 챗봇입니다. 무엇을 도와드릴까요?":
                with st.chat_message("assistant"):
                    st.markdown(recommendation_response)
                
                # 안내 메시지를 세션 및 UI에 추가
                assistant_message = {
                    "role": "assistant",
                    "content": recommendation_response,
                    "timestamp": datetime.now()
                }
                st.session_state.chat_history.append(assistant_message)
                current_session["messages"].append(assistant_message)
            else:
                # 제주도와 관련된 질문이므로 에이전트 실행
                agent_input = {"input": prompt, "chat_history": st.session_state.chat_history, "intermediate_steps": []}
                agent_outcome = run_agent(agent_input)

                # 에이전트 결과 처리
                if should_continue(agent_outcome) == "end":
                    bot_response = agent_outcome["agent_outcome"].return_values.get("output", "응답을 생성하는 데 문제가 발생했습니다.")
                    
                    if isinstance(bot_response, list) and all('url' in item and 'content' in item for item in bot_response):
                        display_sites(bot_response)
                    else:
                        with st.chat_message("assistant"):
                            st.markdown(bot_response)
                else:
                    agent_action = agent_outcome["agent_outcome"]
                    output = tool_executor.invoke(agent_action)

                    if isinstance(output, list) and all('url' in item and 'content' in item for item in output):
                        display_sites(output)
                        bot_response = "추천 블로그 목록을 보여드렸습니다."
                    else:
                        bot_response = str(output)
                        with st.chat_message("assistant"):
                            st.markdown(bot_response)

                # 챗봇 응답을 세션 및 UI에 추가
                assistant_message = {
                    "role": "assistant",
                    "content": bot_response,
                    "timestamp": datetime.now()
                }
                st.session_state.chat_history.append(assistant_message)
                current_session["messages"].append(assistant_message)

        else:
            # 제주와 관련되지 않은 입력은 에이전트를 바로 실행
            agent_input = {"input": prompt, "chat_history": st.session_state.chat_history, "intermediate_steps": []}
            agent_outcome = run_agent(agent_input)

            # 에이전트 결과 처리
            if should_continue(agent_outcome) == "end":
                bot_response = agent_outcome["agent_outcome"].return_values.get("output", "응답을 생성하는 데 문제가 발생했습니다.")
                
                if isinstance(bot_response, list) and all('url' in item and 'content' in item for item in bot_response):
                    display_sites(bot_response)
                else:
                    with st.chat_message("assistant"):
                        st.markdown(bot_response)
            else:
                agent_action = agent_outcome["agent_outcome"]
                output = tool_executor.invoke(agent_action)

                if isinstance(output, list) and all('url' in item and 'content' in item for item in output):
                    display_sites(output)
                    bot_response = "추천 블로그 목록을 보여드렸습니다."
                else:
                    bot_response = str(output)
                    with st.chat_message("assistant"):
                        st.markdown(bot_response)

            # 챗봇 응답을 세션 및 UI에 추가
            assistant_message = {
                "role": "assistant",
                "content": bot_response,
                "timestamp": datetime.now()
            }
            st.session_state.chat_history.append(assistant_message)
            current_session["messages"].append(assistant_message)

else:
    st.write("안녕하세요 ☺️ 제주도 여행 계획 중이신가요? \n\n왼쪽에서 채팅방을 선택하거나 새로 시작하세요.")