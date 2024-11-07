import streamlit as st
from streamlit_extras.switch_page_button import switch_page

st.title("Main Application Page")
st.markdown("""LangChain Agent는 기본 LLM 기반의 모델에 검색엔진을 적용하였습니다. 
            따라서 원하는 답변을 받을 확률이 높으며 최신 정보도 얻을 수 있습니다.
            
            """)
            
st.markdown("""LangChain Chatbot은 기본 LLM모델 기반에 여행관련 자료에 특화된 모델입니다.
            여행과 관련된 답변에 전문화 되어 있으므로 여행관련 질문은 해당 모델을 사용하시면 됩니다.""")


# 버튼 컨테이너를 생성하여 가로로 배치
col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)

with col1:
    if st.button("LangChain Agent"):
        switch_page("langchain_agent")

with col2:
    if st.button("LangChain Chatbot"):
        switch_page("langchain_chatbot")

with col3:
    if st.button("LangChain LangGraph"):
        switch_page("langchain_langgraph")

with col4:
    if st.button("Langchain memory exam"):
        switch_page("langchain_memory_exam")

with col5:
    if st.button("langchain final mini"):
        switch_page("langchain_final_mini")
with col6:
    if st.button("langchain vector db"):
        switch_page("langchain_vectordb_page")


page = st.sidebar.selectbox(
    "Select a page",
    ["streamlit_main_page", "langchain_agent", "langchain_chatbot", "langchain_langgraph", "langchain_memory_exam", "langchain_final_mini"],  # "Select a page" 옵션 제거
    index=0  # langchain_agent를 기본 선택값으로 설정
)
st.write("vectorDB 모델에 에러가 있음... 어떻게 해결해야 할지 감도 안옴")
# 페이지 이동
if page == "home":
    switch_page("streamlit_main_page")
elif page == "langchain_agent":
    switch_page("langchain_agent")
elif page == "langchain_langgraph":
    switch_page("langchain_langgraph")
elif page == "langchain_memory":
    switch_page("langchain_memory_exam")
elif page == "langchain_final_mini":
    switch_page("langchain_final_mini")

