import os
from dotenv import load_dotenv
# from google.colab import drive

# drive.mount('/content/drive')

try:
    import google.colab
    print("Colab 환경에서 실행 중입니다.")
    PATH = '/content/drive/MyDrive/data/'
    env_path = PATH + "env/.env"
except ImportError:
    print("로컬 환경에서 실행 중입니다.")
    import platform
    os_name = platform.system()
    if os_name == "Windows":
        print("Windows 로컬 환경에서 실행 중입니다.")
        PATH = './'
        env_path = PATH + "env/.env"
    elif os_name == "Linux":
        print("Linux 환경에서 실행 중입니다. (Colab일 가능성 있음)")
    else:
        print(f"운영 체제: {os_name}")

load_dotenv(dotenv_path=env_path)

# getenv로 환경 변수 가져오기
# KEY 쓸 때, 띄워쓰기 하면 안됨...
# env에 있는 모든 워드들은 띄워쓰기하면 못 알아봄
api_key = os.getenv('MY_OWN_KEY')

if not api_key:
    raise ValueError(".env 파일에서 API 키를 로드하지 못했습니다.")


# 가져온 값을 environ에 저장
os.environ['OPENAI_API_KEY'] = api_key

from langchain_memory_common.constant import CHATBOT_ROLE, CHATBOT_MESSAGE
from langchain_memory_common.prompt import create_message
from langchain_memory_common.chat import response_from_langchain, response_from_runnable_lambda, response_from_runnable_parallel, response_from_langgraph
import streamlit as st
from streamlit_extras.switch_page_button import switch_page

# web ui/ux
import streamlit as st

st.title("Chat Bot Memory Exam")


page = st.sidebar.selectbox(
    "Select a page",
    ["streamlit_main_page", "langchain_agent", "langchain_chatbot", "langchain_langgraph", "langchain_memory_exam", "langchain_final_mini", "langchain_vectordb"],  # "Select a page" 옵션 제거
    index=4  # langchain_agent를 기본 선택값으로 설정
)

# 페이지 이동
if page == "home":
    switch_page("streamlit_main_page")
elif page == "langchain_agent":
    switch_page("langchain_agent")
elif page == "langchain_chatbot":
    switch_page("langchain_chatbot")
elif page == "langchain_langgraph":
    switch_page("langchain_langgraph")
elif page == "langchain_final_mini":
    switch_page("langchain_final_mini")
elif page == "langchain_vectordb":
    switch_page("langchain_vectordb_page")

# ... existing code ...

# 메세지를 저장 
# messages = {"role":"", "content":""}
#   role -> user(사용자) / assistant(AI)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 저장한 메세지를 화면에 표현 
for message in st.session_state.messages:
    if message[CHATBOT_MESSAGE.role.name] in CHATBOT_ROLE.__members__:
        with st.chat_message(message[CHATBOT_MESSAGE.role.name]):
            st.markdown(message[CHATBOT_MESSAGE.content.name])

# 사용자 입력
prompt = st.chat_input("입력해주세요")
# 사용자 입력이 있다면,
if prompt:
    message = create_message(role=CHATBOT_ROLE.user, prompt=prompt)
    
    if message:
        # 화면에 표현
        with st.chat_message(CHATBOT_ROLE.user.name):
            st.write(prompt)
        # st.session_state.messages.append({"role" : "user", "content": prompt})
        # 챗봇 답변 
        with st.chat_message(CHATBOT_ROLE.assistant.name):
            # assistant_response = response_from_llm(prompt)
            # st.markdown(assistant_response)
            assistant_response = st.write(response_from_langgraph(prompt=prompt, message_history=st.session_state.messages))



# 버튼으로 메인 페이지로 돌아가기
if st.button("Back to Main"):
    switch_page("streamlit_main_page")
