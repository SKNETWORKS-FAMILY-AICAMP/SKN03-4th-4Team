<<<<<<< HEAD
=======
<<<<<<< HEAD
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
=======
>>>>>>> dev
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
<<<<<<< HEAD
from chromadb.config import Settings
=======
>>>>>>> dev
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
import streamlit as st
import os
import tempfile
<<<<<<< HEAD
=======
from model import pdf_to_document
# Load .env 
>>>>>>> dev
from dotenv import load_dotenv
load_dotenv()

# 제목
st.title("ChatPDF")
st.write("---")
<<<<<<< HEAD

=======
>>>>>>> dev
# 파일 업로드
uploaded_file = st.file_uploader("Choose a file")
st.write("---")

<<<<<<< HEAD
# PDF 파일을 문서로 변환
def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

# 파일 업로드가 되었을 때 실행되는 코드
if uploaded_file is not None:
    # 페이지 로드
    pages = pdf_to_document(uploaded_file)
    st.write(f"페이지 수: {len(pages)} 페이지가 로드되었습니다.")

    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.split_documents(pages)
    st.write(f"총 {len(texts)} 개의 텍스트 청크로 분할되었습니다.")

    # 임베딩 모델 초기화 및 임베딩 생성
    embeddings_model = OpenAIEmbeddings()

    # 벡터 저장소에 임베딩된 텍스트 저장
    # db = Chroma.from_documents(texts, embeddings_model)

    # Chroma를 메모리에만 저장하도록 설정
    db = Chroma.from_documents(
        documents=texts,
        embedding=embeddings_model,
        persist_directory=".chroma_db"
        
    )
    # 질문 입력
    st.header("PDF에게 질문해보세요!!")
    question = st.text_input('질문을 입력하세요')

    # # MultiQueryRetriever 추가
    # retriever = MultiQueryRetriever(
    #     retriever=db.as_retriever(),
    #     llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)  # 온도를 약간 높임
    # )

    retriever = MultiQueryRetriever.from_llm(
    retriever=db.as_retriever(),
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
    )

    # 질문 버튼
    if st.button('질문하기'):
        with st.spinner('질문에 대한 답변을 찾고 있습니다...'):
            try:
                llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
                qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, verbose=True)
                print(qa_chain)
                result = qa_chain.invoke({"query": question})
                print(result)

                # 결과 출력
                st.write(result.get("result", "답변을 찾지 못했습니다. PDF의 내용을 확인해 주세요."))
            except Exception as e:
                st.write("오류가 발생했습니다. PDF 내용 및 설정을 확인해 주세요.")
                st.write(f"오류 내용: {e}")
=======
#업로드 되면 동작하는 코드
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

    #Split
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 1000,
        chunk_overlap  = 20,
        length_function = len,
        is_separator_regex = False,
    )
    texts = text_splitter.split_documents(pages)

    #Embedding
    embeddings_model = OpenAIEmbeddings()

    # load it into Chroma
    db = Chroma.from_documents(texts, embeddings_model)

    #Question
    st.header("PDF에게 질문해보세요!!")
    question = st.text_input('질문을 입력하세요')

    if st.button('질문하기'):
        with st.spinner('Wait for it...'):
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever(), return_source_documents=True)
            result = qa_chain.invoke({"query": question})
            print(result['source_documents'])
            st.write(result["result"])



>>>>>>> a762fa9f9050eb150ae4f02cee0766e0b7025e1c
>>>>>>> dev
