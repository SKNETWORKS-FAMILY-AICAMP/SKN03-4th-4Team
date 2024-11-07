import streamlit as st
import tiktoken
from loguru import logger
from dotenv import load_dotenv
import os
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.memory import StreamlitChatMessageHistory
from langchain.embeddings import OpenAIEmbeddings

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

###################################################################################
def main():
    st.set_page_config(page_title="(╭☞• ⍛• )╭☞♥︎", page_icon="🎄")
    st.title("(ง ͡° ͜ʖ ͡°)ง 논문 쉽게 접근하기 !! ")

    # 대화 흐름 관리를 위한 세션 상태 초기화
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None


    # 사이드바 파일 업로드와 처리 버튼
    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your PDF file", type=['pdf'], accept_multiple_files=True)
        process = st.button("Process")

    # 파일 텍스트 추출 및 대화 체인 초기화
    if process:
        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vetorestore = get_vectorstore(text_chunks)

        st.session_state.conversation = get_conversation_chain(vetorestore, openai_api_key)
        st.session_state.processComplete = True

    # 초기 화면 메시지 설정
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant",
                                         "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]
    # 기존 메시지 출력
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 채팅 입력 및 사용자 질문 처리
    history = StreamlitChatMessageHistory(key="chat_messages")


    # 질문 입력 및 챗봇 응답
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        # 챗봇 응답 생성 및 참고 문서 표시
        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("Thinking..."):
                result = chain({"question": query}) # 응답 생성
                st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    for doc in source_documents[:3]:  # Display the top 3 sources
                        st.markdown(doc.metadata['source'], help=doc.page_content)

        # 응답을 대화 히스토리에 추가
        st.session_state.messages.append({"role": "assistant", "content": response})


####################################################################################
# 텍스트의 토큰 수 계산
def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

# pdf파일에서 텍스트 추출
def get_text(docs):
    doc_list = []

    for doc in docs:
        file_name = doc.name
        with open(file_name, "wb") as file:
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")

        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
        
        documents = loader.load_and_split()
        doc_list.extend(documents)
    return doc_list

# 텍스트를 청크로 나누기
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks

# 벡터 스토어 생성
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

# 대화 체인 생성
def get_conversation_chain(vetorestore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo', temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vetorestore.as_retriever(search_type='mmr', verbose=True),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )
    return conversation_chain

# 메인 함수 실행
if __name__ == '__main__':
    main()
