from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
load_dotenv()

# 제목
st.title("ChatPDF")
st.write("---")

# 파일 업로드
uploaded_file = st.file_uploader("Choose a file")
st.write("---")

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
