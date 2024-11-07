from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
import streamlit as st
import os
import tempfile
from model import pdf_to_document
# Load .env 
from dotenv import load_dotenv
load_dotenv()

# 제목
st.title("ChatPDF")
st.write("---")
# 파일 업로드
uploaded_file = st.file_uploader("Choose a file")
st.write("---")

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



