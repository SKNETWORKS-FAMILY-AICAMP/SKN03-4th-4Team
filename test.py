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

#Loader
loader = PyPDFLoader("book.pdf")
pages = loader.load_and_split()


#Split
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 300,
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
question = "아내가 먹고 싶어하는 음식은 무엇이야?"
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever())
result = qa_chain.invoke({"query": question})
print(result)