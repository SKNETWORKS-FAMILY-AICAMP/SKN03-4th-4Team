# RAG = 질문 인입 -> 임베딩 -> 벡터db뒤져서 유사성 검사 후 유사답변 추출 -> LLM에 넘김
# 벡터 db에 넣을 요소가 필요(원천데이터 폴더) => 원천데이터에서 로더기능으로 벡터 디비에 저장
# 임베딩 모델선정, 스플릿 청크 단위설정
# 유사성검사 후 유사성 더러울 경우 어떻게할지 결정필요
# 추출답변을 자연스럽게 바꾸는 로직필요(답변 확인 후 결정)
# 
# 로더설정 => 파일로드 => 임베딩, 스플릿 => FAISS 벡터 디비에 저장
# 보고서 작성법
# 로드과정에서 목차, 스타트페이지 버리고 로드 후 벡터 디비에 저장
# 불러오기 실험 // 

#pdf loader
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.document_transformers import LongContextReorder
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
import os
from glob import glob
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader

# PDF 파일이 위치한 폴더 경로

import os

# 전역 변수로 경로 설정
DATA_PATH = "C:/dev/langchain_chatbot (2)/pdf_folder/보고서 작성 메뉴얼-청와대 비서실.pdf"
DB_SAVE_FOLDER = "faiss_db"
DB_INDEX_NAME = "2009 report_manual_bluehouse_300chunk"  # 영문자로 변경

_db_instance = None

def load_or_initialize_db():
    """벡터 DB를 불러오거나, 없을 시 생성하여 데이터를 추가하고 저장"""
    global _db_instance
    
    # 임베딩 모델 초기화
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # DB 폴더가 존재하지 않으면 생성
    if not os.path.exists(DB_SAVE_FOLDER):
        os.makedirs(DB_SAVE_FOLDER)
    
    # FAISS 인덱스 파일(.faiss)과 메타데이터 파일(.pkl)이 모두 존재하는지 확인
    faiss_file = f"{DB_SAVE_FOLDER}/{DB_INDEX_NAME}.faiss"
    pkl_file = f"{DB_SAVE_FOLDER}/{DB_INDEX_NAME}.pkl"
    if os.path.exists(faiss_file) and os.path.exists(pkl_file):
        print("기존 DB 파일을 로드합니다.")
        _db_instance = FAISS.load_local(
            folder_path=DB_SAVE_FOLDER,
            index_name=DB_INDEX_NAME,
            embeddings=embeddings,
            allow_dangerous_deserialization=True  # 역직렬화 허용
        )
        return _db_instance
    # DB 파일이 없으므로 새로 생성하여 데이터 추가
    print("새로운 DB를 생성하고 데이터를 추가합니다.")
    _db_instance = initialize_faiss_db(embeddings)
    
    # 생성된 DB 반환
    return _db_instance

def get_all_documents_from_faiss(db):
    documents = []
    for doc_id in db.docstore._dict:
        documents.append(db.docstore._dict[doc_id].page_content)
    return documents

def initialize_faiss_db(embeddings):
    """새로운 벡터 DB를 생성하고 데이터를 추가한 후 저장합니다."""
    loader = PDFPlumberLoader(DATA_PATH)
    docs = loader.load()
    docs = docs[3:]

    # 스플리터 설정 및 청크 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        is_separator_regex=False,
    )
    split_doc1 = text_splitter.split_documents(docs)

    # 임베딩 및 차원 계산
    dimension_size = len(embeddings.embed_query(split_doc1[0].page_content))

    # FAISS 벡터 저장소 생성
    db = FAISS(
        embedding_function=embeddings,
        index=faiss.IndexFlatL2(dimension_size),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    # 데이터 추가
    db.add_documents(split_doc1)

    # DB 저장
    db.save_local(folder_path=DB_SAVE_FOLDER, index_name=DB_INDEX_NAME)
    
    return db

load_or_initialize_db()

# def faiss_db():
#     DATA_PATH = "C:/dev/langchain_chatbot (2)/pdf_folder/보고서 작성 메뉴얼-청와대 비서실.pdf"
#     # pdf 로드 후 docs 변수에 할당
#     loader = PDFPlumberLoader(DATA_PATH)
#     docs = loader.load()
#     docs = docs[3:]

#     #splitter
#     # 스플리터 설정 정의
#     text_splitter = RecursiveCharacterTextSplitter(
#         # 청크 크기를 매우 작게 설정합니다. 예시를 위한 설정입니다.
#         chunk_size=600,
#         # 청크 간의 중복되는 문자 수를 설정합니다.
#         chunk_overlap=50,
#         # 구분자로 정규식을 사용할지 여부를 설정합니다.
#         is_separator_regex=False,
#     )

#     # pdf를 설정에 맞게 split한 후 split_doc1에 할당
#     split_doc1 = text_splitter.split_documents(docs)

#     #==> pdf를 읽어온 뒤 일정 청크 단위로 잘라내기 과정 완료

#     #embedding
#     embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

#     #차원계산
#     # 문서 청크 중 첫 번째 청크를 사용하여 차원 계산
#     dimension_size = len(embeddings.embed_query(split_doc1[0].page_content))
#     #==> 질문을 임베딩 벡터화


#     # FAISS 벡터 저장소 생성 // 생 아무것도 없는 저장소 생성
#     # db에 데이터가 있는지 확인
#     if not db or len(db.index_to_docstore_id) == 0:
#     # 데이터가 없으면 db 생성
#         db = FAISS(
#             embedding_function=embeddings,
#             index=faiss.IndexFlatL2(dimension_size),
#             docstore=InMemoryDocstore(),
#             index_to_docstore_id={},
#         )

#     #데이터추가

#     # db에 pdf파일 집어넣으면 임베딩 되고 잡다하게 처리 다 되서 벡터 디비에 들어감
#     db.add_documents(split_doc1)

#     return db
    #===============================================================================

# from typing import List

# def initialize_retrievers(db, embeddings_model) -> tuple:
#     # 벡터 DB에서 문서 가져오기
#     doc_list = [doc.page_content for doc in db.get_all_documents()]  # `get_all_documents`는 벡터 DB의 문서 추출 메서드로 가정
    
#     # BM25 검색기 초기화
#     bm25_retriever = BM25Retriever.from_texts(doc_list)
#     bm25_retriever.k = 1  # 검색 결과 개수 설정

#     # FAISS 검색기 초기화
#     faiss_vectorstore = FAISS.from_texts(doc_list, embeddings_model)
#     faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 1})

#     # 앙상블 검색기 초기화
#     ensemble_retriever = EnsembleRetriever(
#         retrievers=[bm25_retriever, faiss_retriever],
#         weights=[0.3, 0.7]
#     )
    
#     return bm25_retriever, faiss_retriever, ensemble_retriever

# def search_documents(prompt: str, ensemble_retriever) -> List[str]:
#     # 앙상블 검색 수행
#     ensemble_result = ensemble_retriever.invoke(prompt)
#     return ensemble_result

# def reorder_documents(docs: List[str]) -> List[str]:
#     # 문서 재정렬
#     reordering = LongContextReorder()
#     reordered_docs = reordering.transform_documents(docs)
#     return reordered_docs

# def format_docs(docs: List[str]) -> str:
#     # 재정렬된 문서를 형식화하여 반환
#     return "\n".join([doc.page_content for doc in docs])

# def generate_answer(prompt: str, reordered_docs: List[str], model_name="gpt-4o-mini") -> str:
#     # 답변 생성 체인 정의
#     chain = (
#         {
#             "context": itemgetter("question")
#             | ensemble_retriever
#             | RunnableLambda(reorder_documents),  # 질문을 기반으로 문맥 검색
#             "question": itemgetter("question"),
#             "language": itemgetter("language"),
#         }
#         | prompt  # 프롬프트 템플릿에 값을 전달
#         | ChatOpenAI(model=model_name)  # 언어 모델에 전달
#         | StrOutputParser()  # 문자열로 파싱
#     )
#     # 답변 생성
#     return chain.invoke(prompt)

# def perform_query_answer_pipeline(prompt: str, db) -> str:
#     # 임베딩 모델 초기화
#     embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

#     # 1. 검색기 초기화
#     bm25_retriever, faiss_retriever, ensemble_retriever = initialize_retrievers(db, embeddings)

#     # 2. 문서 검색 수행
#     search_results = search_documents(prompt, ensemble_retriever)

#     # 3. 문서 재정렬
#     reordered_docs = reorder_documents(search_results)

#     # 4. 답변 생성
#     answer = generate_answer(prompt, reordered_docs)
#     return answer

# # 사용 예시
# prompt = "my favorite fruit is apple"
# db = faiss_db()  # 벡터 DB 인스턴스를 설정
# answer = perform_query_answer_pipeline(prompt, db)
# print(answer)