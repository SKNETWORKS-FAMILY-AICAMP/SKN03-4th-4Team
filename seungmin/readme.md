# PDF Q&A 챗봇 README

## 프로젝트 개요
이 프로젝트는 Streamlit을 기반으로 PDF 문서를 업로드하고, 사용자의 질문에 PDF 내용을 바탕으로 답변을 제공하는 PDF Q&A 챗봇 애플리케이션입니다. PDF 파일을 벡터 데이터베이스에 저장하고, 벡터 검색을 통해 관련 문서를 찾은 후 RAG (Retrieval-Augmented Generation) 체인을 사용해 사용자 질문에 대한 답변을 생성합니다.

## 주요 기능
1. **PDF 업로드 및 임시 저장**: 사용자가 업로드한 PDF 문서를 서버의 임시 폴더에 저장합니다.
2. **PDF 문서 파싱 및 Document 생성**: PyMuPDF 모듈을 이용해 PDF 문서를 Document 객체로 변환합니다.
3. **Document 분할 및 벡터 DB 저장**: Document를 더 작은 단위로 나눈 후, FAISS를 사용하여 벡터 데이터베이스에 저장합니다.
4. **사용자 질문 처리**: 사용자의 질문을 받아 벡터 검색으로 관련 문서를 찾고, RAG 체인을 통해 답변을 생성합니다.
5. **PDF 페이지 이미지 변환**: PDF 페이지를 이미지로 변환해 Streamlit 인터페이스에서 페이지별로 표시할 수 있습니다.
6. **관련 문서 및 페이지 표시**: 질문과 관련된 문서를 보여주며, 문서의 페이지를 선택하면 해당 페이지 이미지를 표시합니다.

## 사용된 기술 및 라이브러리
- **Streamlit**: 웹 애플리케이션을 간단히 만들 수 있는 Python 기반 라이브러리.
- **LangChain**: RAG 체인 구성과 문서 분할을 지원하는 오픈 소스 라이브러리.
- **FAISS**: 벡터 데이터베이스로, 빠른 유사도 검색을 가능하게 함.
- **PyMuPDF (fitz)**: PDF 파일을 파싱하고 페이지 이미지를 생성하기 위한 라이브러리.
- **dotenv**: 환경 변수 관리를 위한 라이브러리.
- **OpenAIEmbeddings 및 ChatOpenAI**: 문서 임베딩 및 답변 생성을 위한 모델.

## 설치 및 실행 방법
1. **필수 패키지 설치**:
   ```bash
   pip install streamlit langchain faiss-cpu pymupdf python-dotenv
   ```

2. **환경 변수 설정**:
   프로젝트 루트에 `.env` 파일을 생성하고 OpenAI API 키 등 필요한 환경 변수를 추가합니다.
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

3. **프로젝트 실행**:
   ```bash
   streamlit run main.py
   ```

## 코드 설명

### 1단계: PDF 문서를 벡터DB에 저장하는 함수들
- `save_uploadedfile()`: 업로드된 PDF 파일을 서버에 임시 저장.
- `pdf_to_documents()`: PyMuPDFLoader를 사용해 PDF를 Document 객체로 변환.
- `chunk_documents()`: Document를 작은 조각으로 나누어 저장.
- `save_to_vector_store()`: OpenAI 임베딩 모델을 사용해 Document를 벡터로 변환하고 FAISS에 저장.

### 2단계: RAG 기능 구현과 관련된 함수들
- `process_question()`: 사용자 질문을 받아 관련 문서를 검색하고, RAG 체인으로 답변 생성.
- `get_rag_chain()`: 사용자 질문과 컨텍스트를 기반으로 답변을 생성하는 RAG 체인 구성.

### 3단계: PDF 페이지 이미지 변환 및 표시
- `convert_pdf_to_images()`: PyMuPDF를 사용해 PDF 페이지를 이미지로 변환.
- `display_pdf_page()`: 변환된 이미지 파일을 Streamlit 인터페이스에 표시.
- `natural_sort_key()`: 파일 이름을 자연스럽게 정렬.

### 메인 함수
- `main()`: Streamlit 인터페이스를 구성하며, PDF 업로드, 질문 처리, 관련 문서 표시 및 PDF 페이지 이미지를 보여주는 기능을 포함.

## 사용법
1. **PDF 파일 업로드**: 인터페이스에서 PDF 파일을 업로드한 후 'PDF 업로드하기' 버튼을 클릭.
2. **질문 입력**: PDF 문서에 대해 궁금한 내용을 입력하면 답변과 관련 문서를 확인할 수 있음.
3. **관련 페이지 보기**: 관련 문서의 페이지를 선택하여 페이지 이미지를 확인.

## 주의사항
- PDF 파일은 업로드 시 임시 폴더에 저장되며, 사용자 질문에 따라 벡터 검색이 이루어집니다.
- 벡터 검색 및 답변 생성에는 OpenAI API 키가 필요합니다.
- 이미지 변환 시 DPI 설정을 통해 이미지 품질을 조절할 수 있습니다.

## 향후 개선 사항
- PDF 문서 외에 다양한 문서 포맷 지원.
- 답변의 정확도 및 성능 개선을 위한 모델 최적화.
- 다중 PDF 문서의 통합 검색 기능 추가.
