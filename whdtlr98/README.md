# LLM을 연동한 내외부 문서 기반 질의응답 시스템

## 목차

- [소개](#소개)
- [기능](#기능)
- [기술 스택](#기술-스택)
- [설치](#설치)
  - [사전 요구 사항](#사전-요구-사항)
  - [로컬 설치](#로컬-설치)
- [사용법](#사용법)
  - [문서 추가](#문서-추가)
  - [서버 실행](#서버-실행)
- [예시](#예시)
  - [Streamlit 사용 예시](#streamlit-사용-예시)
  - [API 사용 예시](#api-사용-예시)
  - [Python 코드에서 사용](#python-코드에서-사용)

## 소개

기존 **LangChain** 에이전트를 활용하여 데이터베이스의 내부문서에 대한 질의응답을 처리하는 챗봇개발. 또한 사람과 유사한 말투를 적용하여 사용자로 하여금 즐거운 경험을 제공함을 목표로 함  

## 기능

- **문서 인덱싱**: 다양한 형식의 문서를 파싱하고 인덱싱.
- **질의응답**: 사용자의 질문에 대해 문서 내에서 답변을 찾아 제공.
- **자연어 이해**: LLM을 활용하여 자연어로 표현된 질문을 이해.
- **확장성**: 새로운 문서나 데이터 소스의 추가가능.
- **RAG (Retrieval-Augmented Generation)**: LLM이 답변할 수 없는 질문에 대해 검색을 통한 답변을 제공.
- **검색**: 실시간 이슈 파악을 위한 데이터 검색 툴 추가
- **말투**: 사용자의 특수한 경험을 위한 말투를 추가(ex 짜증)

## 기술 스택

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3.7-orange)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2.44-brightgreen)
![OpenAI GPT-4](https://img.shields.io/badge/OpenAI-GPT--4-blueviolet?logo=openai&logoColor=white)
![OpenAI GPT-4](https://img.shields.io/badge/OpenAI-GPT--4--mini-blueviolet?logo=openai&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.39.0-red?logo=streamlit&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Models-yellow?logo=huggingface&logoColor=white)

**프로그래밍 언어**: Python 3.12+

**LLM 관련 라이브러리**:

- **LangChain**: LLM을 효과적으로 활용하기 위한 체인과 에이전트 구성
- **LangGraph**: 복잡한 대화 흐름과 그래프 기반의 대화 모델링 지원

**LLM 모델**: OpenAI GPT-4

**검색모델**: TavilySearchResults

**기타 라이브러리**:

- **문서 파싱**: `PDFPlumberLoader`
- **토큰화**: HuggingFace 모델 및 토크나이저
- **vector db**: FAISS

### 구조도
![alt text](image-1.png)

### Streamlit 사용 예시

Streamlit 웹 애플리케이션에서 질문을 입력하여 답변을 받을 수 있습니다.

![alt text](image.png)

### API 사용 예시
TavilySearchResults API를 기반으로 검색을 수행합니다.

![alt text](image-2.png)

