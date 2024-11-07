# SKN03-4th-4Team

팀명 : DocQ

## 프로젝트명

LLM 기반 PDF 채팅 프로그램

### 프로젝트 개요

> LLM을 활용해 PDF 파일의 내용을 읽고, 사용자가 자연어로 질문을 하면 해당 PDF 내용에 기반한 답변을 제공하는 프로그램 개발

### 주요 기능 설명

- **파일 업로드 기능** : 사용자로부터 PDF 파일을 업로드받아 텍스트로 변환하는 기능.
- **문서 텍스트 분할 및 임베딩** : PDF 내용을 처리 가능한 작은 텍스트 조각으로 나눈 후 임베딩하여 검색 성능을 최적화.
- **질문 응답 기능** : 사용자가 입력한 질문에 대해 PDF 내용에서 관련 정보를 찾아 응답하는 기능.

### 전체 구조

![mini structure-1](https://github.com/user-attachments/assets/f0a5edc4-8f2e-42a3-9476-beb5f6bd9ad2)

### 구현

![image-6](https://github.com/user-attachments/assets/1f372c44-9ffd-4403-be0b-33a9ec2b930c)

### 에러리포트

문제점 1. streamlit을 적용하는 과정에서 답이 나오지 않는 문제

- streamlit 적용 전
  ![image-1](https://github.com/user-attachments/assets/84862c6b-e57d-4162-8414-0dab7deeaea8)

- streamlit 적용 후
  ![image-2](https://github.com/user-attachments/assets/07ebe394-fa07-49a5-85aa-5e40150783be)

- 문제점 확인

![image-3](https://github.com/user-attachments/assets/6f6ba621-8c04-45b2-b192-d608063a3a7b)

질문에 대한 내용이 포함되어 있지 않는 문제를 발견함

- 해결과정
  ![image](https://github.com/user-attachments/assets/6c22aecc-b9b1-4ded-b804-dd5dc3401f40)
  위의 코드를 사용하면 질문을 입력한 후에 반환된 결과에서 내용을 검토할 수 있음

      이를 통해 검색된 문서들이 질문과 적절한 연관성을 가지는지 평가할 수 있음

- 해결 방법

chunk_size 를 늘림

![image-4](https://github.com/user-attachments/assets/49720eeb-0144-44a6-bf70-eb1137eb1d85)

![image-5](https://github.com/user-attachments/assets/78a95747-9f8c-4bb3-8fc3-47bad1e00bae)
