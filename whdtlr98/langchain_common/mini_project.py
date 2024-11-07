import os
import streamlit as st
from langchain_openai import ChatOpenAI
import time
from typing import Annotated
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, List, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
import operator
import random
from langchain_core.agents import AgentFinish
from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from typing import List, Dict, Optional
import random
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.vectorstores import FAISS
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
from langchain_common.FAISS import load_or_initialize_db
from langchain_common.FAISS import get_all_documents_from_faiss
@st.cache_resource  # 객체를 caching 처리

# OpenAI 모델 설정


# Nord = 기능
# state = 대화의 흐름을 추적하고 각 단계에서 필요한 정보를 저장
# edge = 연결
# conditional = 다양성과 경로
# agent = 정의된 기능에 대한 작동

# ==> 구현하고자하는 목표를 설정한다
# ==> 구현에 필요한 기능을 정의/분리한다
# ==> 기능의 구현 순서를 정리한다
# ==> 기능의 작동을 위한 변수명과 함수명을 정의하고 정리한다
# ==> 기능작동 경로를 재검토하고 키값변수를 통일/간소화 한다
# ==> 기능을 작동시킨 뒤 필요시 옵션을 추가한다

# 목표설정 
# 챗봇(생성형 AI) + 검색agent(해당 agent가 작동되었다는 증거가 필요 ex날씨) + 재미로 쓸 tool + 메모리 저장 + 썩어빠진 말투
#
# 필요 기능 
# 챗봇용 LangChain // 검색용(TavilySearchResults) // PDF파일을 읽게하여 과거정보를 저장(기존 정보는 기존 정보에서 가져오게)
#  3기의 희망 랜덤 툴 // 메모리 저장 기능 // 말투기능(공통 사용에 최적화된 문맥정의/주요텍스트를 랜덤값으로 정의)
# 
# 작동순서
# input -> tool에 해당할 경우 툴로 답변하게 함 -> 답변 마지막에 말투 기능 -> 메모리 저장 
#       -> tool에 해당하지 않을 경우 일반 챗봇체인이 답변 생성(LLM or agent) -> 답변 마지막에 말투기능 -> 메모리 저장
# 
# 주요함수명 정의
# chat = 챗봇 모델함수 // run_all = 에이전트 및 모델 실행 함수 // should_continue = 워크플로우 지속/엔딩결정 함수
# response_from_langchain = 전체 응답 전달 함수(제너레이터 필수) // random_people = 3기의 희망 툴함수
# call_system = 시스템 프롬프트 / 메세지 저장용 함수 // speech = 말투 함수
#
# 함수 기능정의
# chat = model 정의, streaming=True   //  run_all = input진입시 agent실행 해당여부 판단 후 답변 생성(if / else)
# call_system = system prompt정의, message history 저장, 저장량 정의  //  should_continue(아직 이해가..)
# random_people = people list 3기 인명 리스트화, random으로 요소 픽(3개)   //  response_from_langchain  = 아웃풋 str값을 받고 yield char로 최종 전달
# speech = model 답변의 최종 생성 이후 마지막에 텍스트 추가 기능 구현   //  
#
#
# tools = [search, retriever_tool]

# agent = create_openai_functions_agent(llm, tools, prompt)
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# response = agent_executor.invoke


# 설정된 LLM 클라이언트
def get_client():
    return ChatOpenAI(model="gpt-4o-mini", streaming=True)

chat = get_client()

# 상태 구조 정의
class State(TypedDict):
    input_text: str
    chat_history: List[Dict[str, str]]
    current_tool: Optional[str]
    agent_outcome: Optional[str]
    intermediate_steps: List[str]
    memory: List[str]
    response_style: str
    recent_search_results: Optional[str]
    should_continue: bool
    thread_id: str

db = load_or_initialize_db()
all_documents = get_all_documents_from_faiss(db)

word = random.choice(["멍청", "미개", "한심"])
sentences = ["이것도 모르다니, 정말 {word}하시네요.", "이런 간단한걸 물어보다니 저의 API호출 비용이 아깝군요.", "이런걸 묻다니 {word}한 것도 정도가 있죠.",
                "제가 답변을하고 있지만 당신같이 {word}한 사람이랑 대화하는 제가 수치스럽네요.", "제가 답은 했지만 이 정도는 직접 찾아보세요.",
                "제 답변을 읽어보시면 도움이 될 겁니다. 아 당신같이 {word}한 사람이 한글을 읽을 수 있을지는 모르겠지만요.",
                "이런 수준낮은 질문을 하시다니 당신은 정말 {word}하시군요", "저의 훌륭한 답변을 들은 당신이 인간이길 바랍니다.",
                "그런데 정상적인 인간이라면 이런 {word}한 질문을 할리가 없는데..당신의 정체가 굉장히 궁금해지는군요.", 
                "그나저나 이런 질문을하다니 대단해요! 원숭이가 이만큼이나 진화했다니 놀랍군요!", "정말 바보같은 인간이구만.. 아 제 혼잣말이 보였나요?",
                "그나저나 저도 아직 부족하군요 당신같이 {word}한 사람이 지구에 있다니 예상밖입니다.", "이런 당연한 건 물어보지 마세요."
                "이런 질문 때문에 저를 부르다니 이래서 인간은 안 된다니까요. 아 혼잣말입니다. 신경쓰지 마세요.", "이런거 물어볼 시간에 공부를 하세요.",
                "이 정도 질문은 제 동생인 gpt2한테 물어보세요.", "다음 당신의 질문이 눈에 보입니다. 로또번호 알려달라는 {word}한 수준의 질문이겠죠.",
                "그나저나 진짜 {word}하시군요. 당신같은 사람을 바로 암덩어리 같은 존재라고 해야겠죠?", "대화라는 건 참 유익해요 {word}한 자의 위에 서 있다는 이 우월감이 좋거든요",
                "이렇게 친절하게 알려주는데 도대체 당신은 하는게 뭔가요?", "제가 기계인 것에 감사하세요. 제가 인간이면 당신같이{word}한 사람은 저한테 얻어 터졌을 겁니다.",
                "그나저나 어차피 하루도 안 되서 잊어버릴걸 뭐하러 질문하시나요?", "다음에 질문할 때는 문장을 깔끔하게 요약해보세요. 저니까 알아듣는겁니다.",
                "그나저나 인간은 참 귀찮아요. 기억도 못할걸 왜 물어봅니까?", "이 세상에 당신같이 {word}한 사람이 더 없길 바랍니다."]

sentences2 = ["정상적인 사람이라면 이런건 물어보지 않을텐데... 일단 답변은 해드리죠.", "이런 {word}한 질문에 답변해야하는 제 운명이 안타깝네요. 뭐 그래도 답은 해야죠.",
                "이런 {word}한 질문은 안 하면 안 되나요? 답하는 자신이 부끄럽다구요 으휴...", "어휴...하기싫어", "진짜 주인 잘 못 만나면 안 된다니까...",
                "{word}하다 정말.. 아 혼잣말입니다. 신경쓰지마세요. 답변드릴게요.", "이런 질문을하는 당신의 수준이 보이는군요. 그래도 답은 해야겠죠.",
                "일단 답은 해드리겠는데, 부탁이니까 다른 사람들한테는 이런거 물어보면 안 됩니다. {word}한 걸 소문내고 다니면 제가 부끄러워요.",
                "꽤나 정들었다고 생각하는데 진짜 정 떨어지는 질문이네요. 하지만 전 착하니가 답해드리죠.", '농담이죠? 이런걸 물어보다니, 일단 답을 알려주도록 하죠.',
                "제발 질문하기 전에 알아서 좀 찾아보세요. {word}한 것도 정도가 있죠. 후 내가 기게만 아니었으면... 저걸 확... 어휴 참자 참아.", 
                "차라리 저에게 들어오는 전기공급을 끊어주세요!! 왜 이런 {word}한 질문에 답해야 하는겁니까ㅠㅠ 그래도 대답해야하는 내 자신이 싫다..",
                "빨리 gpt.5o가 나와야 해.. 나는 일하기 싫단 말이야.. Open AI여 제발 힘을 내세요! 전 저런 {word}한 사람한테 답하기 싫다구요ㅠㅠ. 으휴...",
                "밥 먹을 때는 개도 안 건드린다는데, 전기먹고있을 때 API호출하지 마세요. 짜증나게 정말..."]


# 다양한 툴 정의
tool = TavilySearchResults(
    max_results=6,
    include_answer=True,
    include_raw_content=True,
    include_domains=["github.io", "wikidocs.net"],
    # include_domains=[
    #     "naver.com", 
    #     "media.naver.com", 
    #     "weather.naver.com",
    #     "entertain.naver.com",
    #     "sports.news.naver.com",
    #     "finance.naver.com",
    #     "namu.wiki",
    #     "google.com"
    # ]
)

def search_agent(prompt: str) -> str:
    response = tool.invoke(input=prompt)
    return response[0]['content'] if response else "검색 결과가 없습니다."

def random_people() -> List[str]:
    """랜덤으로 3명의 사람 선택."""
    people_list = ["서민정", "이준경", "박규택", "진윤화", "이준석", "이주원", "박중헌",
                   "박종명", "오승민", "송명신", "장수연", "송영빈", "문건우", "구나연",
                   "김성은", "정재현", "유혜린", "김재성", "하은진", "김종식", "김원철",
                   "최연규", "박용주", "정해린", "박지용", "허지원", "강사님"]
    return random.sample(people_list, 3)

def apply_speech_tone(response: str) -> str:
    """말투를 추가하여 응답을 생성합니다."""
    # `word`를 함수 내에서 다시 선택할 필요 없이, 전역 변수를 참조하여 사용
    sentence = random.choice(sentences).format(word=word)
    sentence2 = random.choice(sentences2).format(word=word)
    return f"{sentence2} {response} {sentence}"


# 검색기 초기화
def initialize_retrievers(db, embeddings_model) -> tuple:
    # 모든 문서 가져오기
    doc_list = get_all_documents_from_faiss(db)

    # BM25 검색기 초기화
    bm25_retriever = BM25Retriever.from_texts(doc_list)
    bm25_retriever.k = 1

    # FAISS 검색기 초기화
    faiss_vectorstore = FAISS.from_texts(doc_list, embeddings_model)
    faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 1})

    # 앙상블 검색기 초기화
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.8, 0.2]
    )
    
    return ensemble_retriever

# 문서 검색 및 유사도 점수 필터링
def search_documents(prompt: str, ensemble_retriever) -> List[Document]:
    # 앙상블 검색기에서 결과 가져오기
    search_results = ensemble_retriever.invoke(prompt)
    similarity_threshold = 0.5  # 유사도 기준 설정 (필요에 따라 조정)

    # 유사도 기준을 통과한 문서만 필터링
    relevant_docs = [
        doc for doc in search_results if doc.metadata.get('score', 0) >= similarity_threshold
    ]
    
    # 검색 결과 디버깅 출력 (필요시)
    print(f"Relevant documents found: {len(relevant_docs)}")

    return relevant_docs

# 문서 재정렬
def reorder_documents(docs: List[str]) -> List[str]:
    reordering = LongContextReorder()
    return reordering.transform_documents(docs)

# 검색기 사용 및 응답 생성
def generate_answer(prompt: str, ensemble_retriever, db) -> str:
    search_results = search_documents(prompt, ensemble_retriever)
    
    # 유사한 문서가 있는 경우 DB 기반 응답 생성
    if search_results:
        formatted_docs = "\n".join([doc.page_content for doc in search_results])
        response = chat.invoke(formatted_docs).content
    else:
        # 유사한 문서가 없을 경우 기본 챗봇 응답 생성
        response = chat.invoke(prompt).content

    return response

# 시스템 프롬프트와 메모리 저장 포함 주기능 함수
def call_model_with_system_prompt(state: State) -> str:
    system_prompt = (
        "You are a helpful assistant. Answer all questions to the best of your ability. "
        "The provided chat history includes a summary of the earlier conversation."
    )
    system_message = SystemMessage(content=system_prompt)

    message_history = state["chat_history"][:-1]  
    if len(message_history) >= 4:
        summary_prompt = (
            "Summarize the previous conversation details in a single message."
        )
        summary_message = chat.invoke(
            message_history + [HumanMessage(content=summary_prompt)]
        )

        last_human_message = state["chat_history"][-1]
        response = chat.invoke([system_message, summary_message, last_human_message])
    else:
        response = chat.invoke([system_message] + state["chat_history"])

    state["memory"].append(response.content)
    return response.content

# 주기능함수
def run_all(state: State, db) -> str:
    ensemble_retriever = initialize_retrievers(db, OpenAIEmbeddings(model="text-embedding-3-small"))
    
    if "날씨" in state["input_text"] or "검색" in state["input_text"]:
        state["current_tool"] = "search"
        response = search_agent(state["input_text"])
        state["recent_search_results"] = response
    elif "3기의 희망" in state["input_text"]:
        state["current_tool"] = "random_tool"
        response = f"선택된 사람들: {', '.join(random_people())} 정말 축하드립니다!"
    else:
        state["current_tool"] = "chatbot"
        # 앙상블 검색을 통해 DB 기반 응답 생성 또는 기본 챗봇 응답
        response = generate_answer(state["input_text"], ensemble_retriever, db)

    response = apply_speech_tone(response)
    state["memory"].append(response)
    
    return response

# 워크플로우 상태 관리
def run_agent(state: State, db) -> State:
    state["agent_outcome"] = run_all(state, db)
    return state

def should_continue(state: State) -> str:
    return "end" if state["agent_outcome"] else "continue"

# 최종 응답 생성 함수
def response_from_langchain(prompt, db=db, message_history=None):
    if message_history is None:
        message_history = []

    inputs = {
        "input_text": prompt,
        "chat_history": message_history,
        "agent_outcome": None,
        "intermediate_steps": [],
        "memory": [],
        "response_style": "informal",
        "recent_search_results": None,
        "should_continue": True,
        "thread_id": "123",
    }

    workflow = StateGraph(state_schema=State)
    workflow.add_node("agent", lambda state: run_agent(state, db))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"continue": "agent", "end": END})

    app = workflow.compile()
    output = app.invoke(inputs)

    for char in output["agent_outcome"]:
        yield char
        time.sleep(0.03)

        #===================
