from langchain.tools import tool
from typing import List, Dict
from langchain_community.tools.tavily_search import TavilySearchResults
import random

def recommend_airplane_site() -> str:
    """Recommends a list of popular airplane ticket booking sites, each on a new line."""
    sites = [
        "https://www.skyscanner.net",
        "https://www.kayak.com",
        "https://www.expedia.com",
        "https://www.google.com/flights",
        "https://www.trip.com"
    ]
    # 각 사이트를 줄바꿈하여 표시할 수 있도록 문자열로 반환
    return "\n".join(sites)


@tool("jeju_recommendation", return_direct=True)
def jeju_recommendation(location: str) -> str:
    """Provides a response if a location outside Jeju is requested."""
    if "제주" not in location:
        return "죄송합니다. 제주도 추천 챗봇입니다. 다른 지역 추천을 원하시면 네이버에서 검색해보세요: https://www.naver.com"
    else:
        return "제주도 여행 추천 챗봇입니다. 무엇을 도와드릴까요?"

@tool("search", return_direct=True)
def search_site(query: str) -> List[Dict[str, str]]:
    """Search News by input keyword"""
    site_tool = TavilySearchResults(
        max_results=6,
        include_answer=True,
        include_raw_content=True,
        include_domains=["google.com", "naver.com"],
    )
    return site_tool.invoke({"query": query})

@tool
def say_hello(user_input: str) -> str:
    """Responds with a greeting if the user says hello or a similar greeting."""
    greetings = ["안녕하세요", "hello", "hi", "안녕", "안뇽", "헬로", "하이"]
    
    # 인삿말을 사용자가 입력한 경우 응답 반환
    if any(greeting in user_input.lower() for greeting in greetings):
        return "안녕하세요✋ 제주 여행을 계획중이신가요? 필요하신 정보를 입력해주세요 ☺️"
    else:
        return ""


# 도구들을 리스트로 묶어 정의
tools = [recommend_airplane_site, jeju_recommendation, search_site, say_hello]