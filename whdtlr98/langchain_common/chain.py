import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
import time
from langchain_common.prompt import create_prompt

# 모델 생성
chat = ChatOpenAI(
    model="gpt-4o-mini"
)

base_word = ["취업", "취업정보", "취업사이트", "취업공부법", "채용"]


def response_from_langchain(prompt, message_history=None):

    #  체인 실행 및 응답 처리 공통 함수 호출
    response_content = run_chain(prompt)

    #  응답이 문자열인지 확인하고 한 글자씩 출력
    if isinstance(response_content, str):
        for char in response_content:
            yield char
            time.sleep(0.01)

def run_chain(prompt):
    if any(word in prompt for word in base_word):
        # 키워드에 해당하면 advice 체인 실행       
        chain = (
            {"user_input": RunnableLambda(advice)}
            | create_prompt()
            | chat
        )
        response = chain.invoke({"user_input": f"{prompt}"})
    else:
        # 일반 프롬프트 처리
        chain = create_prompt() | chat

    # 체인 실행 후 응답 생성
    response = chain.invoke({"user_input": f"{prompt}"})
    return response.content  # 응답 내용을 반환



def advice(prompt):
    """두 개의 체인을 병렬로 실행하고, 결과를 반환합니다."""

    # 첫 번째 체인: 취업 관련 응답
    chain1 = (
        {"user_input": RunnablePassthrough()}
        | PromptTemplate.from_template("{user_input}와 취업에 필요한건 뭐야")
        | chat
    )

    # 두 번째 체인: 한국의 수도 관련 응답
    chain2 = (
        {"user_input": RunnablePassthrough()}
        | PromptTemplate.from_template("{user_input}과 취업의 응원멘트를 알려줘")
        | chat
    )

    # 두 체인을 병렬로 실행
    combined_chain = RunnableParallel(web=chain1, ment=chain2)

    # 'prompt'를 입력으로 두 체인에 전달
    response = combined_chain.invoke({"user_input": prompt})


    # 각 체인의 응답을 병합하여 반환
    combined_content = f"{response['web'].content}\n{response['ment'].content}"
    return combined_content  # 병합된 응답 반환


