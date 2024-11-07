
from .constant import CHATBOT_MESSAGE, CHATBOT_ROLE
from langchain_core.prompts import ChatPromptTemplate

def create_message(role:CHATBOT_ROLE, prompt:str):

    return {
        CHATBOT_MESSAGE.role.name: role.name,
        CHATBOT_MESSAGE.content.name: prompt
    }

# 프롬프트 생성
def create_prompt():
    return ChatPromptTemplate.from_messages([
        (CHATBOT_ROLE.assistant.name, "당신의 역할은 도움을 주는 어시스턴트입니다."),
        (CHATBOT_ROLE.user.name, "{user_input}"),
    ])


