from .constant import CHATBOT_MESSAGE, CHATBOT_ROLE
from langchain_core.prompts import PromptTemplate
from langchain import hub

def create_message(role:CHATBOT_ROLE, prompt:str):

    return {
        CHATBOT_MESSAGE.role.name: role.name,
        CHATBOT_MESSAGE.content.name: prompt
    }

def create_runnable_lambda():
    return PromptTemplate.from_template("{today} 날짜에 발생한 {country}의 역사적 사실 1개 알려줘")

def pull_langgraph_prompt():
    return hub.pull("hwchase17/openai-functions-agent")