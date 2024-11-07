from langchain_memory_common.model import creat_chat_model, create_prompt
from langchain_memory_common.prompt import create_runnable_lambda
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from datetime import datetime

def make_chain(model_id):
    return create_prompt() | creat_chat_model(model_id)

def runnable_parallel_chain():
    chain_1 = (
        PromptTemplate.from_template("{country}의 날씨는 어때?")
        | creat_chat_model()
    )
    chain_2 = (
        PromptTemplate.from_template("{country}의 건국 몇주년이야?")
        | creat_chat_model()
    )
    chain_3 = (
        PromptTemplate.from_template("{country}는 여행하기 좋은 나라야?")
        | creat_chat_model()
    )
    return RunnableParallel(capital=chain_1, area=chain_2, recommend=chain_3)

def get_today(a):
    return datetime.today().strftime("%b-%d")
# chain 을 생성합니다.
def runnable_lambda_chain():
    chain = (
        {"today": RunnableLambda(get_today), "country": RunnablePassthrough()}
        | create_runnable_lambda()
        | creat_chat_model()
    )
    return chain