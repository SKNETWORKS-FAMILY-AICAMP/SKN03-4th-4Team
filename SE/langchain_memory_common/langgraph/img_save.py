# PIL(Pillow) 라이브러리 사용
from PIL import Image
from langchain_memory_common.langgraph.display import make_img

def save_langgraph_structure_img():
    print(make_img())
    if make_img() != None:
        make_img().save('./output.jpg')
