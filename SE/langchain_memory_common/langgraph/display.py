# from IPython.display import Image, display
# from langchain_langgraph.langgraph.graph import set_workflow_to_app

# def make_img():
#     try:
#         img = display(Image(set_workflow_to_app().get_graph().draw_mermaid_png()))
#         return img
#     except Exception:
        # return None
    
import graphviz
from langchain_memory_common.langgraph.graph import set_workflow_to_app

def make_img():
    try:
        # Graphviz DOT 형식으로 변환
        dot = graphviz.Digraph()
        graph = set_workflow_to_app().get_graph()
        
        # 노드와 엣지 추가
        for node in graph.nodes():
            dot.node(str(node))
        for edge in graph.edges():
            dot.edge(str(edge[0]), str(edge[1]))
            
        # PNG 파일로 저장
        dot.render("workflow", format="png", cleanup=True)
        return "workflow.png"
    except Exception:
        return None

# import networkx as nx
# import matplotlib.pyplot as plt
# from langchain_langgraph.langgraph.graph import set_workflow_to_app

# def make_img():
#     try:
#         # NetworkX 그래프 생성
#         G = nx.DiGraph()
#         graph = set_workflow_to_app().get_graph()
        
#         # 노드와 엣지 추가
#         G.add_nodes_from(graph.nodes())
#         G.add_edges_from(graph.edges())
        
#         # 그래프 그리기
#         plt.figure(figsize=(10, 8))
#         nx.draw(G, with_labels=True, node_color='lightblue', 
#                 node_size=1500, arrowsize=20)
        
#         # PNG 파일로 저장
#         plt.savefig("workflow.png")
#         plt.close()
#         return "workflow.png"
#     except Exception:
#         return None