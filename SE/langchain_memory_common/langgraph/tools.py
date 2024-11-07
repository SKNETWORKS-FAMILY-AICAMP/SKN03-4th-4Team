##   Custom Tools
from langchain.tools import BaseTool, StructuredTool, Tool, tool
import random

@tool("upper_case", return_direct=True)
def to_lower_case(input:str) -> str:
  """Returns the input as all upper case."""
  return input.upper() #.lower()

@tool("random_number", return_direct=True)
def random_number_maker(input:str) -> str:
    """Returns a random number between 0-100."""
    return random.randint(0, 100)

# tools
def get_langgraph_tools():
    return [to_lower_case, random_number_maker]