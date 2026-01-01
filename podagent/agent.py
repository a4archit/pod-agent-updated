
#---------------------------------------------------------------------------------------------------
# Dependencies 
#---------------------------------------------------------------------------------------------------

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END 
from langgraph.graph.message import add_messages
from typing import Literal, Dict, Annotated, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv






## loading secret keys
load_dotenv()





#---------------------------------------------------------------------------------------------------
# Configuration
#---------------------------------------------------------------------------------------------------


COMMON_LLM = "gemini-2.5-flash"






#---------------------------------------------------------------------------------------------------
# LLM instance
#---------------------------------------------------------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model = COMMON_LLM,
    max_tokens = 4096,
    verbose = False,
    temperature = 0.5
)






#---------------------------------------------------------------------------------------------------
# Schema
#---------------------------------------------------------------------------------------------------

class PodAgentSchema(BaseModel):

    messages: Annotated[List[BaseMessage], add_messages]






#---------------------------------------------------------------------------------------------------
# Nodes
#---------------------------------------------------------------------------------------------------

def agent_node(state: PodAgentSchema):

    messages = state.model_dump()['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}









#---------------------------------------------------------------------------------------------------
# Workflow
#---------------------------------------------------------------------------------------------------

# graph
graph = StateGraph(PodAgentSchema)

# adding nodes 
graph.add_node("agent", agent_node)

# adding edges 
graph.add_edge(START, "agent")
graph.add_edge("agent", END)

workflow = graph.compile()











#---------------------------------------------------------------------------------------------------
# Testing scripts
#---------------------------------------------------------------------------------------------------

def test_agent():
    config = {'configurable': {'thread_id':'thread-1'}}

    prompt = "who am i"
    initial_state = PodAgentSchema(messages=[HumanMessage(content=prompt)], config=config)
    # print(type(initial_state))
    response = workflow.invoke(initial_state)

    print(response['messages'])


