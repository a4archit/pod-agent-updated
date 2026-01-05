#------------------------------------------------------------------------------------------
# Dependencies
#------------------------------------------------------------------------------------------

# external
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from dotenv import load_dotenv

# local
from rag import ConversationalAgenticRAG
from configs import PodagentConfigs

# built-in
from typing import List, Optional, Literal, Annotated, Dict






## loading secret keys
load_dotenv()







#------------------------------------------------------------------------------------------
# Confiurations
#------------------------------------------------------------------------------------------

COMMON_LLM = "gemini-2.5-flash-lite"
COMMON_TOKENS_SIZE = 32000



#------------------------------------------------------------------------------------------
# LLM instance: Gemini
#------------------------------------------------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model = COMMON_LLM,
    verbose = False,
    max_tokens = COMMON_TOKENS_SIZE,
    temperature = 0.3 
)







#------------------------------------------------------------------------------------------
# Schema
#------------------------------------------------------------------------------------------

class PodagentSchema(BaseModel):

    messages: Annotated[List[BaseMessage], add_messages]
    fetched_docs: Optional[str] = ""









#------------------------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------------------------

def get_last_human_message(messages: List[BaseMessage]) -> HumanMessage:
    """
    It will return the last HumanMessage from the provided messages list
    
    :param messages: list of messages
    :type messages: List[BaseMessage]
    :return: last human message
    :rtype: HumanMessage
    """


    for msg in messages[::-1]:
        if isinstance(msg, HumanMessage):
            return msg











#------------------------------------------------------------------------------------------
# Nodes
#------------------------------------------------------------------------------------------

def agent_chat_node(state: PodagentSchema):
    messages = state.model_dump()['messages']
    print(f"\n\nAgent chat node [come in] -> messages: {messages}")
    response = llm.invoke(f"{messages}, \n\nretrieved docs: {state.fetched_docs} ")

    final_response = {
        "messages": messages + [response]
    }

    print(f"\n\nreturning from agent node -> {final_response}")
    return final_response




rag = ConversationalAgenticRAG(PodagentConfigs.pdf_path)

def retriever(state: PodagentSchema) -> dict:
    """
    use to get information from knowledge base
    
    :param query: query that use to retrieve docs
    :type query: str
    :return: retrieved docs
    :rtype: dict
    """
    query = get_last_human_message(state.messages).content
    print(f"\n\nQuery = {query}\n")
    retrieved_docs = rag.fetch_docs(query=query, conversational=False)

    merged = ""
    for doc in retrieved_docs:
        merged += doc.page_content

    # print(f"\n\nMerged: {merged}")

    return { "fetched_docs": merged }

    











#------------------------------------------------------------------------------------------
# Workflow
#------------------------------------------------------------------------------------------

# graph instance
graph = StateGraph(PodagentSchema)

# adding nodes
graph.add_node("agent", agent_chat_node)
graph.add_node("retriever", retriever)

# connecting edges
graph.set_entry_point("retriever")
graph.add_edge("retriever","agent")
graph.add_edge("agent", END)


# extracting workflow
workflow = graph.compile()







#------------------------------------------------------------------------------------------
# Test agent
#------------------------------------------------------------------------------------------

def test_agent():

    prompt = "list out all chapter names"
    # prompt = "how many states in india"
    initial_state = PodagentSchema(messages=[HumanMessage(content=prompt)])
    response = workflow.invoke(initial_state)

    print(response)

    print(f"\n\nAI) {response['messages'][-1].content}")








if __name__ == "__main__":

    from subagents.chapter_content_loader import test_agent as t2

    t2()

    # initial_state = PodagentSchema(messages=[HumanMessage(content="hello")])

    # print(initial_state)









