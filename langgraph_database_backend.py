from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import sqlite3
import os

load_dotenv(override=True)

api_key = os.environ.get("GROQ_API_KEY")
if api_key and api_key.strip():
    api_key = api_key.strip()
    os.environ["GROQ_API_KEY"] = api_key

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

from langchain_core.messages import SystemMessage

def chat_node(state: ChatState):
    messages = state['messages']
    
    # Prepend a system message to enforce short answers
    sys_msg = SystemMessage(content="You are a helpful medical assistant. Please keep your answers short, concise, and to the point.")
    
    # Invoke LLM with the system message followed by the user messages
    response = llm.invoke([sys_msg] + messages)
    return {"messages": [response]}

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
# Checkpointer
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)

def delete_thread(thread_id):
    cursor = conn.cursor()
    cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (str(thread_id),))
    cursor.execute("DELETE FROM writes WHERE thread_id = ?", (str(thread_id),))
    conn.commit()

def delete_all_threads():
    cursor = conn.cursor()
    cursor.execute("DELETE FROM checkpoints")
    cursor.execute("DELETE FROM writes")
    conn.commit()

