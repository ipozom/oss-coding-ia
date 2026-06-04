import json
import os
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolCall
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from src.tools import read_file, write_file, run_command, list_directory

# Load environment variables
load_dotenv()

# Define the state for the agent
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]

def get_llm():
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    
    if provider == "openai":
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"), # Optional, for OpenRouter
            temperature=0
        )
    elif provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", "gemini-1.5-flash"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0
        )
    else:
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "qwen2.5-coder:14b"),
            temperature=0
        )

# Initialize the LLM
llm = get_llm()

# Bind tools to the LLM
tools = [read_file, write_file, run_command, list_directory]
llm_with_tools = llm.bind_tools(tools)

# Define the system prompt
system_prompt = SystemMessage(content=(
    "You are an expert AI software engineer. "
    "Your goal is to perform coding tasks using the provided tools. "
    "CRITICAL: After you have gathered enough information using tools (like reading a file), "
    "you MUST stop using tools and provide your final answer/analysis in plain text. "
    "Do NOT repeat the same tool call if you already have the result. "
    "If you have read a file once, do not read it again. "
    "Current directory structure contains a 'src' folder. Relative paths like 'src/tools.py' are correct. "
    "ALWAYS end your turn by providing a helpful response to the user."
))

# Define the function that calls the model
def call_model(state: AgentState):
    messages = list(state['messages'])
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [system_prompt] + messages
    
    # LIMIT: If we have too many messages, the model might be looping
    if len(messages) > 10:
        # Prompt the model to finish
        messages.append(HumanMessage(content="You have used many tools. Please summarize your findings and provide a final answer now."))

    response = llm_with_tools.invoke(messages)
    
    # HEURISTIC: If tool_calls is empty but content looks like a tool call JSON, parse it
    content = response.content.strip()
    if not response.tool_calls:
        # Handle markdown blocks if present
        json_str = content
        if content.startswith("```json"):
            json_str = content.replace("```json", "").replace("```", "").strip()
        elif content.startswith("```"):
            json_str = content.replace("```", "").strip()

        if json_str.startswith("{"):
            try:
                tool_data = json.loads(json_str)
                if "name" in tool_data and "arguments" in tool_data:
                    response.tool_calls = [{
                        "name": tool_data["name"],
                        "args": tool_data["arguments"],
                        "id": response.id or "call_" + response.response_metadata.get("id", "123")
                    }]
            except:
                pass
            
    return {"messages": [response]}

# Define the logic to determine whether to continue or end
def should_continue(state: AgentState):
    messages = state['messages']
    last_message = messages[-1]
    
    # 1. INFINITE LOOP PROTECTION: If the last 3 turns are the same tool call, stop.
    if len(messages) > 6:
        last_three_tool_calls = []
        for m in reversed(messages):
            if isinstance(m, AIMessage) and m.tool_calls:
                last_three_tool_calls.append(m.tool_calls[0]['name'] + str(m.tool_calls[0]['args']))
            if len(last_three_tool_calls) == 3:
                break
        
        if len(last_three_tool_calls) == 3 and len(set(last_three_tool_calls)) == 1:
            print("--- LOOP DETECTED, TERMINATING ---")
            return "end"

    # 2. Check structured tool_calls
    if last_message.tool_calls:
        return "continue"
    
    # 3. Check for raw/markdown tool calls
    content = last_message.content.strip()
    if (content.startswith("```json") and "}" in content) or (content.startswith("{") and "}" in content):
        return "continue"
        
    return "end"

# Build the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", call_model)
workflow.add_node("action", ToolNode(tools))

# Set entry point
workflow.set_entry_point("agent")

# Add edges
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END
    }
)
workflow.add_edge("action", "agent")

# Compile the graph
app = workflow.compile()

def run_agent(query: str):
    inputs = {"messages": [HumanMessage(content=query)]}
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Output from node '{key}':")
            print(value)
            print("-" * 20)
