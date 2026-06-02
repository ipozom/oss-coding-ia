import json
from typing import Annotated, Sequence, TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from src.tools import read_file, write_file, run_command, list_directory

# Define the state for the agent
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]

# Initialize the LLM with Ollama
llm = ChatOllama(model="qwen2.5-coder:14b", temperature=0)

# Bind tools to the LLM
tools = [read_file, write_file, run_command, list_directory]
llm_with_tools = llm.bind_tools(tools)

# Define the system prompt
system_prompt = SystemMessage(content=(
    "You are an expert AI software engineer. "
    "You have access to tools to read, write, and execute code. "
    "When you need to use a tool, you MUST use the provided tool-calling format. "
    "Do not just explain what you will do; actually call the tool. "
    "For example, to see a file, call `read_file(path='tools.py')`. "
    "IMPORTANT: Use absolute paths or paths relative to the project root. "
    "Do NOT use 'src/' prefix if you are currently inside the 'src' directory. "
    "After you get the tool result, continue your analysis. "
    "Current date is June 2, 2026."
))

import json
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolCall

# ... (rest of imports)

# Define the function that calls the model
def call_model(state: AgentState):
    messages = state['messages']
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [system_prompt] + list(messages)
    
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
    last_message = state['messages'][-1]
    # Check both structured tool_calls AND raw content that looks like a tool call
    if last_message.tool_calls:
        return "continue"
    
    # Check if content looks like a tool call in a code block or raw JSON
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
