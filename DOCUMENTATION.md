# OSS Coding IA Agent Documentation

This document provides a technical overview of the components and logic behind the OSS Coding IA Agent.

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Component Details](#component-details)
    - [src/main.py](#srcmainpy)
    - [src/agent.py](#srcagentpy)
    - [src/tools.py](#srctoolspy)
3. [Key Technical Features](#key-technical-features)
    - [Multi-Provider Support](#multi-provider-support)
    - [State Management & Reducers](#state-management--reducers)
    - [Infinite Loop Protection](#infinite-loop-protection)
    - [Tool Call Heuristics](#tool-call-heuristics)

---

## Architecture Overview

The agent is built using **LangGraph**, which allows for a stateful, cyclic execution flow. The graph consists of two main nodes:
- **Agent Node**: Calls the LLM and decides whether to invoke a tool or provide a final answer.
- **Action Node**: Executes the tools requested by the Agent Node and returns the results back to the graph state.

The state is managed using a `TypedDict` that stores a sequence of messages, which is updated using the `add_messages` reducer to ensure a complete conversational history is maintained.

---

## Component Details

### `src/main.py`
The entry point of the application. It handles:
- Environment variable loading.
- Path resolution (adding the project root to `sys.path`).
- The REPL (Read-Eval-Print Loop) for user interaction.

### `src/agent.py`
The "brain" of the agent. It contains:
- **LLM Selection Logic**: Dynamically chooses between Ollama (local), OpenAI, or Gemini.
- **Graph Definition**: Sets up the `StateGraph`, nodes, and edges.
- **System Prompt**: Provides the persona and operational constraints for the AI.
- **Wait/Continue Logic**: Determines if the model's response requires tool execution or should terminate the loop.

### `src/tools.py`
Defines the capabilities available to the agent:
- `read_file`: Reads content from the filesystem.
- `write_file`: Writes content to the filesystem (auto-creates directories).
- `run_command`: Executes shell commands with a 30-second timeout.
- `list_directory`: Lists files in a given directory.

---

## Key Technical Features

### Multi-Provider Support
The agent is provider-agnostic. By changing the `LLM_PROVIDER` in the `.env` file, users can switch between:
- **Local**: Ollama (e.g., `qwen2.5-coder:14b`)
- **Cloud**: OpenAI (GPT-4o) or Google (Gemini 1.5 Pro/Flash)

### State Management & Reducers
Using `Annotated[Sequence[BaseMessage], add_messages]` allows the agent to remember every turn. This is critical for preventing the agent from re-reading the same file repeatedly, as it has a record of previous tool outputs.

### Infinite Loop Protection
To prevent runaway processes and excessive resource usage, two layers of protection are implemented:
1. **Turn Limit**: If the conversation exceeds 10 messages, a nudge is sent to the model to finish.
2. **Repetition Detection**: If the same tool with the same arguments is called three times consecutively, the graph terminates automatically with a `LOOP DETECTED` warning.

### Tool Call Heuristics
Local models like Qwen sometimes output tool calls as raw JSON strings or within Markdown blocks instead of using the formal `tool_calls` attribute. `src/agent.py` contains a heuristic parser that detects these patterns and manually populates the `tool_calls` object to ensure consistent execution.
