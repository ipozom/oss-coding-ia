# OSS Coding IA Agent

This project implements an open-source coding agent using **LangGraph** and **Ollama**.

## Prerequisites

- **Ubuntu 22.04**
- **NVIDIA GPU** (GTX 1080 Ti)
- **Ollama** installed and running
- **Python 3.10+**

## Setup

1. **Install Ollama**:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. **Pull the Model**:
   ```bash
   ollama pull qwen2.5-coder:14b
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the agent:
```bash
python src/main.py
```

## Features

- **Local Execution**: Uses Ollama to run models on your GTX 1080 Ti.
- **Multi-Cloud Support**: Configure OpenAI or Gemini by changing a single environment variable.
- **Stateful Agent**: Built with LangGraph for reliable tool use and loops.
- **Loop Protection**: Built-in detection for redundant tool calls and runaway loops.
- **Heuristic Tool Parsing**: Specialized handling for local models that output tool calls in varied formats.
- **Human-Readable Output**: Clean formatted output with emojis and truncation for better user experience.
- **Coding Tools**:
  - `read_file`: Read source code safely.
  - `write_file`: Create or edit files (with folder auto-creation).
  - `run_command`: Execute terminal commands (build, test, etc.).
  - `list_directory`: Explore project structure.

## Technical Documentation

For in-depth technical details on the agent's logic, heuristics, and architecture, see [DOCUMENTATION.md](DOCUMENTATION.md).

## Hardware Optimization

With 32GB RAM and 11GB VRAM, the `qwen2.5-coder:14b` model fits comfortably. If you experience slowness, you can switch to `qwen2.5-coder:7b` for even faster performance.