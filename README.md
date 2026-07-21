# LLM Function-Calling Agent

A small AI agent that can browse and read files in a project folder by reasoning about which tool to use at each step, built to compare two different ways of connecting an LLM's decisions to real code execution.

## What it does

The agent is given a task in plain English (e.g. "what files are in this folder?") and autonomously decides which of its tools to call, in what order, until the task is complete:

- `list_files(directory)` — lists files in the project folder or a specific subfolder
- `read_file(file_name)` — reads the contents of a file
- `terminate(message)` — ends the task and reports a summary back to the user

The agent runs in a loop: it builds a prompt from its instructions and everything that has happened so far, asks the LLM what to do next, executes the chosen tool, and feeds the result back into memory for the next iteration. This continues until the model calls `terminate` or a safety limit is reached.

Both tools are sandboxed to the project's own folder: any request to read a file outside that folder (via `../` traversal or an absolute path) is rejected, and `.env` is explicitly blocked from ever being read, regardless of what's asked.

## Two implementations, one comparison

This repo intentionally contains two versions of the same agent, to demonstrate two different architectures for connecting an LLM to real software actions:

**`file_agent.py`** — uses prompt engineering + manual parsing. The model is instructed to reply in a specific text format (a fenced ` ```action ` block containing JSON), which the program then has to extract and parse by hand. This approach works with any LLM, but is inherently fragile: it depends on the model reliably following formatting instructions, and requires custom error-recovery logic when it doesn't.

**`func_file_agent.py`** — uses native LLM function calling. Tools are defined once using JSON Schema and passed directly to the model's API. The model returns a structured, guaranteed-valid tool call instead of free text, eliminating the need for manual parsing entirely. This version additionally returns structured `{"content": ...}` / `{"error": ...}` responses from every tool (rather than plain strings) and includes "just-in-time" error messages that point the agent back toward `list_files` when it makes a mistake. Both versions share the same folder-sandboxing and `.env` protection.

## Tech stack

- Python
- [LiteLLM](https://github.com/BerriAI/litellm) — unified interface across LLM providers (Google Gemini in this implementation, swappable to OpenAI, Anthropic, etc.)
- Google Gemini API (`gemini-flash-latest`)

## Running it

1. Create a virtual environment and install dependencies:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Create a `.env` file in this folder with your API key:
   ```
   GEMINI_API_KEY=your_key_here
   ```
3. Run either version:
   ```
   python func_file_agent.py
   ```
   or
   ```
   python file_agent.py
   ```
4. When prompted, describe what you'd like the agent to do, e.g. "what files are in this project?"
