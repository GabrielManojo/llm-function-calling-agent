import json
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from litellm import completion
from typing import List, Dict

# BASE_DIR = the folder this script itself lives in — this repo's root.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# TARGET_DIR = the folder the agent is allowed to browse.
TARGET_DIR = BASE_DIR

# Files the agent is never allowed to read, no matter what it's asked.
# Compared case-insensitively, since Windows filesystems are.
BLOCKED_FILES = {".env"}


def safe_path(file_name: str) -> str:
    """
    Resolves `file_name` relative to TARGET_DIR and verifies the result
    is still actually inside TARGET_DIR. This blocks path traversal
    (e.g. "../../secrets.txt"), absolute paths that would otherwise
    let the agent escape the intended folder entirely, AND symbolic
    links inside the project folder that point somewhere else.

    Uses realpath() rather than abspath(): abspath() only cleans up
    the path string (resolving "..", making it absolute) without
    touching the filesystem, so a symlink inside the project folder
    that points outside it would pass an abspath()-based check even
    though actually opening it would read/write outside the sandbox.
    realpath() follows any symlinks to their real target first, so
    the containment check runs against where the path actually leads.
    Raises ValueError if the resulting path is outside TARGET_DIR.
    """
    candidate = os.path.realpath(os.path.join(TARGET_DIR, file_name))
    target_root = os.path.realpath(TARGET_DIR)
    try:
        common = os.path.commonpath([candidate, target_root])
    except ValueError:
        raise ValueError("That path is outside the allowed folder.")
    if common != target_root:
        raise ValueError("That path is outside the allowed folder.")
    return candidate


def list_files(directory: str = "") -> Dict:
    """Lists files inside the project folder, or a subfolder of it."""
    try:
        path = safe_path(directory) if directory else TARGET_DIR
    except ValueError as e:
        return {"error": str(e)}
    try:
        return {"files": os.listdir(path)}
    except Exception as e:
        return {"error": f"Could not list '{directory or 'top-level folder'}': {str(e)}. Call list_files with no directory to see the available top-level folders."}


def read_file(file_name: str) -> Dict:
    """Reads a file's contents. Refuses to read blocked files like .env,
    and refuses to read anything outside the project folder entirely."""
    if os.path.basename(file_name).lower() in BLOCKED_FILES:
        return {"error": "Reading this file is not permitted."}

    try:
        file_path = safe_path(file_name)
    except ValueError as e:
        return {"error": str(e)}

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return {"content": file.read()}
    except FileNotFoundError:
        return {"error": f"'{file_name}' does not exist. Call list_files to see the correct file names and their folder."}
    except (IsADirectoryError, PermissionError):
        return {"error": f"'{file_name}' is a folder, not a file. Use list_files with that folder name to see what's inside it."}
    except Exception as e:
        return {"error": str(e)}


def terminate(message: str) -> str:
    """Ends the loop. Whatever message is passed in gets shown to the user."""
    return message


tool_functions = {
    "list_files": list_files,
    "read_file": read_file,
    "terminate": terminate,
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "Lists files in the project folder, or inside a specific subfolder if one is given.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Optional. A subfolder name to look inside. Leave blank to list the top-level folder."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Reads the content of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "The name of the file to read, e.g. 'subfolder/file.py'."
                    }
                },
                "required": ["file_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "terminate",
            "description": "Ends the agent loop and provides a summary of the task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Summary message to return to the user."
                    }
                },
                "required": ["message"]
            }
        }
    }
]

agent_rules = [{
    "role": "system",
    "content": """
You are an AI agent that can perform tasks by using available tools.

If a user asks about files, documents, or content, first list the files before reading them.

When you are done, call the "terminate" tool with a summary message.
"""
}]


def preview(value, limit=300):
    """Shorten long text so debug prints stay readable in the terminal."""
    text = str(value)
    if len(text) > limit:
        return text[:limit] + f"... [truncated — {len(text)} characters total]"
    return text


max_iterations = 10
user_task = input("What would you like me to do? ")
memory: List[Dict] = [{"role": "user", "content": user_task}]

iterations = 0
while iterations < max_iterations:
    prompt = agent_rules + memory

    print("Agent thinking...")
    response = completion(
        model="gemini/gemini-flash-latest",
        messages=prompt,
        tools=tools
    )

    message = response.choices[0].message  # type: ignore

    if not message.tool_calls:
        print(f"Agent response: {message.content}")
        break

    tool_call = message.tool_calls[0]
    tool_name = tool_call.function.name
    tool_args = json.loads(tool_call.function.arguments)

    print(f"Agent wants to call: {tool_name}({tool_args})")

    if tool_name not in tool_functions:
        result = {"error": f"Unknown tool: {tool_name}"}
    else:
        try:
            output = tool_functions[tool_name](**tool_args)
            result = output if isinstance(output, dict) else {"result": output}
        except Exception as e:
            result = {"error": f"Error executing {tool_name}: {str(e)}"}

    print(f"Action result: {preview(result)}")

    memory.append({
        "role": "assistant",
        "content": message.content,
        "tool_calls": [{
            "id": tool_call.id,
            "type": "function",
            "function": {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments
            }
        }]
    })
    memory.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(result)
    })

    if tool_name == "terminate":
        # .get() with a fallback so a "terminate" call missing the
        # "message" argument doesn't crash with an uncaught KeyError.
        print(tool_args.get("message", "Task complete."))
        break

    iterations += 1
