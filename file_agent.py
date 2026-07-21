import json
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())  # this line reads your .env file and loads GEMINI_API_KEY so the code below can use it

from litellm import completion
from typing import List, Dict

# BASE_DIR = the folder this script itself lives in
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# TARGET_DIR = one folder up from that = the project root
TARGET_DIR = os.path.dirname(BASE_DIR)

# Files the agent is never allowed to read, no matter what it's asked.
BLOCKED_FILES = {".env"}


def extract_markdown_block(response: str, block_type: str = "json") -> str:
    """
    The AI's reply comes back as one big block of text, and inside that
    text there's usually a section wrapped like this:

    ```action
    {"tool_name": "list_files", "args": {}}
    ```

    This function's job is simple: dig out just the part between those
    ``` marks, so we're left with only the {...} part and none of the
    ``` symbols around it.
    """
    if not '```' in response:
        return response

    code_block = response.split('```')[1].strip()

    if code_block.startswith(block_type):
        code_block = code_block[len(block_type):].strip()

    return code_block


def generate_response(messages: List[Dict]) -> str:
    """
    This is the function that actually talks to the AI model.
    You give it the full conversation so far (as a list), and it
    sends that to Gemini and gives you back the AI's reply as plain text.
    """
    response = completion(
        model="gemini/gemini-flash-latest",
        messages=messages
    )
    choice = response.choices[0]  # type: ignore
    return choice.message.content.strip()  # type: ignore


def parse_action(response: str) -> Dict:
    """
    This function turns the AI's text reply into something our Python
    code can actually use: a dictionary with a "tool_name" (which tool
    to run) and "args" (what information that tool needs).

    Two things can go wrong here: the text inside the ``` marks might
    not be valid JSON, or it might be valid JSON but missing the pieces
    we need. In both cases, instead of crashing the program, we return
    a small dictionary that says "tool_name": "error". This error gets
    shown to the AI on the next turn, so it has a chance to notice its
    mistake and try again correctly.
    """
    try:
        response = extract_markdown_block(response, "action")
        response_json = json.loads(response)
        if "tool_name" in response_json and "args" in response_json:
            return response_json
        else:
            return {"tool_name": "error", "args": {"message": "You must respond with a JSON tool invocation."}}
    except json.JSONDecodeError:
        return {"tool_name": "error", "args": {"message": "Invalid JSON response. You must respond with a JSON tool invocation."}}


def list_files(directory: str = "") -> List[str]:
    """
    A "tool" the agent can use. Lists every file in the current folder,
    or a subfolder of it. The AI does NOT run this code itself — it can
    only ask us (the program) to run it on its behalf.
    """
    path = os.path.join(TARGET_DIR, directory) if directory else TARGET_DIR
    try:
        return os.listdir(path)
    except Exception as e:
        return [f"Error: {str(e)}"]


def read_file(file_name: str) -> str:
    """
    Another tool: opens a file and returns its text content. Refuses to
    read blocked files like .env, and never lets the program crash if
    something goes wrong.
    """
    if os.path.basename(file_name) in BLOCKED_FILES:
        return "Error: reading this file is not permitted."

    file_path = os.path.join(TARGET_DIR, file_name)
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return f"Error: {file_name} not found."
    except (IsADirectoryError, PermissionError):
        return f"Error: {file_name} is a folder, not a file. Use list_files with that folder name to see what's inside it."
    except Exception as e:
        return f"Error: {str(e)}"


def preview(value, limit=300):
    """Shorten long text so our debug prints stay readable in the terminal."""
    text = str(value)
    if len(text) > limit:
        return text[:limit] + f"... [truncated — {len(text)} characters total]"
    return text


agent_rules = [{
    "role": "system",
    "content": """
You are an AI agent that can perform tasks by using available tools.

Available tools:

```json
{
    "list_files": {
        "description": "Lists files in the project folder, or inside a specific subfolder if one is given.",
        "parameters": {
        "directory": {
            "type": "string",
            "description": "Optional. A subfolder name to look inside. Leave blank to list the top-level folder."
            }
        }
    },
    "read_file": {
        "description": "Reads the content of a file.",
        "parameters": {
            "file_name": {
                "type": "string",
                "description": "The name of the file to read."
            }
        }
    },
    "terminate": {
        "description": "Ends the agent loop and provides a summary of the task.",
        "parameters": {
            "message": {
                "type": "string",
                "description": "Summary message to return to the user."
            }
        }
    }
}
```

If a user asks about files, documents, or content, first list the files before reading them.

When you are done, terminate the conversation by using the "terminate" tool and I will provide the results to the user.

Important!!! Every response MUST have an action.
You must ALWAYS respond in this format:

<Stop and think step by step. Parameters map to args. Insert a rich description of your step by step thoughts here.>

```action
{
    "tool_name": "insert tool_name",
    "args": {...fill in any required arguments here...}
}
```"""
}]

# max_iterations is just a safety limit. Without it, if something went
# wrong and the AI never called "terminate", this program could keep
# looping forever. This guarantees it stops after 10 tries no matter what.
max_iterations = 10

user_task = input("What would you like me to do? ")

memory = [{"role": "user", "content": user_task}]

iterations = 0

# --- This while loop is the actual "agent" part of the program ---
# It keeps repeating these same six steps until either the AI says
# it's finished, or we hit the max_iterations safety limit.
while iterations < max_iterations:

    # STEP 1 — Build the full message we're about to send to the AI.
    prompt = agent_rules + memory

    # STEP 2 — Actually send that to the AI and get its reply back.
    print("Agent thinking...")
    response = generate_response(prompt)
    print(f"Agent response: {response}")

    # STEP 3 — Turn the AI's text reply into a usable Python dictionary.
    action = parse_action(response)
    result = "Action executed"

    # STEP 4 — Look at which tool the AI asked for, and actually run it.
    if action["tool_name"] == "list_files":
        directory = action["args"].get("directory", "")
        result = {"result": list_files(directory)}
    elif action["tool_name"] == "read_file":
        result = {"result": read_file(action["args"]["file_name"])}
    elif action["tool_name"] == "error":
        result = {"error": action["args"]["message"]}
    elif action["tool_name"] == "terminate":
        print(action["args"]["message"])
        break
    else:
        result = {"error": "Unknown action: " + action["tool_name"]}

    print(f"Action result: {preview(result)}")

    # STEP 5 — Save what just happened into memory, so next time
    # through the loop, the AI can see it.
    memory.extend([
        {"role": "assistant", "content": response},
        {"role": "user", "content": json.dumps(result)}
    ])

    # STEP 6 — Check again whether the AI wanted to stop.
    if action["tool_name"] == "terminate":
        break

    iterations += 1
