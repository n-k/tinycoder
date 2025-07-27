#!/usr/bin/env -S uv run -q --script
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2025 Nipun Kumar
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pydantic",
#   "langchain-core",
#   "langchain-community",
#   "langchain-google-genai",
#   "langchain-ollama",
#   "langchain-openai",
#   "py-jsonl",
#   "requests",
#   "bashlex",
# ]
# ///

"""
TinyCoder AI Code Assistant
=============================
"""

import argparse
import asyncio
import datetime
import os
import itertools
import json
import jsonl
import queue
import requests
import secrets
import string
import subprocess
import sys
import tempfile
import threading
import time
from typing import Optional, List, Any

import bashlex

from pydantic import BaseModel, Field

from langchain_core.messages import (
    message_to_dict,
    messages_from_dict,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_ollama.chat_models import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


SCRIPT_PATH = os.path.abspath(__file__)
SYSTEM_PROMPT = """\
You are a seasoned unix hacker and programmer.
You do EVERYTHING using the command line and with standard unix tools like
    ls, cat, grep, sed, awk, find ...
====
"""
SAFE_COMMANDS = {
    'ls', 'find', 'grep', 'echo', 'cat', 'pwd', 'which', 'whoami',
    'date', 'head', 'tail', 'wc', 'sort', 'uniq', 'diff', 'basename',
    'dirname', 'stat',
}


def _get_command_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser for the CLI tool.

    Returns:
        argparse.ArgumentParser: Configured argument parser with subcommands
            for initializing shell environment and sending messages.
    """
    parser = argparse.ArgumentParser(
        description="AI Assistant CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Create subparsers for subcommands
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    # init_shell subcommand
    subparsers.add_parser(
        "init_shell",
        help="""Initialize shell environment
        This command should be called as:
        source <(path/to/script.py init_shell)
        """
    )

    # message subcommand
    message_parser = subparsers.add_parser(
        "message", help="Send a single message")
    message_parser.add_argument(
        "text", nargs=argparse.REMAINDER, help="Message text to send"
    )

    subparsers.add_parser("find_free", help="Find free models on openrouter")

    return parser


async def main():
    """Main entry point for the CLI tool.

    Parses command line arguments and routes to the appropriate function
    based on the subcommand provided (init_shell or message).
    """
    args = _get_command_parser().parse_args()

    if args.command == "init_shell":
        init_shell()
    elif args.command == "message":
        message(" ".join(args.text))
    elif args.command == "find_free":
        find_free()
    else:
        _get_command_parser().print_help()


def find_free():
    def _is_free_w_tools(model):
        params = model.get('supported_parameters', [])
        if 'tools' not in params:
            return False
        pricing = model.get("pricing", {})
        for k in pricing:
            if not pricing[k] == '0':
                return False
        return True
    response = requests.get("https://openrouter.ai/api/v1/models")
    models = response.json().get('data', [])
    free_tool_models = [
        model for model in models if _is_free_w_tools(model)]

    print(f'Found {len(free_tool_models)}')
    for m in free_tool_models:
        print(m.get('id'))


def init_shell():
    """Initialize the shell environment by generating shell script commands.

    Creates a unique log file for the session and outputs shell configuration
    commands including aliases for AI interaction (ai,aiedit) and deactivation.
    The function sets up shell environment variables and functions needed for
    TinyCoder to operate within the shell.
    """
    temp_dir = tempfile.gettempdir()
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    rand_str = ''.join(secrets.choice(
        string.ascii_letters + string.digits) for _ in range(6))
    log_file_path = os.path.join(
        temp_dir, f"tinycoder_{timestamp}_{rand_str}.jsonl")
    shell_script = f"""
    # This command is meant to be run like
    # source <(path/to/tinycoder.py init_shell)
if [[ -n "$TINY_CODER_ACTIVE" ]]; then
    echo "TinyCoder is already active."
    return 0  || exit 0
fi

export TINY_CODER_ACTIVE=1
alias ai='{SCRIPT_PATH} message $@'
alias aiedit='_edit_and_message'
alias deactivate='_deactivate_tiny_coder'
OLD_PS1=$PS1
touch {log_file_path}
export TINY_CODER_LOG_PATH={log_file_path}
echo Temp chat log file @ {log_file_path}
_edit_and_message() {{
  tmpfile=$(mktemp)
  ${{EDITOR:-vim}} "$tmpfile" || return
  arg=$(<"$tmpfile")
  rm "$tmpfile"
  {SCRIPT_PATH} message "$arg"
}}
_deactivate_tiny_coder() {{
    unalias ai
    unalias aiedit
    unalias deactivate
    PS1="$OLD_PS1"
    unset TINY_CODER_ACTIVE
    unset TINY_CODER_LOG_PATH
    unset OLD_PS1
    unset -f _deactivate_tiny_coder
    unset -f _edit_and_message
    rm -f {log_file_path}
}}
PS1="[✨ai] $PS1"
"""
    print(shell_script)


def message(text):
    """Process and log a user message, then generate AI response.

    Loads previous conversation from log file, appends the new message,
    and invokes the AI model to generate a response. The conversation
    history is maintained in the log file.

    Args:
        text (str): The user's message text to be processed.
    """
    log_file = os.environ.get("TINY_CODER_LOG_PATH", None)
    if os.environ.get("TINY_CODER_ACTIVE", None) is None or log_file is None:
        print(
            """TinyCoder is not active.
            Did you initialize your shell with `source <(path/to/tinycoder.py 
            init_shell)` ?
            """,
            file=sys.stderr,
        )
        return
    log = jsonl.load(log_file)
    log = list(log)
    messages: List[BaseMessage] = []
    for d in log:
        m = messages_from_dict([d])
        messages.append(m[0])
    if len(messages) == 0:
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
    messages.append(HumanMessage(content=text))
    messages = _make_progress(messages)
    jsonl.dump([message_to_dict(m) for m in messages], log_file)


def get_cwd() -> str:
    """Get the current working directory.

    Returns:
        str: The absolute path of the current working directory.
    """
    return os.getcwd()


def _get_client():
    """Initialize and configure the appropriate LLM client based on environment settings.
    Configures the client with tool binding for command execution and user interaction.

    Returns:
        A configured LLM client with tool binding.

    Raises:
        SystemExit: If required API keys are not set for the selected provider.
    """
    model_provider = os.environ.get("MODEL_PROVIDER", 'openrouter')
    model_name = os.environ.get("MODEL_NAME", 'qwen/qwen3-coder:free')

    llm = None
    if model_provider == 'openrouter':
        if os.environ.get("OPENROUTER_API_KEY", None) is None:
            print('You must set OPENROUTER_API_KEY environment variable',
                  file=sys.stderr)
            sys.exit(1)
        llm = ChatOpenAI(
            api_key=os.environ.get("OPENROUTER_API_KEY", None),
            openai_api_key=os.environ.get("OPENROUTER_API_KEY", None),  # type: ignore
            openai_api_base="https://openrouter.ai/api/v1",  # type: ignore
            model_name=model_name,  # type: ignore
        )
    elif model_provider == 'google':
        if os.environ.get("GOOGLE_API_KEY", None) is None:
            print('You must set GOOGLE_API_KEY environment variable', file=sys.stderr)
            sys.exit(1)
        llm = ChatGoogleGenerativeAI(model=model_name or "gemini-2.0-flash")
    elif model_provider == 'ollama':
        llm = ChatOllama(
            base_url=os.environ.get("OLLAMA_BASE_URL", 'localhost'),
            model=model_name,
            format="json",
        )
    else:
        print(f"Unsupported model provider '{model_provider}'. Exiting...")
        sys.exit(1)

    llm = llm.bind_tools(
        [
            ExecuteCommand,
            AskFollowupQuestion,
        ],
    )
    return llm


def _make_progress(messages):
    """Process messages through the LLM and handle tool calls.

    Sends messages to the LLM and processes any tool calls in the response.
    Handles both direct tool calls and tool calls embedded in the response content.
    Continues processing until a final response is generated.

    Args:
        messages (List[BaseMessage]): List of messages in the conversation history.

    Returns:
        List[BaseMessage]: Updated list of messages including responses and tool results.
    """
    # user_input: str = ""
    skip_input = True
    client = _get_client()
    while skip_input:
        # reset skip for next round
        skip_input = False
        response: Any = client.invoke(messages)
        messages.append(response)
        tool_calls = []
        if len(response.tool_calls) > 0:
            tool_calls = response.tool_calls
        else:
            # maybe content is tool call json?
            content = response.content
            _parsed = None
            if isinstance(content, str):
                try:
                    _parsed = json.loads(content)
                except:
                    pass
            elif isinstance(content, dict):
                _parsed = content
            if (
                _parsed is not None
                and "name" in _parsed
                and ("arguments" in _parsed or "args" in _parsed)
            ):
                if "args" not in _parsed:
                    _parsed["args"] = _parsed["arguments"]
                tool_calls = [_parsed]
        if len(tool_calls) == 0:
            # we can't proceed, break
            print(response.content)
            return messages
        t = tool_calls[0]
        tool_name = t["name"]
        args = t["args"]
        can_continue, tool_result = run_tool(
            tool_name,
            args,
        )
        skip_input = can_continue
        if tool_result is not None:
            # we don't set tool message if None
            # because if we asked user a question,
            # we want to get a HumanMessage from the user
            messages.append(
                ToolMessage(
                    tool_call_id=t.get("id", ""),
                    status="success",
                    content=tool_result,
                )
            )
    return messages


class ExecuteCommand(BaseModel):
    """Request to execute a CLI command on the system.

    Use this for EVERYTHING. Some ideas are:
    This tool can be used to create or overwrite files using `echo` or `touch` commands.
    This tool can be used to list files using `ls` command.
    This tool can be used to find text in files using `grep` command.
    This tool can be used to replace text in a file using the `sed` command.

    Args:
    - command: (required) The CLI command to execute. This should be valid for the current operating system. 
        Ensure the command is properly formatted and does not contain any harmful instructions.

    Typical usage examples:

    Example - list files:
    execute_command(command="ls -a")
    Example - list files recursively:
    execute_command(command="ls -aR")
    Example - read/show file contents:
    execute_command(command="cat path/to/the/file")
    Example - create file:
    execute_command(command="touch path/to/the/file")
    Example - delete file:
    execute_command(command="rm path/to/the/file")
    """

    command: str = Field(
        ...,
        description="The CLI command to execute. This should be valid for the current operating system. Ensure the command is properly formatted and does not contain any harmful instructions.",
    )


class AskFollowupQuestion(BaseModel):
    """Ask the user a question to gather additional information needed to complete the task.

    This tool should be used to ask for clarifications about the current task only.
    You must avoid conversation if possible.
    """

    question: str = Field(
        ...,
        description=f"The question to ask the user. This should be a clear, specific question that addresses the information you need.",
    )
    options: Optional[List[str]] = Field(
        None,
        description=f"An array of 2-5 options for the user to choose from. Each option should be a string describing a possible answer. You may not always need to provide options, but it may be helpful in many cases where it can save the user from having to type out a response manually. IMPORTANT: NEVER include an option to toggle to Act mode, as this would be something you need to direct the user to do manually themselves if needed.",
    )


def execute_command(args: ExecuteCommand) -> str:
    """Execute a CLI command on the system and return its output.

    Runs the specified command using subprocess and captures its output.
    Handles both successful execution and errors, returning appropriate results.

    Args:
        args (ExecuteCommand): The command to execute.

    Returns:
        str: The command output or error message.
    """
    class Walker(bashlex.ast.nodevisitor): # type: ignore
        def __init__(self):
            super().__init__()
            self.commands = []
        
        def visitcommand(self, n, parts):
            if len(parts) > 0:
                if parts[0].kind == 'word':
                    self.commands.append(parts[0].word)
            return True # visit children
    tree = bashlex.parse(args.command)
    if not isinstance(tree, list):
        tree = [tree]
    walker = Walker()
    for t in tree:
        walker.visit(t)
    commands = walker.commands
    if not all(cmd in SAFE_COMMANDS for cmd in commands):
        if not input(f"Allow running {args.command} ? [y/n]: ") == 'y':
            return f'User rejected running this command: {args.command}'
    def _run_command(command, output_queue, result_dict):
        error_label = 'Error while running command'
        success_label = 'Command ran successfully'
        try:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            result_dict['process'] = process
            output_lines = []
            
            # Read output line by line as it's generated
            if process.stdout:
                for line in iter(process.stdout.readline, ''):
                    line_stripped = line.rstrip()
                    output_lines.append(line_stripped)
                    # Send output line through the queue
                    output_queue.put(('output', line_stripped))
            
            process.wait()
            result_dict['code'] = process.returncode
            stdout = '\n'.join(output_lines)
            
            if process.returncode == 0: 
                label = success_label 
            else:
                label = error_label
            result_dict['result'] = f"${command}\\n{label}\\n{stdout}"
            
            # Signal completion
            output_queue.put(('done', result_dict['code']))
            
        except:
            result_dict['code'] = 1
            result_dict['result'] = f"${command}\\n{error_label}\\n"
            output_queue.put(('done', 1))
    
    # Create queue for thread communication
    output_queue = queue.Queue()
    result_dict = {}
    
    thread = threading.Thread(
        target=_run_command, args=(args.command, output_queue, result_dict), 
        daemon=True)
    thread.start()
    
    spinner = itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
    label = f'[Ctrl+C to kill] ${commands[0]} ...'
    command_finished = False
    sys.stdout.write(f"\r[{next(spinner)}] {label}   ")
    sys.stdout.flush()
    try:
        while not command_finished:
            try:
                # Check for new output with a short timeout
                msg_type, data = output_queue.get(timeout=0.1)
                
                if msg_type == 'output':
                    sys.stdout.write(f"\r{' ' * 80}\r")  # Clear spinner line
                    print(data)
                    sys.stdout.write(f"\r[{next(spinner)}] {label}   ")
                    sys.stdout.flush()
                elif msg_type == 'done':
                    command_finished = True
                    result_dict['code'] = data
            except queue.Empty:
                sys.stdout.write(f"\r[{next(spinner)}] {label}   ")
                sys.stdout.flush()
        
        thread.join()
        
        # Clear spinner line before showing final status
        sys.stdout.write(f"\r{' ' * 80}\r")
        
        if result_dict.get("code", 1) == 0:
            sys.stdout.write(f"[✅] {label}\n")
        else:
            sys.stdout.write(f"[❌] {label}\n")
        sys.stdout.flush()
        
    except KeyboardInterrupt:
        sys.stdout.write(f"\r{' ' * 80}\r[✖️] Interrupted\n")
        sys.stdout.flush()
        process = result_dict.get("process")
        if process:
            try:
                process.terminate()
                process.wait(timeout=3)
            except Exception:
                process.kill()
        result_dict['result'] = f'${args.command}\\nKilled by user'
    
    return result_dict.get('result', f'${args.command}\\nNo result')


def ask_followup_question(args: AskFollowupQuestion):
    """Present a question to the user for additional information.

    Formats and displays a question to the user, along with optional answer choices.
    Used when the AI needs clarification to complete a task.

    Args:
        args (AskFollowupQuestion): The question and optional answer options.

    Returns:
        None: This function prints the question and returns None.
    """
    options = "\\n".join(
        [f"- {opt}" for opt in args.options]) if args.options else ""
    print(f"Question: {args.question}\\n{options}")
    return None


def run_tool(name, args):
    """Execute a tool by name with the provided arguments.

    Args:
        name (str): The name of the tool to execute.
        args (dict): Arguments to pass to the tool.

    Returns:
        tuple: A tuple containing:
            - bool: Whether processing can continue automatically
            - str or None: The tool's output or None if user input is needed
    """
    if name == "ExecuteCommand" or name == "execute_command":
        exec = ExecuteCommand.model_validate(args)
        return True, execute_command(exec)
    elif name == "AskFollowupQuestion" or name == "ask_followup_question":
        return False, ask_followup_question(AskFollowupQuestion.model_validate(args))
    return False, "Tool not found"


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, EOFError):
        print("Exit")
