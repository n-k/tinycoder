# TinyCoder AI Code Assistant

## Overview

TinyCoder is a tiny AI coding assistant designed to be used from command-line.

## Features

*   Works directly within your existing shell environment (bash, zsh, etc.).
*   Tell assistant to do something directly from your command prompt
*   You can also use your favorite CLI editor to edit messages
*   Maintains context across your commands.
*   Can be used in scripts
*   Supports various AI model providers (Ollama, Google Gemini, OpenRouter). More can be added easily, as it is built on LangChain.

## Quick Start

To activate TinyCoder in your current session, run the following command:

```bash
source <(/path/to/tinycoder.py init_shell)
```

Replace `/path/to/tinycoder.py` with the actual path to this script.

I just copy tinycoder.py to my ~/.local/bin folder.

## Usage

Once activated, you'll have access to new aliases:

*   `ai <your prompt>`: Send a message to the AI assistant. The AI will respond by executing commands or asking follow-up questions.
*   `aiedit`: Open your default editor (`$EDITOR` or `vim` if not set), write your prompt, save, and exit. The content will be sent to the AI.

### Example

```bash
ai list all python files modified in the last 24 hours
```

## Deactivation

To deactivate TinyCoder and revert your shell to its previous state:

```bash
deactivate
```

## Configuration

You can configure the AI model by setting environment variables:

*   `MODEL_PROVIDER`: Set to `ollama` (default), `google`, or `openrouter`.
*   `MODEL_NAME`: Specify the model name (e.g., `qwen2.5-coder:32b-instruct` for Ollama, `gemini-2.5-flash` for Google, or specific models for OpenRouter).
*   `OLLAMA_BASE_URL`: For Ollama, specify the base URL if not `localhost`. E.g. `192.168.1.123`.
*   `GOOGLE_API_KEY`: Your Google API key if using the `google` provider.
*   `OPENROUTER_API_KEY`: Your OpenRouter API key if using the `openrouter` provider.
