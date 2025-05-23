# HumanAI-Project
A teaching tool that provides prompts to a small LLM

## Installation

### Requirements:
1. Ollama running on the local host with the specific model (here gemma3:1b)
2. Gradio

### Install uv
* `uv sync`
* `source .venv/bin/activate`
* `curl -fsSL https://ollama.com/install.sh | sh`



# UI:
1. Summarizes the paper by default
2. Follow-up questions in the same tab as the summary
3. Hardcoded the prompt (provided by Ian) to summarize the paper
4. Drop-down to choose from different LLM models (currently only gemm3 with 1B parameters)
5. PDF Parser is used. (PyPDF2)
6. Gradio for the UI to create a public sharable link
7. Live stream output
8. Prompt Suggestions (3 prompts)

# Shortcuts: 
1. Enter: To ask the follow-up question
2. Shift+Enter: To enter into a new line

# To Do:
1. Dynamic Prompt Suggestion updation
2. Test a better system to reduce the processing time.
3. Dedicated Mcq generation (including prompt)
