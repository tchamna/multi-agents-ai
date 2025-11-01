# Multi-Agent AI System with CrewAI and LangChain

A sophisticated multi-agent AI system built with CrewAI and LangChain that demonstrates how multiple AI agents can collaborate to research, analyze, write, and review content on any given topic.

## Overview

This project implements a multi-agent system with four specialized agents:

1. **Research Agent** - Conducts thorough research on topics
2. **Analyst Agent** - Analyzes information and provides insights
3. **Writer Agent** - Creates engaging content based on research
4. **Reviewer Agent** - Reviews and provides quality assurance

The agents work together in a sequential process, each building upon the work of the previous agent to produce high-quality, well-researched content.

## Features

- 🤖 Multiple specialized AI agents with distinct roles
- 🔄 Sequential task processing workflow
- 🔍 Web search and scraping capabilities
- 📊 Data analysis and insights generation
- ✍️ Content creation and quality review
- 🔗 Integration with LangChain and OpenAI

## Prerequisites

# Multi-Agent AI System — CrewAI + LangChain

This project demonstrates a small multi-agent system where specialized AI "agents"
collaborate to research a topic, analyze findings, write an article, and review the
content. It uses CrewAI to orchestrate agents and LangChain/OpenAI as the LLM layer.

The README below focuses on the technology used, what runs under the hood, and
step-by-step instructions so anyone (technical or not) can run and understand it.

## Key technologies

- CrewAI: agent orchestration library used to define agents, tasks, and processes.
- LangChain / OpenAI: provides the large language model used by each agent.
- crewai-tools: optional helper tools (web search, scraping) used by the researcher.
- python-dotenv: loads API keys from a `.env` file.

This project was built with Python and intended to be run inside a virtual
environment (venv). It creates a `runs/` directory for each execution and saves
per-task outputs as well as a final Markdown (`.md`) file.

## What the system does (high level)

1. Researcher agent collects information (using search / scraping tools when available).
2. Analyst agent analyzes the researched material and extracts insights.
3. Writer agent composes an article from the analysis and research.
4. Reviewer agent inspects the article and provides feedback.

Agents are executed sequentially by default. Each agent runs a task and the system
saves intermediate outputs so you can inspect each step.

## Project layout (what to look at)

Files you will use most:

- `agents.py` — defines each agent, their role, and the LLM settings.
- `tasks.py` — creates Task objects (instructions and expected outputs).
- `tools.py` — tools available to agents (search/scrape). Requires API keys for some tools.
- `main.py` — orchestrates the run, creates `runs/<timestamp>/` and writes outputs.
- `requirements.txt` — Python packages used by this project.

## Quick start (Windows PowerShell)

1) Create and activate a Python virtual environment (recommended):

```powershell
# Create venv with whichever python you want; example shown for python3.11 if installed
py -3.11 -m venv .\venv311
# Activate the venv
.\venv311\Scripts\Activate.ps1
```

2) Install dependencies (inside the activated venv):

```powershell
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
```

3) Create your `.env` and add API keys (OpenAI required). Copy the example and edit:

```powershell
copy .\.env.example .\.env
notepad .\.env
```

Set at minimum:

```
OPENAI_API_KEY=sk-...
```

4) Run the system. You can run interactively or pass a topic via CLI (the script supports prompting):

```powershell
# Interactive prompt (recommended)
python .\main.py

# Or provide a topic (if implemented) --example
python .\main.py --topic "African Languages"
```

When the run completes a new directory will be created under `runs/`.

## Where outputs are saved

Each run writes a timestamped folder and files:

- `runs/YYYYMMDD_HHMMSS/task-1-research.txt` — researcher's raw findings
- `runs/YYYYMMDD_HHMMSS/task-2-analysis.txt` — analyst's insights
- `runs/YYYYMMDD_HHMMSS/task-3-writing.txt` — writer's article draft
- `runs/YYYYMMDD_HHMMSS/task-4-review.txt` — reviewer's feedback
- `runs/YYYYMMDD_HHMMSS/final_output.md` — final combined output (Markdown)
- `runs/YYYYMMDD_HHMMSS/summary.json` — run metadata (timestamps, file names, durations)

The `final_output.md` is formatted with a header (topic, generated timestamp, duration)
followed by the aggregated final result. This makes it easy to publish or copy into a
content pipeline.

## Example run flow (what you will see)

1. Start the script and enter a topic when prompted.
2. The researcher runs and saves a `task-1` file.
3. The analyst reads the research and writes `task-2`.
4. The writer composes the article and writes `task-3`.
5. The reviewer examines the article and writes `task-4`.
6. `final_output.md` and `summary.json` are produced in the run folder.

## Making it robust / customization

- Change the LLM: edit `agents.py` and set a different `model` or `temperature`.
- Add tools: expand `tools.py` with more tools or custom tool functions.
- Process types: switch `Process.sequential` to `Process.hierarchical` if you want
    manager/worker patterns (see CrewAI docs).
- Persisting outputs: the project already saves per-task outputs; you can change
    the destination folder or file naming in `main.py`.

## Troubleshooting

- Module import errors: ensure you're running inside the venv where packages were installed.
    Activate the venv, then run `python -m pip list` to verify installed packages.
- `OPENAI_API_KEY` missing: verify `.env` is present in the project root and contains the key.
- `crewai` or `crewai_tools` missing or fail to install: some packages may require build tools
    (C/C++ compiler) or specific wheels for your Python version. Use Python 3.11 on Windows
    and install prebuilt wheels where possible.

## Small checklist for first run

1. Create/activate venv
2. pip install -r requirements.txt
3. copy `.env.example` → `.env` and add keys
4. Run `python .\main.py` and enter a topic
5. Inspect `runs/<timestamp>/` for outputs

## Next steps and suggestions

- Add CLI flags for non-interactive automation (topic, output folder, verbosity).
- Add tests that assert tasks produce outputs of expected shape (use a small stub LLM for tests).
- Hook the final Markdown into a static site generator or CMS for automatic publishing.

If you'd like, I can:

1. Add a `--topic` CLI flag to `main.py` for non-interactive runs.
2. Add unit tests that mock the LLM and validate per-task saving.
3. Convert `final_output.md` into a nicer article template (frontmatter, author, tags).

Tell me which of the above you'd like next and I'll implement it.

---

License: MIT
