# Exercise 13 — Autonomous Agent (Task Decomposition)

## Concept: What is an autonomous agent?
An **autonomous agent** takes a high-level task, breaks it into subtasks, executes each step, and synthesizes a final outcome. It mimics how a person would plan and work through a problem.

## What this exercise demonstrates
This lab builds a minimal autonomous agent that:

1. **Decomposes** a user task into ordered subtasks (via LLM, JSON list).
2. **Executes** each subtask sequentially (LLM reasoning per step).
3. **Exports** the entire process (task, subtasks, each step’s reasoning + result) to a timestamped text file for inspection.

## Key idea
Instead of hardcoding logic, we let the LLM both plan and solve. The exported log makes the agent’s reasoning transparent and easy to review.

## Files
- `autonomous_agent.py` — the autonomous agent CLI
- `requirements.txt` — dependencies
- `.env` — Groq key + model (you create this)

## Important note
Do not commit `.env` files or API keys to git. Keep keys private.

## How to run (quick)
1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
```

3. Run:

```bash
python autonomous_agent.py "Plan a 3-day trip to Tokyo"
```

```bash
python autonomous_agent.py "Write a simple blog post about climate change"
```

Optional:
- `--model` to override the model
- `--temperature` to control creativity
- `--export-dir` to change where the process log is saved (default: `process_logs`)

The agent will print progress and save a detailed log like `process_logs/task_process_20250124_235959.txt`.
