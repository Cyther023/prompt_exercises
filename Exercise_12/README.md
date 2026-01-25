# Exercise 12 — Agent with External Tools (LangChain)

## Concept: What is an agent with tools?
An **agent** is a system that decides which **tool** to use to answer a question. Instead of hardcoding logic, the model:

1. Chooses a tool (e.g., web search).
2. Calls the tool with an input.
3. Observes the result.
4. Repeats until it can provide a final answer.

LangChain makes it easy to define tools and let the LLM orchestrate them.

## What this exercise demonstrates
This lab builds a minimal CLI agent with one tool:

- **Web search (Serper API)**: fetches live web results and returns top snippets.

The app uses a simple heuristic to decide whether to search, then calls the tool and synthesizes an answer using Groq. This avoids the complexity of full agent executors while still demonstrating tool integration.

## Key idea
Instead of writing if/else logic, you give the model **tools** and let it **reason** about which to use. This scales to more complex workflows.

## Files
- `agent_app.py` — the agent CLI app
- `requirements.txt` — dependencies
- `.env` — Groq key + Serper key (you create this)

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
SERPER_API_KEY=your_serper_api_key
```

3. Run:

```bash
python agent_app.py "Who won yesterday wpl in india" 
```

```bash
python agent_app.py "What is the latest news about OpenAI?"
```

Optional:
- `--model` to override the model
- `--temperature` to control creativity
- `--verbose` to see the agent’s reasoning steps
