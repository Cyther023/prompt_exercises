# Exercise 11 — ReAct-Style Prompting

## Concept: What is ReAct?
**ReAct** = **Reason + Act**. It’s a prompting pattern where the model:

1. **Reasons** about what to do next.
2. **Acts** (chooses a tool/action).
3. **Observes** the result.
4. Repeats until it can answer.

This mimics human step-by-step problem solving and makes the model’s process visible.

## What this exercise demonstrates
This lab implements a minimal ReAct loop with Groq:

- The model outputs **JSON** with: `thought`, `action`, `action_input`.
- Supported actions: `search`, `calculate`, `finish`.
- We simulate simple observations (search returns a stub; calculate evaluates safe arithmetic).
- The loop repeats until the model chooses `finish`.

## Key idea
Instead of asking the model to answer directly, we force it to **think out loud** and **act step by step**. This improves reliability on multi-step questions.

## Files
- `react_app.py` — the ReAct CLI app
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
python react_app.py "What is 15 * 3 + 7?"
```

Optional:
- `--model` to override the model
- `--temperature` to control creativity
- `--max-steps` to set a reasoning step limit
