# Exercise 9 — Prompt Chaining (LangChain + Groq)

## Concept: Prompt Chaining
**Prompt chaining** is a pattern where you run multiple prompts in sequence, where the output of one step becomes an input to the next step.

Instead of asking the model to do everything in one shot, you split the problem into smaller steps. This usually improves:

- structure
- clarity
- reliability

## What this exercise demonstrates
This lab implements a simple 2-step chain:

1. **Outline step**
   Generates a bullet-point outline for your topic.

2. **Expand step**
   Uses the outline to write the final answer.

In short: **plan first, then write**.

## Why this helps
- The outline forces the model to organize ideas before generating the final text.
- The second step uses the outline as constraints, reducing off-topic output.
- Each step is easier to prompt and debug than one large prompt.

## Files
- `chain_app.py` — the chaining CLI app
- `requirements.txt` — Python dependencies
- `.env` — your Groq key and model (you create this)

## Important note
Do not commit `.env` files or API keys to git. Keep keys private.

## How to run (quick)
1. Install:

```bash
pip install -r requirements.txt
```

2. Create `.env`:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
```

3. Run:

```bash
python chain_app.py "Explain transformers in simple terms" --audience "beginners" --style "concise"
```
