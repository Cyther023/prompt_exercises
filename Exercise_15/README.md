# Exercise 15 — Self-Correcting Agent

## Concept: Self-correction loop
A **self-correcting agent** produces an output, evaluates it, and then improves it using the evaluation feedback. A simple and effective loop is:

1. **Draft**: generate an initial answer.
2. **Critique**: score the answer using a rubric and list concrete issues.
3. **Revise**: rewrite the answer to fix those issues.
4. Repeat until a target score is reached or a max-iteration limit is hit.

## What this exercise demonstrates
This lab implements a self-correction pipeline with:

- **Rubric-based scoring** (clarity, completeness, structure, style_fit, factuality)
- **Stop conditions** (target score / evaluator stop flag)
- **Iteration trace** (every draft + critique is visible)
- **Diff view** (see what changed between iterations)
- **Exportable logs** (TXT + JSON)

## Rubric (what each score means)
- `clarity`: How easy it is to understand (unambiguous wording, no confusing jumps).
- `completeness`: Whether it covers the key points required by the task and constraints.
- `structure`: Organization, headings/flow, and whether the response has a logical progression.
- `style_fit`: How well the tone/voice matches the requested style and audience.
- `factuality`: Whether claims are correct and supported (and penalized if contradicted by evidence when search is enabled).

Optional:
- **Evidence search** (Serper): fetches web snippets to help the evaluator judge factuality.

## Files
- `streamlit_app.py` — Streamlit app for running the self-correcting loop
- `requirements.txt` — dependencies
- `.env.example` — environment variables template

## Important note
Do not commit `.env` files or API keys to git. Keep keys private.

## How to run (quick)
1. Install:

```bash
pip install -r requirements.txt
```

2. Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
# Optional:
SERPER_API_KEY=your_serper_api_key
```

3. Run:

```bash
streamlit run streamlit_app.py
```

In the UI:
- Enter a task
- Choose iterations + target score
- Optionally enable evidence search
- Download logs (TXT/JSON) to inspect the full improvement process
