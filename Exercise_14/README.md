# Exercise 14 — Chatbot with Memory and Fallback

## Concept: What is a chatbot with memory and fallback?
A chatbot with **memory** retains conversation history so it can refer back to earlier messages. **Fallback logic** means the bot tries to answer from its own knowledge first; if it’s uncertain, it falls back to an external tool (e.g., web search) and then answers.

## What this exercise demonstrates
This lab builds a CLI chatbot that:

- Maintains conversation memory (LangChain messages).
- Attempts to answer using its knowledge.
- Detects uncertainty and triggers a web search (Serper API) as a fallback.
- Uses the search results to give a final, grounded answer.

## Key idea
Memory makes conversations coherent. Fallback improves reliability when the bot’s knowledge is insufficient or outdated.

## Files
- `chatbot.py` — the chatbot CLI
- `streamlit_chatbot.py` — Streamlit UI with live memory sidebar (optional)
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

**CLI mode:**
```bash
python chatbot.py
```

**Streamlit mode (UI with memory sidebar):**
```bash
streamlit run streamlit_chatbot.py
```

Optional flags (CLI only):
- `--model` to override the model
- `--temperature` to control creativity
- `--verbose` to see when fallback is triggered and the raw search results

Type `exit` or `quit` to stop the CLI chat. In Streamlit, just close the tab.

## Example interaction
```
You: Who won the 2022 World Cup?
Bot: I am not sure; I can search the web for you.
[searches web]
Bot: Argentina won the 2022 World Cup, defeating France on penalties.
```
