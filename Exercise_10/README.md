# Exercise 10 — Retrieval-Augmented Generation (RAG)

## Concept: What is RAG?
**Retrieval-Augmented Generation (RAG)** is a pattern where the model answers a question using **external documents**.

Instead of relying only on what the model “knows”, RAG adds a retrieval step:

1. **Retrieve** the most relevant parts of your documents.
2. **Generate** an answer using the retrieved text as context.

This makes answers more **grounded**, more **up-to-date**, and easier to verify.

## What this exercise demonstrates
This lab implements a minimal RAG pipeline over **custom documents you provide**:

- **Load** a file or a folder of files
- **Split** documents into chunks
- **Embed** chunks into vectors
- **Store** vectors in a vector database
- **Retrieve** the top-k most relevant chunks for a question
- **Answer** using Groq, constrained to the retrieved context

Supported document types in this exercise:

- `.pdf`
- `.txt`
- `.md`

## Key idea
The model is prompted to use **ONLY** the retrieved context. If the context doesn’t contain the answer, it should respond with:

`I don't know based on the provided documents.`

## Files
- `rag_app.py` — the RAG CLI app
- `streamlit_app.py` — upload + chat UI (optional)
- `requirements.txt` — dependencies
- `.env` — Groq key + model (you create this)

## Streamlit UI settings
When you run `streamlit run streamlit_app.py`, the sidebar provides these controls:

- **Groq model**: Which Groq model to use for answering (e.g., `llama-3.3-70b-versatile`).
- **Temperature**: Controls creativity (0 = deterministic, 1 = very creative).
- **Top-k retrieval**: How many document chunks to retrieve for each question.
- **Chunk size**: Length of each text chunk when splitting documents.
- **Chunk overlap**: Overlap between consecutive chunks (helps preserve context).

## Extra 
You can export the generated chunks and their embedding vectors as simple text files. This makes it easier to see what RAG is doing.

The export creates:

- `rag_chunks.txt`
- `rag_vectors.txt`

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

3. Streamlit mode (upload + chat):

```bash
streamlit run streamlit_app.py
```
