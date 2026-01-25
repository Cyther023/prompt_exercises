import argparse
import os
import sys
import httpx
from typing import List

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_groq import ChatGroq


def web_search(query: str) -> str:
    """Simple web search via Serper API."""
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return "Error: SERPER_API_KEY not set."
    try:
        with httpx.Client(timeout=10) as client:
            response = client.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
                json={"q": query, "num": 5},
            )
            response.raise_for_status()
            data = response.json()
            items = data.get("organic", [])
            if not items:
                return "No results found."
            snippets = [f"{i.get('title', '')}: {i.get('snippet', '')}" for i in items]
            return "\n".join(snippets)
    except Exception as e:
        return f"Search error: {e}"


def should_fallback(answer: str) -> bool:
    """Heuristic: if the answer admits uncertainty or is generic, fallback to search."""
    uncertain = [
        "i don't know",
        "i'm not sure",
        "i do not have",
        "i cannot",
        "i am not aware",
        "i don't have real-time",
        "i cannot browse",
    ]
    return any(phrase in answer.lower() for phrase in uncertain)


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Exercise 14: Chatbot with memory and fallback")
    parser.add_argument(
        "--model",
        default=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        help="Groq model name (or set GROQ_MODEL)",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Model temperature")
    parser.add_argument("--verbose", action="store_true", help="Show fallback reasoning")
    args = parser.parse_args()

    if not os.getenv("GROQ_API_KEY"):
        print("Missing GROQ_API_KEY. Set it in your environment or create a .env file.", file=sys.stderr)
        return 2

    llm = ChatGroq(model=args.model, temperature=args.temperature)

    # Prompt with memory
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful chatbot with memory. Answer the user's question based on the conversation history. "
                "If you are not certain or the information might be outdated, say: 'I am not sure; I can search the web for you.'",
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    # Fallback prompt after search
    fallback_prompt = ChatPromptTemplate.from_template(
        "You are a helpful chatbot. Use the following web search results to answer the user's question.\n"
        "If the results are empty or irrelevant, say 'I could not find reliable information.'\n\n"
        "Question: {question}\n\nSearch results:\n{results}\n\nAnswer:"
    )

    chain_with_memory = prompt | llm | StrOutputParser()
    fallback_chain = fallback_prompt | llm | StrOutputParser()

    history: List[BaseMessage] = []

    print("Chatbot with memory and fallback. Type 'exit' to quit.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            break
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        # First attempt: answer with memory
        answer = chain_with_memory.invoke({"input": user_input, "history": history})
        if should_fallback(answer):
            if args.verbose:
                print(f"[Bot] Uncertain, falling back to web search...", file=sys.stderr)
            results = web_search(user_input)
            if args.verbose:
                print(f"[Search Results]\n{results}\n", file=sys.stderr)
            answer = fallback_chain.invoke({"question": user_input, "results": results})

        print(f"Bot: {answer}")
        # Update history
        history.extend([HumanMessage(content=user_input), AIMessage(content=answer)])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
