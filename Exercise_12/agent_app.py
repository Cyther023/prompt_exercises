import argparse
import os
import sys
import httpx

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_groq import ChatGroq


# --------------------
# Real web search via Serper API
# --------------------
@tool
def web_search(query: str) -> str:
    """Searches the web using Serper API and returns top results."""
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return "Error: SERPER_API_KEY not set in environment."
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


def should_search(question: str) -> bool:
    """Simple heuristic: if the question asks for recent info or specific facts, search."""
    q = question.lower()
    triggers = ["who", "what", "when", "where", "latest", "news", "won", "result", "current", "today"]
    return any(t in q for t in triggers)


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Exercise 12: LangChain agent with external tools")
    parser.add_argument("question", help="Question for the agent to answer using tools")
    parser.add_argument(
        "--model",
        default=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        help="Groq model name (or set GROQ_MODEL)",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Model temperature")
    parser.add_argument("--verbose", action="store_true", help="Show agent reasoning steps")
    args = parser.parse_args()

    if not os.getenv("GROQ_API_KEY"):
        print("Missing GROQ_API_KEY. Set it in your environment or create a .env file.", file=sys.stderr)
        return 2

    llm = ChatGroq(model=args.model, temperature=args.temperature)

    search_tool = RunnableLambda(web_search)

    # Decide whether to search
    decision_prompt = ChatPromptTemplate.from_template(
        "You are deciding whether to search the web for the question: {question}\n"
        "Respond with only 'yes' or 'no'."
    )
    decision_chain = decision_prompt | llm | StrOutputParser()

    # Final answer prompt
    answer_prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Use the following web search results to answer the user's question.\n"
        "If the results are empty or irrelevant, say 'I could not find reliable information.'\n\n"
        "Question: {question}\n\nSearch results:\n{results}\n\nAnswer:"
    )
    answer_chain = answer_prompt | llm | StrOutputParser()

    # Decide whether to search
    if should_search(args.question):
        if args.verbose:
            print("[Decision] Searching the web.", file=sys.stderr)
        results = search_tool.invoke(args.question)
        if args.verbose:
            print(f"[Search Results]\n{results}\n", file=sys.stderr)
        answer = answer_chain.invoke({"question": args.question, "results": results})
    else:
        if args.verbose:
            print("[Decision] Not searching; answering directly.", file=sys.stderr)
        direct_prompt = ChatPromptTemplate.from_template(
            "You are a helpful assistant. Answer the question concisely.\n\nQuestion: {question}\n\nAnswer:"
        )
        answer = direct_prompt | llm | StrOutputParser()
        answer = answer.invoke({"question": args.question})

    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
