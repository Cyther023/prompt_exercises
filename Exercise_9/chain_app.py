import argparse
import os
import sys

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq


def build_chain(model: ChatGroq):
    outline_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that produces structured, actionable outlines.",
            ),
            (
                "human",
                "Create a clear outline for a response about: {topic}.\n"
                "Audience: {audience}\n"
                "Style: {style}\n"
                "Return only the outline as bullet points.",
            ),
        ]
    )

    expand_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that writes high-quality final responses from an outline.",
            ),
            (
                "human",
                "Using the outline below, write the final response.\n\n"
                "Topic: {topic}\n"
                "Audience: {audience}\n"
                "Style: {style}\n\n"
                "Outline:\n{outline}\n\n"
                "Write the final answer.",
            ),
        ]
    )

    outline_chain = outline_prompt | model | StrOutputParser()

    chain = (
        RunnablePassthrough.assign(outline=outline_chain)
        | expand_prompt
        | model
        | StrOutputParser()
    )

    return chain


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Exercise 9: LangChain prompt chaining")
    parser.add_argument("topic", help="Topic to generate a response about")
    parser.add_argument("--audience", default="general", help="Target audience")
    parser.add_argument("--style", default="concise", help="Writing style")
    parser.add_argument(
        "--model",
        default=os.getenv("GROQ_MODEL", "llama3-8b-8192"),
        help="Groq model name (or set GROQ_MODEL)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Model temperature",
    )
    args = parser.parse_args()

    if not os.getenv("GROQ_API_KEY"):
        print(
            "Missing GROQ_API_KEY. Set it in your environment or create a .env file.",
            file=sys.stderr,
        )
        return 2

    model = ChatGroq(model=args.model, temperature=args.temperature)
    chain = build_chain(model)

    result = chain.invoke(
        {"topic": args.topic, "audience": args.audience, "style": args.style}
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
