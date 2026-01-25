import argparse
import json
import os
import sys
from typing import Literal

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

Action = Literal["search", "calculate", "finish"]


def simulate_action(action: Action, query: str) -> str:
    if action == "search":
        # Simple simulated search result
        return f"Search result for '{query}': The answer is 42."
    if action == "calculate":
        # Very naive calculator simulation
        try:
            # Evaluate only safe arithmetic expressions
            allowed = set("0123456789+-*/(). ")
            if all(c in allowed for c in query):
                result = eval(query)
                return f"Calculation result: {result}"
            else:
                return "Invalid expression for calculation."
        except Exception:
            return "Error during calculation."
    if action == "finish":
        return ""
    raise ValueError("Unknown action")


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Exercise 11: ReAct-style prompting CLI")
    parser.add_argument("question", help="Question to answer using ReAct")
    parser.add_argument(
        "--model",
        default=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        help="Groq model name (or set GROQ_MODEL)",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Model temperature")
    parser.add_argument("--max-steps", type=int, default=6, help="Maximum reasoning steps")
    args = parser.parse_args()

    if not os.getenv("GROQ_API_KEY"):
        print("Missing GROQ_API_KEY. Set it in your environment or create a .env file.", file=sys.stderr)
        return 2

    llm = ChatGroq(model=args.model, temperature=args.temperature)

    react_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that uses ReAct (Reason + Act) to answer questions. "
                "For each step, output a JSON with exactly these keys:\n"
                "- 'thought': your reasoning\n"
                "- 'action': one of: 'search', 'calculate', 'finish'\n"
                "- 'action_input': the input for the action (omit if action is 'finish')\n\n"
                "Example:\n"
                '{{"thought": "I need the current year.", "action": "search", "action_input": "current year"}}\n\n'
                "When you have enough information, use action 'finish' and omit 'action_input'.",
            ),
            (
                "human",
                "Question: {question}\n\nPrevious steps:\n{history}\n\nNext step (JSON only):",
            ),
        ]
    )

    history: list[str] = []
    for step in range(args.max_steps):
        formatted_prompt = react_prompt.format_messages(
            question=args.question, history="\n".join(history) or "None"
        )
        response = llm.invoke(formatted_prompt)
        raw = response.content.strip()

        try:
            step_obj = json.loads(raw)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON from model:\n{raw}", file=sys.stderr)
            return 2

        thought = step_obj.get("thought", "")
        action = step_obj.get("action", "")
        action_input = step_obj.get("action_input", "")

        if action not in {"search", "calculate", "finish"}:
            print(f"Invalid action: {action}", file=sys.stderr)
            return 2

        step_line = f"Step {step + 1}: Thought: {thought} | Action: {action}"
        if action != "finish":
            step_line += f" | Action Input: {action_input}"
        print(step_line)

        if action == "finish":
            print("\n=== Final Answer ===")
            print(thought)
            return 0

        observation = simulate_action(action, action_input)
        print(f"Observation: {observation}\n")
        history.append(
            f"Step {step + 1}: Thought: {thought} | Action: {action} | Action Input: {action_input} | Observation: {observation}"
        )

    print("\nReached max steps without finishing.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
