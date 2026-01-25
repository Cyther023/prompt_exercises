import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


def decompose_task(task: str, llm: ChatGroq) -> list[str]:
    """Ask the LLM to break a task into subtasks (JSON list)."""
    prompt = ChatPromptTemplate.from_template(
        "Break the following task into a list of clear, ordered subtasks. "
        "Respond with a JSON array of strings only, no extra text.\n\nTask: {task}"
    )
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"task": task})
    try:
        subtasks = json.loads(raw)
        if not isinstance(subtasks, list):
            raise ValueError("Not a list")
        return [str(s) for s in subtasks]
    except Exception as e:
        print(f"Failed to parse subtasks: {e}\nRaw output: {raw}", file=sys.stderr)
        return []


def execute_subtask(subtask: str, llm: ChatGroq) -> str:
    """Ask the LLM to solve a single subtask."""
    prompt = ChatPromptTemplate.from_template(
        "You are an autonomous assistant. Solve the following subtask. "
        "Provide a concise answer or result.\n\nSubtask: {subtask}\n\nAnswer:"
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"subtask": subtask})


def export_process(task: str, subtasks: list[str], steps: list[dict], output_path: Path) -> None:
    """Write the full process to a text file for inspection."""
    lines = []
    lines.append(f"Task: {task}")
    lines.append(f"Timestamp: {datetime.utcnow().isoformat()}Z")
    lines.append("=" * 80)
    lines.append("\nDecomposed subtasks:")
    for i, s in enumerate(subtasks, start=1):
        lines.append(f"{i}. {s}")
    lines.append("\n" + "=" * 80)
    lines.append("\nStep-by-step execution:\n")
    for step in steps:
        lines.append(f"--- Step {step['index']} ---")
        lines.append(f"Subtask: {step['subtask']}")
        lines.append(f"Result: {step['result']}")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Exercise 13: Autonomous agent that decomposes and solves tasks")
    parser.add_argument("task", help="High-level task to decompose and solve")
    parser.add_argument(
        "--model",
        default=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        help="Groq model name (or set GROQ_MODEL)",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Model temperature")
    parser.add_argument(
        "--export-dir",
        default="process_logs",
        help="Folder where the process text file will be saved",
    )
    args = parser.parse_args()

    if not os.getenv("GROQ_API_KEY"):
        print("Missing GROQ_API_KEY. Set it in your environment or create a .env file.", file=sys.stderr)
        return 2

    llm = ChatGroq(model=args.model, temperature=args.temperature)

    print(f"Task: {args.task}")
    subtasks = decompose_task(args.task, llm)
    if not subtasks:
        print("Failed to decompose task.", file=sys.stderr)
        return 1

    print(f"Decomposed into {len(subtasks)} subtask(s).")
    steps: list[dict] = []
    for idx, subtask in enumerate(subtasks, start=1):
        print(f"Executing step {idx}/{len(subtasks)}: {subtask}")
        result = execute_subtask(subtask, llm)
        print(f"Result: {result}\n")
        steps.append({"index": idx, "subtask": subtask, "result": result})

    # Export
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"task_process_{timestamp}.txt"
    export_path = export_dir / filename
    export_process(args.task, subtasks, steps, export_path)
    print(f"Process exported to: {export_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
