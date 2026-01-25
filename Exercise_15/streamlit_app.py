import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from difflib import unified_diff
from typing import Any, Dict, List, Optional

import httpx
import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


@dataclass
class Critique:
    overall_score: float
    rubric: Dict[str, float]
    issues: List[str]
    fixes: List[str]
    stop: bool
    comparison_notes: str
    raw: Dict[str, Any]


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("No JSON object found")
    return json.loads(match.group(0))


def serper_search(query: str, *, k: int = 5) -> str:
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return "SERPER_API_KEY not set."

    try:
        with httpx.Client(timeout=15) as client:
            resp = client.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
                json={"q": query, "num": k},
            )
            resp.raise_for_status()
            data = resp.json()
            items = data.get("organic", [])
            if not items:
                return "No results found."
            lines = []
            for i, it in enumerate(items, start=1):
                title = it.get("title", "")
                snippet = it.get("snippet", "")
                link = it.get("link", "")
                lines.append(f"{i}. {title}\n{snippet}\n{link}")
            return "\n\n".join(lines)
    except Exception as e:
        return f"Search error: {e}"


def make_llm(model: str, temperature: float) -> ChatGroq:
    return ChatGroq(model=model, temperature=temperature)


def generate_draft(llm: ChatGroq, task: str, audience: str, style: str, length: str) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a high-quality writing assistant. Produce a strong first draft.",
            ),
            (
                "human",
                "Task: {task}\nAudience: {audience}\nStyle: {style}\nLength: {length}\n\nDraft:",
            ),
        ]
    )
    return (prompt | llm | StrOutputParser()).invoke(
        {"task": task, "audience": audience, "style": style, "length": length}
    )


def critique_draft(
    llm: ChatGroq,
    task: str,
    audience: str,
    style: str,
    length: str,
    draft: str,
    evidence: Optional[str],
    previous_draft: Optional[str],
    previous_critique: Optional[Dict[str, Any]],
) -> Critique:
    evidence_block = evidence if evidence else "(no external evidence provided)"
    previous_draft_block = previous_draft if previous_draft else "(no previous draft)"
    previous_critique_block = (
        json.dumps(previous_critique, indent=2) if previous_critique else "(no previous critique)"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a strict evaluator. Score the draft and propose fixes. "
                "Return ONLY valid JSON. Do not include markdown.",
            ),
            (
                "human",
                "Evaluate the draft for the task and constraints.\n\n"
                "Task: {task}\nAudience: {audience}\nStyle: {style}\nLength: {length}\n\n"
                "Previous draft (for comparison):\n{previous_draft}\n\n"
                "Previous critique (for comparison):\n{previous_critique}\n\n"
                "Draft:\n{draft}\n\n"
                "Optional external evidence (search snippets):\n{evidence}\n\n"
                "Rubric (0-10 each): clarity, completeness, structure, style_fit, factuality.\n"
                "Compute overall_score (0-10) as your best judgment.\n"
                "IMPORTANT: Compare this draft to the previous draft. If it clearly fixed issues or improved quality, the relevant rubric scores and overall_score should increase. If it did not improve (or regressed), decrease scores. If the draft is nearly identical to the previous draft, keep scores similar and explicitly mention lack of change.\n"
                "If evidence contradicts the draft, reduce factuality and list what is unsupported.\n\n"
                "Return JSON with keys:\n"
                "- overall_score: number\n"
                "- rubric: object with keys clarity, completeness, structure, style_fit, factuality\n"
                "- issues: array of strings (what is wrong)\n"
                "- fixes: array of strings (actionable improvements)\n"
                "- stop: boolean (true if overall_score >= 8.5 and no major issues remain)\n"
                "- comparison_notes: string (1-3 sentences describing what changed vs previous)",
            ),
        ]
    )

    raw_text = (prompt | llm | StrOutputParser()).invoke(
        {
            "task": task,
            "audience": audience,
            "style": style,
            "length": length,
            "draft": draft,
            "evidence": evidence_block,
            "previous_draft": previous_draft_block,
            "previous_critique": previous_critique_block,
        }
    )

    obj = _extract_json(raw_text)
    rubric = obj.get("rubric", {}) if isinstance(obj.get("rubric", {}), dict) else {}

    return Critique(
        overall_score=float(obj.get("overall_score", 0)),
        rubric={k: float(v) for k, v in rubric.items() if isinstance(v, (int, float, str))},
        issues=[str(x) for x in obj.get("issues", [])] if isinstance(obj.get("issues", []), list) else [],
        fixes=[str(x) for x in obj.get("fixes", [])] if isinstance(obj.get("fixes", []), list) else [],
        stop=bool(obj.get("stop", False)),
        comparison_notes=str(obj.get("comparison_notes", "")),
        raw=obj,
    )


def revise_draft(
    llm: ChatGroq,
    task: str,
    audience: str,
    style: str,
    length: str,
    draft: str,
    critique: Critique,
) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a rewriting assistant. Improve the draft by applying the evaluator's fixes. "
                "Keep the same task, audience, style, and length. "
                "You must address the most important fixes and make meaningful edits (the revision should not be identical). "
                "Output only the revised draft.",
            ),
            (
                "human",
                "Task: {task}\nAudience: {audience}\nStyle: {style}\nLength: {length}\n\n"
                "Current draft:\n{draft}\n\n"
                "Issues:\n{issues}\n\n"
                "Fixes:\n{fixes}\n\n"
                "Revised draft:",
            ),
        ]
    )

    return (prompt | llm | StrOutputParser()).invoke(
        {
            "task": task,
            "audience": audience,
            "style": style,
            "length": length,
            "draft": draft,
            "issues": "\n".join(f"- {x}" for x in critique.issues) or "(none)",
            "fixes": "\n".join(f"- {x}" for x in critique.fixes) or "(none)",
        }
    )


def diff(a: str, b: str) -> str:
    return "\n".join(unified_diff(a.splitlines(), b.splitlines(), fromfile="before", tofile="after", lineterm=""))


def build_export_text(run: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"Task: {run['task']}")
    lines.append(f"Timestamp: {run['timestamp']}")
    lines.append(f"Iterations: {len(run['iterations'])}")
    lines.append("=" * 80)

    for it in run["iterations"]:
        lines.append(f"\n--- Iteration {it['index']} ---")
        lines.append(f"Overall score: {it['critique']['overall_score']}")
        if it["critique"].get("comparison_notes"):
            lines.append(f"Comparison notes: {it['critique']['comparison_notes']}")
        lines.append("Rubric:")
        for k, v in it["critique"].get("rubric", {}).items():
            lines.append(f"- {k}: {v}")
        lines.append("Issues:")
        for x in it["critique"].get("issues", []):
            lines.append(f"- {x}")
        lines.append("Fixes:")
        for x in it["critique"].get("fixes", []):
            lines.append(f"- {x}")
        lines.append("\nDraft:")
        lines.append(it["draft"])

        if it.get("diff"):
            lines.append("\nDiff (previous -> current):")
            lines.append(it["diff"])

    return "\n".join(lines)


def main() -> None:
    load_dotenv()

    st.set_page_config(page_title="Exercise 15 - Self-Correcting Agent", layout="wide")
    st.title("Exercise 15 — Self-Correcting Agent")

    if not os.getenv("GROQ_API_KEY"):
        st.error("Missing GROQ_API_KEY. Add it to your environment or a .env file.")
        st.stop()

    with st.sidebar:
        st.header("Settings")
        model = st.text_input("Groq model", value=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
        max_iters = st.slider("Max iterations", 1, 6, 3, 1)
        target_score = st.slider("Target score", 1.0, 10.0, 8.5, 0.5)
        enable_search = st.checkbox("Enable evidence search (Serper)", value=False)
        show_raw_evidence = st.checkbox("Show raw evidence", value=False)

    col1, col2 = st.columns([2, 1])
    with col1:
        task = st.text_area("Task / Prompt", height=140, placeholder="e.g., Write a 1-page explanation of transformers for beginners")
        audience = st.text_input("Audience", value="general")
        style = st.text_input("Style", value="clear and concise")
        length = st.selectbox("Length", ["very short", "short", "medium", "long"], index=1)

    with col2:
        st.markdown("### What this app does")
        st.markdown(
            "It generates a draft, critiques it with a scored rubric, revises it, and repeats until it reaches the target score or hits the iteration limit."
        )

    run_button = st.button("Run self-correction")

    if "runs" not in st.session_state:
        st.session_state.runs = []

    if run_button:
        if not task.strip():
            st.warning("Please enter a task.")
            st.stop()

        llm = make_llm(model, temperature)

        evidence = None
        if enable_search:
            if not os.getenv("SERPER_API_KEY"):
                st.warning("SERPER_API_KEY is not set. Evidence search will be skipped.")
            else:
                with st.spinner("Searching for external evidence..."):
                    evidence = serper_search(task)

        with st.spinner("Generating initial draft..."):
            current = generate_draft(llm, task, audience, style, length)

        run: Dict[str, Any] = {
            "task": task,
            "audience": audience,
            "style": style,
            "length": length,
            "model": model,
            "temperature": temperature,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "evidence": evidence,
            "iterations": [],
        }

        prev = ""
        last_critique_raw: Optional[Dict[str, Any]] = None
        for i in range(1, max_iters + 1):
            with st.spinner(f"Critiquing iteration {i}..."):
                critique = critique_draft(
                    llm,
                    task,
                    audience,
                    style,
                    length,
                    current,
                    evidence,
                    previous_draft=prev if prev else None,
                    previous_critique=last_critique_raw,
                )

            last_critique_raw = critique.raw

            it: Dict[str, Any] = {
                "index": i,
                "draft": current,
                "critique": {
                    "overall_score": critique.overall_score,
                    "rubric": critique.raw.get("rubric", {}),
                    "issues": critique.issues,
                    "fixes": critique.fixes,
                    "stop": critique.stop,
                    "comparison_notes": critique.comparison_notes,
                },
                "diff": diff(prev, current) if prev else "",
            }
            run["iterations"].append(it)

            if critique.stop or critique.overall_score >= target_score:
                break

            prev = current
            with st.spinner(f"Revising iteration {i} -> {i + 1}..."):
                current = revise_draft(llm, task, audience, style, length, current, critique)

        st.session_state.runs.insert(0, run)

    if st.session_state.runs:
        st.markdown("---")
        st.subheader("Latest run")
        run = st.session_state.runs[0]

        if run.get("evidence") and (show_raw_evidence or enable_search):
            with st.expander("External evidence (search snippets)", expanded=False):
                st.text(run["evidence"])

        scores = [it["critique"]["overall_score"] for it in run["iterations"]]
        st.line_chart(scores)

        for it in run["iterations"]:
            with st.expander(f"Iteration {it['index']} | score={it['critique']['overall_score']}", expanded=it["index"] == 1):
                if it["critique"].get("comparison_notes"):
                    st.markdown("**Comparison notes**")
                    st.write(it["critique"]["comparison_notes"])
                st.markdown("**Rubric**")
                st.json(it["critique"]["rubric"]) 
                st.markdown("**Issues**")
                st.write("\n".join(f"- {x}" for x in it["critique"].get("issues", [])) or "(none)")
                st.markdown("**Fixes**")
                st.write("\n".join(f"- {x}" for x in it["critique"].get("fixes", [])) or "(none)")
                st.markdown("**Draft**")
                st.text_area("", it["draft"], height=220, key=f"draft_{it['index']}")

                if it.get("diff"):
                    st.markdown("**Diff (previous -> current)**")
                    st.code(it["diff"], language="diff")

        export_text = build_export_text(run)
        export_json = json.dumps(run, indent=2)

        st.download_button(
            label="Download log (TXT)",
            data=export_text,
            file_name="self_correcting_log.txt",
            mime="text/plain",
        )
        st.download_button(
            label="Download log (JSON)",
            data=export_json,
            file_name="self_correcting_log.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
