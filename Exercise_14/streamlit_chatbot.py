import os
import httpx
from typing import List

import streamlit as st
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
    uncertain_phrases = [
        "i don't know",
        "i'm not sure",
        "i do not have",
        "i cannot",
        "i am not aware",
        "i don't have real-time",
        "i cannot browse",
    ]
    return any(phrase in answer.lower() for phrase in uncertain_phrases)


def format_message(msg: BaseMessage) -> str:
    if isinstance(msg, HumanMessage):
        return f"You: {msg.content}"
    elif isinstance(msg, AIMessage):
        return f"Bot: {msg.content}"
    return str(msg.content)


def main():
    load_dotenv()

    if not os.getenv("GROQ_API_KEY"):
        st.error("Missing GROQ_API_KEY. Add it to your environment or a .env file.")
        st.stop()

    st.set_page_config(page_title="Chatbot with Memory & Fallback", layout="wide")
    st.title("Exercise 14 — Chatbot with Memory and Fallback")

    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    verbose = st.sidebar.checkbox("Verbose (show fallback reasoning)", value=False)

    llm = ChatGroq(model=model, temperature=temperature)

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
    fallback_prompt = ChatPromptTemplate.from_template(
        "You are a helpful chatbot. Use the following web search results to answer the user's question.\n"
        "If the results are empty or irrelevant, say 'I could not find reliable information.'\n\n"
        "Question: {question}\n\nSearch results:\n{results}\n\nAnswer:"
    )

    chain_with_memory = prompt | llm | StrOutputParser()
    fallback_chain = fallback_prompt | llm | StrOutputParser()

    # Initialize session state
    if "history" not in st.session_state:
        st.session_state.history: List[BaseMessage] = []

    # Sidebar: show memory
    st.sidebar.header("Conversation Memory")
    if st.session_state.history:
        for i, msg in enumerate(st.session_state.history, start=1):
            st.sidebar.text(f"{i}. {format_message(msg)}")
    else:
        st.sidebar.info("No messages yet.")

    # Chat UI
    user_input = st.chat_input("Your message")
    if user_input:
        # First attempt: answer with memory
        answer = chain_with_memory.invoke({"input": user_input, "history": st.session_state.history})
        fallback_used = False
        if should_fallback(answer):
            if verbose:
                st.sidebar.info("Uncertain, falling back to web search...")
            results = web_search(user_input)
            if verbose:
                st.sidebar.text_area("Search Results", results, height=200)
            answer = fallback_chain.invoke({"question": user_input, "results": results})
            fallback_used = True

        # Store in memory
        st.session_state.history.extend([HumanMessage(content=user_input), AIMessage(content=answer)])

        # Display messages
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            st.markdown(answer)
        if fallback_used and verbose:
            st.caption("Fallback: web search was used to answer.")

    # Render past messages
    for msg in st.session_state.history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)


if __name__ == "__main__":
    main()
