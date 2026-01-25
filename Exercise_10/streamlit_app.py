import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma


def load_uploaded_file_to_docs(file_path: Path) -> list[Document]:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(file_path)).load()
    if suffix in {".txt", ".md"}:
        return TextLoader(str(file_path), encoding="utf-8").load()
    raise ValueError("Unsupported file type. Upload a .pdf, .txt, or .md")


def format_docs(docs: list[Document]) -> str:
    parts: list[str] = []
    for i, d in enumerate(docs, start=1):
        source = d.metadata.get("source", "uploaded")
        page = d.metadata.get("page", "")
        tag = f"[Source {i}: {source}]"
        if page != "":
            tag = f"[Source {i}: {source} | page={page}]"
        parts.append(f"{tag}\n{d.page_content}")
    return "\n\n".join(parts)


def build_rag_chain(retriever, llm: ChatGroq):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a retrieval-augmented assistant. Use ONLY the provided context to answer. "
                "If the answer is not in the context, say: 'I don't know based on the provided document.'",
            ),
            (
                "human",
                "Question: {question}\n\nContext:\n{context}\n\nAnswer:",
            ),
        ]
    )

    question_runnable = RunnableLambda(lambda x: x["question"])

    chain = (
        {
            "context": question_runnable | retriever | RunnableLambda(format_docs),
            "question": question_runnable,
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def export_chunks_and_vectors(export_dir: Path, chunks: list[Document], embeddings: FastEmbedEmbeddings) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = export_dir / "rag_chunks.txt"
    vectors_path = export_dir / "rag_vectors.txt"

    with chunks_path.open("w", encoding="utf-8") as f:
        for i, c in enumerate(chunks, start=1):
            source = c.metadata.get("source", "uploaded")
            page = c.metadata.get("page", "")
            header = f"CHUNK {i} | source={source}"
            if page != "":
                header += f" | page={page}"
            f.write(header + "\n")
            f.write(c.page_content.strip() + "\n")
            f.write("\n" + ("-" * 80) + "\n\n")

    vectors = embeddings.embed_documents([c.page_content for c in chunks])
    with vectors_path.open("w", encoding="utf-8") as f:
        for i, v in enumerate(vectors, start=1):
            f.write(f"CHUNK {i} | dim={len(v)}\n")
            f.write(",".join(f"{x:.6f}" for x in v) + "\n\n")


def main():
    load_dotenv()

    st.set_page_config(page_title="Exercise 10 - RAG Chat", layout="wide")
    st.title("Exercise 10 — RAG Chat (Groq + Custom Docs)")

    if not os.getenv("GROQ_API_KEY"):
        st.error("Missing GROQ_API_KEY. Add it to your environment or a .env file.")
        st.stop()

    with st.sidebar:
        st.header("Settings")
        model = st.text_input("Groq model", value=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
        k = st.slider("Top-k retrieval", min_value=1, max_value=10, value=4, step=1)
        chunk_size = st.number_input("Chunk size", min_value=200, max_value=2000, value=900, step=50)
        chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=500, value=150, step=25)
        enable_export = st.checkbox("Export chunks + vectors", value=False)

    upload = st.file_uploader("Upload a PDF/TXT/MD document", type=["pdf", "txt", "md"])

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if upload is not None:
        uploads_dir = Path(__file__).parent / ".uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        saved_path = uploads_dir / upload.name
        saved_path.write_bytes(upload.getbuffer())

        try:
            raw_docs = load_uploaded_file_to_docs(saved_path)
        except Exception as e:
            st.error(str(e))
            st.stop()

        splitter = RecursiveCharacterTextSplitter(chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap))
        chunks = splitter.split_documents(raw_docs)

        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        vectorstore = Chroma.from_documents(chunks, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": int(k)})

        if enable_export:
            export_dir = Path(__file__).parent / "exports"
            export_chunks_and_vectors(export_dir, chunks, embeddings)
            st.sidebar.success(f"Exported to: {export_dir}")

        llm = ChatGroq(model=model, temperature=float(temperature))
        st.session_state.chain = build_rag_chain(retriever, llm)
        st.session_state.doc_name = upload.name

    doc_name = st.session_state.get("doc_name")
    if doc_name:
        st.caption(f"Document loaded: {doc_name}")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Ask a question about the uploaded document")

    if user_q:
        if "chain" not in st.session_state:
            st.error("Upload a document first.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        with st.chat_message("assistant"):
            answer = st.session_state.chain.invoke({"question": user_q})
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
