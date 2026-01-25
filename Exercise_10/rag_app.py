import argparse
import os
import sys
from pathlib import Path

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


def _load_single_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(path)).load()
    if suffix in {".txt", ".md"}:
        return TextLoader(str(path), encoding="utf-8").load()
    raise ValueError(f"Unsupported file type: {suffix}. Use .pdf, .txt, or .md")


def load_documents(docs_path: str) -> list[Document]:
    path = Path(docs_path)

    if not path.exists():
        raise FileNotFoundError(f"Docs path not found: {docs_path}")

    if path.is_file():
        return _load_single_file(path)

    docs: list[Document] = []
    for p in sorted(path.rglob("*")):
        if not p.is_file():
            continue
        try:
            docs.extend(_load_single_file(p))
        except ValueError:
            continue
    return docs


def export_chunks_and_vectors(export_dir: Path, chunks: list[Document], embeddings: FastEmbedEmbeddings) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = export_dir / "rag_chunks.txt"
    vectors_path = export_dir / "rag_vectors.txt"

    with chunks_path.open("w", encoding="utf-8") as f:
        for i, c in enumerate(chunks, start=1):
            source = c.metadata.get("source", "unknown")
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


def format_docs(docs: list[Document]) -> str:
    parts: list[str] = []
    for i, d in enumerate(docs, start=1):
        source = d.metadata.get("source", "unknown")
        parts.append(f"[Source {i}: {source}]\n{d.page_content}")
    return "\n\n".join(parts)


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Exercise 10: Simple RAG with custom documents")
    parser.add_argument("--docs", required=True, help="Path to a text file or a folder of text files")
    parser.add_argument("question", help="Question to ask over your custom documents")
    parser.add_argument("--k", type=int, default=4, help="Number of chunks to retrieve")
    parser.add_argument("--chunk-size", type=int, default=900, help="Chunk size for splitting")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="Chunk overlap for splitting")
    parser.add_argument(
        "--export-dir",
        default=None,
        help="Optional folder to export chunks and embedding vectors as text files",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        help="Groq model name (or set GROQ_MODEL)",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Model temperature")
    args = parser.parse_args()

    if not os.getenv("GROQ_API_KEY"):
        print("Missing GROQ_API_KEY. Set it in your environment or create a .env file.", file=sys.stderr)
        return 2

    raw_docs = load_documents(args.docs)
    if not raw_docs:
        print("No documents were loaded. Provide a .txt/.md file or a folder containing text files.", file=sys.stderr)
        return 2

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    chunks = splitter.split_documents(raw_docs)

    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    if args.export_dir:
        export_chunks_and_vectors(Path(args.export_dir), chunks, embeddings)

    vectorstore = Chroma.from_documents(chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": args.k})

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a retrieval-augmented assistant. Use ONLY the provided context to answer. "
                "If the answer is not in the context, say: 'I don't know based on the provided documents.'",
            ),
            (
                "human",
                "Question: {question}\n\nContext:\n{context}\n\nAnswer:",
            ),
        ]
    )

    llm = ChatGroq(model=args.model, temperature=args.temperature)

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

    answer = chain.invoke({"question": args.question})
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
