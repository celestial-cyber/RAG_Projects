import os
from pathlib import Path
from typing import List

# Document loading
from PyPDF2 import PdfReader
from docx import Document  # for .docx if needed; not used for PDF example
from bs4 import BeautifulSoup

# Embeddings
from sentence_transformers import SentenceTransformer

# Vector store
import chromadb
from chromadb.config import Settings

def load_text_from_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    text_parts = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text_parts.append(t)
    return "\n".join(text_parts)

def load_text_from_txt(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_text_from_html(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n")

def load_document(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_text_from_pdf(path)
    if ext == ".txt":
        return load_text_from_txt(path)
    if ext in {".html", ".htm"}:
        return load_text_from_html(path)
    # add more loaders as needed
    raise ValueError(f"Unsupported document type: {ext}")

def chunk_text(text: str, chunk_size_tokens: int = 500, overlap_tokens: int = 100) -> List[str]:
    # Simple whitespace-based tokenizer for chunking size guidance
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        j = min(i + chunk_size_tokens, len(words))
        chunks.append(" ".join(words[i:j]))
        i += max(1, chunk_size_tokens - overlap_tokens)
    return chunks

def main(doc_path: str, output_dir: str = "./rag_store", model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    doc_path = Path(doc_path)
    text = load_document(doc_path)
    chunks = chunk_text(text, chunk_size_tokens=500, overlap_tokens=100)

    # Embedding model
    model = SentenceTransformer(model_name)

    # Compute embeddings
    embeddings = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)

    # Setup vector store (Chroma)
    chroma_dir = Path(output_dir)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.Client(Settings(
        chroma_dir=str(chroma_dir / "chromadb"),
        anonymized_telemetry=False
    ))

    collection_name = "rag_local"
    if collection_name in client.list_collections():
        collection = client.get_collection(collection_name)
    else:
        collection = client.create_collection(name=collection_name, metadata={"source": str(doc_path)})

    # Upsert chunks
    docs = chunks
    mets = [{"source": str(doc_path), "chunk_id": i} for i in range(len(chunks))]
    collection.add(
        documents=docs,
        embeddings=embeddings,
        metadatas=mets
    )

    print(f"Indexed {len(chunks)} chunks from {doc_path} into collection '{collection_name}'.")
    print(f"Chroma data stored at: {chroma_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest a document into a local RAG pipeline (PDF → chunks → embeddings → Chroma).")
    parser.add_argument("document", help="Path to the document to ingest (PDF preferred for this script).")
    parser.add_argument("--out", default="./rag_store", help="Output directory for Chroma data.")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model name from sentence-transformers.")
    args = parser.parse_args()
    main(args.document, args.out, args.model)