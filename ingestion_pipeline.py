# Upgrade pip first
!pip install --upgrade pip

# Core orchestration + utilities
!pip install langchain

# Vector DB (Chroma). Replace with faiss-cpu if you prefer FAISS:
!pip install chromadb
# OR (alternative) uncomment to use FAISS instead of Chroma:
# !pip install faiss-cpu

# Embeddings / LLM clients
!pip install openai
!pip install sentence-transformers
!pip install transformers

# Text splitting / token helpers
!pip install tiktoken

# Document loaders / parsers
!pip install pdfplumber python-docx beautifulsoup4
# Optional broader extractor (uncomment if needed)
# !pip install tika

# Optional extras (connectors, community utils)
!pip install langchain-community

# Clean up and show installed versions (optional)
!python - <<'PY'
import importlib, pkgutil, sys
pkgs = ["langchain","chromadb","openai","sentence_transformers","transformers","tiktoken","pdfplumber","python_docx","bs4"]
for p in pkgs:
    try:
        mod = importlib.import_module(p)
        print(p, "OK")
    except Exception as e:
        print(p, "NOT INSTALLED or import name differs;", e)
#pip install langchain langchain-community langchain_text_splitters langchain_openai langchain_chroma
import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma #vector database - it can be host locally 
from dotenv import load_dotenv

load_dotenv()

def main():
    print("Main function")


if __name__=="__main__":
    main()

#load the document 

if len()





