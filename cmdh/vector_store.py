from datetime import datetime
from os import environ
from pathlib import Path

from git import Repo
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.git import GitLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma

CHROMA_PERSIST_DIR = Path(
    environ.get("XDG_STATE_HOME", environ.get("HOME", "~") + "/.local/state")
).joinpath("cmdh")


def init_vector_store() -> Chroma:
    repo_path = CHROMA_PERSIST_DIR.joinpath("data_repos").joinpath("tldr").__str__()
    repo = Repo.clone_from("https://github.com/tldr-pages/tldr", to_path=repo_path)
    branch = repo.head.reference

    print(f"[{datetime.now()}] Loading git repo...")
    loader = GitLoader(
        repo_path,
        branch=branch.__str__(),
        file_filter=lambda file_path: (
            "/pages/" in file_path and file_path.endswith(".md")
        ),
    )
    print(f"[{datetime.now()}] Done loading git repo")
    print(f"[{datetime.now()}] Starting data load...")
    data = loader.load()
    print(f"[{datetime.now()}] Done loading data")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    all_splits = text_splitter.split_documents(data)

    ollama_embedding = OllamaEmbeddings(model="mistral:latest")
    print(f"[{datetime.now()}] Creating vector store...")
    vector_store = Chroma.from_documents(
        collection_name="tldr_pages",
        documents=all_splits,
        embedding=ollama_embedding,
        persist_directory=CHROMA_PERSIST_DIR.__str__(),
    )
    print(f"[{datetime.now()}] Finished creating vector store")
    return vector_store
