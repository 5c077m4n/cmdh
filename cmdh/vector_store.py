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


def feed_vector_store(store: Chroma) -> None:
    repo_path = CHROMA_PERSIST_DIR.joinpath("data_repos").joinpath("tldr")
    if repo_path.exists():
        repo = Repo(repo_path)
        repo.remotes.origin.pull()
    else:
        repo = Repo.clone_from("https://github.com/tldr-pages/tldr", to_path=repo_path)

    branch = repo.head.reference
    loader = GitLoader(
        repo_path=repo_path.__str__(),
        branch=branch.__str__(),
        file_filter=lambda file_path: (
            "/pages/" in file_path and file_path.endswith(".md")
        ),
    )
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    all_splits = text_splitter.split_documents(data)
    store.add_documents(all_splits)


def init_vector_store() -> Chroma:
    ollama_embedding = OllamaEmbeddings(model="mistral:latest")
    store = Chroma(
        persist_directory=CHROMA_PERSIST_DIR.__str__(),
        embedding_function=ollama_embedding,
        collection_name="tldr_pages",
    )
    if store._collection.count() == 0:
        feed_vector_store(store)

    return store
