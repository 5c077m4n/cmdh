import subprocess
from os import environ
from pathlib import Path
from tempfile import NamedTemporaryFile

from langchain.document_loaders.text import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma

CHROMA_PERSIST_DIR = Path(
    environ.get("XDG_STATE_HOME", environ.get("HOME", "~") + "/.local/state")
).joinpath("cmdh")


def init_vector_store() -> Chroma:
    with NamedTemporaryFile() as f:
        subprocess.call(["man", "git"], stdout=f)
        loader = TextLoader(f.name)
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
        all_splits = text_splitter.split_documents(data)

        ollama_embedding = OllamaEmbeddings(model="mistral:latest")
        vector_store = Chroma.from_documents(
            collection_name="man_pages",
            documents=all_splits,
            embedding=ollama_embedding,
            persist_directory=CHROMA_PERSIST_DIR.__str__(),
        )
        return vector_store
