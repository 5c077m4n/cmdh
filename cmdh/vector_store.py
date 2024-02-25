import subprocess

from langchain.document_loaders.text import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma


def init_vector_store() -> Chroma:
    with open("/tmp/mangit.txt", "w+") as f:
        subprocess.call(["man", "git"], stdout=f)

    loader = TextLoader("/tmp/mangit.txt")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    all_splits = text_splitter.split_documents(data)

    ollama_embedding = OllamaEmbeddings(model="mistral:latest")
    vector_store = Chroma.from_documents(
        documents=all_splits,
        embedding=ollama_embedding,
        persist_directory="/tmp/chroma/",
    )
    return vector_store
