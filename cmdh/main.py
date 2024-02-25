from langchain.chains import RetrievalQA
from langchain_community.llms.ollama import Ollama

from cmdh.llm_server import OllamaServer
from cmdh.vector_store import init_vector_store

OLLAMA_MODEL = "mistral:latest"


def main() -> None:
    with OllamaServer():
        ollama = Ollama(model=OLLAMA_MODEL)
        vector_store = init_vector_store()
        qa_chain = RetrievalQA.from_chain_type(
            ollama, retriever=vector_store.as_retriever()
        )
        response = qa_chain.invoke(
            {
                "query": "Please complete the following command for me with all of its flags and subcommands: `git commit`"
            }
        )

        print(response)


if __name__ == "__main__":
    main()
