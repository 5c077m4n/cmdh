from langchain_community.llms.ollama import Ollama

from cmdh.llm_server import OllamaServer
from cmdh.pull import fetch_model


def main():
    with OllamaServer():
        fetch_model()
        client = Ollama(base_url=OllamaServer.url)
        print(client("How are you?"))


if __name__ == "__main__":
    main()
