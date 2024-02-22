from langchain_community.llms.ollama import Ollama
from ollama import pull

from cmdh.llm_server import OllamaServer


def main():
    with OllamaServer():
        pull("mistral")

        client = Ollama(model="mistral")
        print(client.invoke("How are you?"))


if __name__ == "__main__":
    main()
