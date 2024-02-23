from collections.abc import Mapping

import ollama

from cmdh.llm_server import OllamaServer

OLLAMA_MODEL = "mistral:latest"


def embeddings_fn(prompt: str):
    return ollama.embeddings(model=OLLAMA_MODEL, prompt=prompt)


def main():
    with OllamaServer():
        ollama_models = ollama.list()["models"]
        if OLLAMA_MODEL not in [model_info["model"] for model_info in ollama_models]:
            print("Pulling requested model...")
            ollama.pull(OLLAMA_MODEL)

        print(embeddings_fn("They sky is blue because of rayleigh scattering"))
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": "How are you?"}],
            format="json",
        )
        if isinstance(response, Mapping):
            print(response["message"]["content"])
        else:
            for chunk in response:
                print(chunk["message"]["content"], end="", flush=True)


if __name__ == "__main__":
    main()
