from collections.abc import Mapping, Sequence

import ollama

from cmdh.llm_server import OllamaServer

OLLAMA_MODEL = "mistral:latest"


def embeddings_fn(prompt: str) -> Sequence[float]:
    return ollama.embeddings(model=OLLAMA_MODEL, prompt=prompt)


def main() -> None:
    with OllamaServer():
        ollama_models = ollama.list()["models"]
        if OLLAMA_MODEL not in [model_info["model"] for model_info in ollama_models]:
            print("Pulling requested model...")
            ollama.pull(OLLAMA_MODEL)

        response = ollama.chat(
            model=OLLAMA_MODEL,
            format="json",
            messages=[
                {
                    "role": "system",
                    "content": "Please respond only valid JSONs",
                },
                {
                    "role": "system",
                    "content": "You are a CLI completion tool, only complete the incoming partail command to ba a valid CLI command",
                },
                {"role": "user", "content": "git commit"},
            ],
        )
        if isinstance(response, Mapping):
            print(response["message"]["content"])
        else:
            for chunk in response:
                print(chunk["message"]["content"], end="", flush=True)


if __name__ == "__main__":
    main()
