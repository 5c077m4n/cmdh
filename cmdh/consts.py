from os import environ
from pathlib import Path

OLLAMA_MODEL = "mistral:latest"
CHROMA_PERSIST_DIR = Path(
    environ.get("XDG_STATE_HOME", environ.get("HOME", "~") + "/.local/state")
).joinpath("cmdh")
