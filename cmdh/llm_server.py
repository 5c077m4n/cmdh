import time
from subprocess import DEVNULL, Popen


class OllamaServer:
    url = "http://localhost:11435"

    def __init__(self):
        self.ollama_serve_proc = Popen(
            ["ollama", "serve"], stdout=DEVNULL, stderr=DEVNULL
        )

    def __enter__(self):
        time.sleep(0.75)
        return self.ollama_serve_proc

    def __exit__(self, *_):
        return self.ollama_serve_proc.kill()
