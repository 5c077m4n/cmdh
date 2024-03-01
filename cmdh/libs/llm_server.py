import time
from subprocess import DEVNULL, Popen


class OllamaServer:
    def __init__(self):
        self.ollama_serve_proc = Popen(
            ["ollama", "serve"], stdout=DEVNULL, stderr=DEVNULL
        )

    def __enter__(self):
        time.sleep(1)
        return self.ollama_serve_proc

    def __exit__(self, *_):
        return self.ollama_serve_proc.kill()
