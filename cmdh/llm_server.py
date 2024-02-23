import os
import time
from subprocess import DEVNULL, Popen


class OllamaServer:
    url = "http://localhost:11435"

    def __init__(self):
        env = os.environ.copy()
        env["OLLAMA_HOST"] = OllamaServer.url

        self.ollama_serve_proc = Popen(
            ["ollama", "serve"], stdout=DEVNULL, stderr=DEVNULL, env=env
        )

    def __enter__(self):
        time.sleep(0.75)
        return self.ollama_serve_proc

    def __exit__(self, *_):
        return self.ollama_serve_proc.kill()
