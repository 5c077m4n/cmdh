import os
import subprocess


class OllamaServer:
    url = "http://localhost:11435"

    def __init__(self):
        env = os.environ.copy()
        env["OLLAMA_HOST"] = OllamaServer.url

        self.ollama_serve_proc = subprocess.Popen(["ollama", "serve"], env=env)
        self.ollama_serve_proc.communicate()

    def __enter__(self):
        return self.ollama_serve_proc

    def __exit__(self, *_):
        return self.ollama_serve_proc.kill()
