"""
serve_unity.py
==============
Servidor HTTP simple para el build de Unity WebGL.
Corre en paralelo con Streamlit y lipsync_server.py

Uso:
    python serve_unity.py --build ./unity_build --port 8080
"""

import argparse
import http.server
import socketserver
import os

class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Handler que agrega headers CORS y soporte para archivos .br y .gz de Unity."""

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        # Headers para archivos comprimidos de Unity
        if self.path.endswith(".br"):
            self.send_header("Content-Encoding", "br")
        elif self.path.endswith(".gz"):
            self.send_header("Content-Encoding", "gzip")
        super().end_headers()

    def guess_type(self, path):
        # Unity WebGL usa extensiones dobles (.js.br, .wasm.br, etc.)
        if path.endswith(".js") or path.endswith(".js.br") or path.endswith(".js.gz"):
            return "application/javascript"
        if path.endswith(".wasm") or path.endswith(".wasm.br") or path.endswith(".wasm.gz"):
            return "application/wasm"
        if path.endswith(".data") or path.endswith(".data.br") or path.endswith(".data.gz"):
            return "application/octet-stream"
        return super().guess_type(path)

    def log_message(self, format, *args):
        pass  # silenciar logs de cada request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", default="./unity_build", help="Carpeta del build de Unity WebGL")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    os.chdir(args.build)

    with socketserver.TCPServer(("", args.port), CORSRequestHandler) as httpd:
        print(f"🎮  Unity WebGL disponible en http://localhost:{args.port}")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
