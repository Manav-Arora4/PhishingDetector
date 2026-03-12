"""Local showcase server for the phishing detection project."""

from __future__ import annotations

import asyncio
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from app.services.demo_service import DemoService
from app.utils.config import PROJECT_ROOT


STATIC_DIR = PROJECT_ROOT / "frontend"


class DemoRequestHandler(BaseHTTPRequestHandler):
    """Serve the local demo frontend and JSON endpoints."""

    demo_service = DemoService()

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            return
        if parsed.path == "/assets/app.js":
            self._send_file(STATIC_DIR / "app.js", "application/javascript; charset=utf-8")
            return
        if parsed.path == "/assets/styles.css":
            self._send_file(STATIC_DIR / "styles.css", "text/css; charset=utf-8")
            return
        if parsed.path == "/api/demo/random-sample":
            try:
                self._send_json(self.demo_service.get_random_sample())
            except FileNotFoundError as exc:
                self._send_json({"detail": str(exc)}, status=404)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/demo/infer":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            sample_id = int(payload["sample_id"])
            result = asyncio.run(self.demo_service.infer_sample(sample_id))
            self._send_json(result)
        except FileNotFoundError as exc:
            self._send_json({"detail": str(exc)}, status=404)
        except KeyError:
            self._send_json({"detail": "'sample_id' is required."}, status=400)
        except (ValueError, IndexError) as exc:
            self._send_json({"detail": str(exc)}, status=400)


def run_demo_server(host: str = "127.0.0.1", port: int = 8080) -> None:
    """Start the local threaded demo server."""

    server = ThreadingHTTPServer((host, port), DemoRequestHandler)
    print(f"Demo server running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run_demo_server()
