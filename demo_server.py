"""Local showcase server for the phishing detection project."""

from __future__ import annotations

import asyncio
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from app.services.demo_service import DemoService
from app.services.container import build_service_container
from app.utils.config import PROJECT_ROOT
from app.utils.text import suspicious_keyword_hits


STATIC_DIR = PROJECT_ROOT / "frontend"
EXTENSION_DIR = PROJECT_ROOT / "browser_extension"
CONTAINER = build_service_container()

EMAIL_CREDENTIAL_TERMS = {
    "verify",
    "password",
    "login",
    "signin",
    "credential",
    "confirm",
    "security",
    "account",
}


async def analyze_email_payload(payload: dict) -> dict:
    """Analyze an email message without treating the enclosing webmail page as the target page."""

    sender = str(payload.get("sender", "")).strip()
    subject = str(payload.get("subject", "")).strip()
    body = str(payload.get("body", "")).strip()
    links = [str(link).strip() for link in payload.get("links", []) if str(link).strip()]
    mailbox_hint = str(payload.get("mailbox_hint", "")).strip().lower()
    combined_text = "\n".join(part for part in [subject, body] if part).strip()
    if not combined_text:
        raise ValueError("'subject' or 'body' is required for email analysis.")

    nlp = CONTAINER.model_service.analyze_text(combined_text)
    link_reports = []
    for link in links[:5]:
        report = await CONTAINER.url_analyzer.analyze(link)
        link_reports.append(report.as_dict())

    max_link_risk = max((report["final_risk_score"] for report in link_reports), default=0.0)
    avg_link_risk = round(sum(report["final_risk_score"] for report in link_reports) / len(link_reports), 4) if link_reports else 0.0
    keyword_hits = set(suspicious_keyword_hits(combined_text))
    credential_hits = sorted(keyword_hits.intersection(EMAIL_CREDENTIAL_TERMS))
    sender_domain = sender.rsplit("@", 1)[-1].lower() if "@" in sender else ""
    trusted_sender = sender_domain in {"google.com", "microsoft.com", "github.com", "amuse.io"} or sender_domain.endswith(
        (".google.com", ".microsoft.com", ".github.com", ".amuse.io")
    )
    mailbox_spam = "spam" in mailbox_hint

    final_risk = (0.75 * float(nlp["phishing_probability"])) + (0.25 * max_link_risk)
    if trusted_sender and max_link_risk < 0.2:
        final_risk -= 0.25
    if max_link_risk < 0.2 and len(credential_hits) < 2:
        final_risk -= 0.15
    if mailbox_spam:
        final_risk += 0.12
    final_risk = round(max(0.0, min(final_risk, 1.0)), 4)

    should_block = (
        max_link_risk >= 0.6
        or (float(nlp["phishing_probability"]) >= 0.92 and max_link_risk >= 0.35)
        or (len(credential_hits) >= 2 and max_link_risk >= 0.2)
        or (float(nlp["phishing_probability"]) >= 0.98 and not trusted_sender and mailbox_spam)
    )
    decision = "BLOCK" if should_block else "ALLOW"
    explanation = {
        "mode": "email",
        "sender": sender,
        "sender_domain": sender_domain,
        "trusted_sender": trusted_sender,
        "mailbox_hint": mailbox_hint,
        "mailbox_spam": mailbox_spam,
        "nlp": nlp,
        "max_link_risk": round(max_link_risk, 4),
        "average_link_risk": avg_link_risk,
        "credential_keyword_hits": credential_hits,
        "link_reports": link_reports,
        "decision_threshold": "email-specific heuristic",
    }
    prediction_id = CONTAINER.feedback_service.log_prediction(
        text=combined_text,
        url=links[0] if links else None,
        decision=decision,
        risk_score=final_risk,
        explanation=explanation,
    )
    return {
        "final_risk_score": final_risk,
        "decision": decision,
        "explanation": explanation,
        "prediction_id": prediction_id,
    }


class DemoRequestHandler(BaseHTTPRequestHandler):
    """Serve the local demo frontend and JSON endpoints."""

    demo_service = DemoService(
        settings=CONTAINER.settings,
        model_service=CONTAINER.model_service,
        url_analyzer=CONTAINER.url_analyzer,
    )
    analysis_service = CONTAINER.analysis_service

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        super().end_headers()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.end_headers()

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
        if parsed.path == "/extension":
            self._send_file(EXTENSION_DIR / "README.html", "text/html; charset=utf-8")
            return
        if parsed.path == "/api/demo/random-sample":
            try:
                self._send_json(self.demo_service.get_random_sample())
            except FileNotFoundError as exc:
                self._send_json({"detail": str(exc)}, status=404)
            return
        if parsed.path == "/api/demo/metrics":
            try:
                metrics_path = self.demo_service.settings.base_metrics_path
                payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                self._send_json(payload)
            except FileNotFoundError:
                self._send_json({"detail": "Base training metrics not found. Run base training first."}, status=404)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path not in {"/api/demo/infer", "/api/analyze/url", "/api/analyze/full", "/api/analyze/email"}:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            if parsed.path == "/api/demo/infer":
                sample_id = int(payload["sample_id"])
                result = asyncio.run(self.demo_service.infer_sample(sample_id))
            elif parsed.path == "/api/analyze/url":
                url = str(payload.get("url", "")).strip()
                if not url:
                    self._send_json({"detail": "'url' is required."}, status=400)
                    return
                result = asyncio.run(self.analysis_service.analyze_url(url))
            elif parsed.path == "/api/analyze/email":
                result = asyncio.run(analyze_email_payload(payload))
            else:
                text = str(payload.get("text", "")).strip()
                url = str(payload.get("url", "")).strip()
                html_content = payload.get("html_content")
                if not text or not url:
                    self._send_json({"detail": "'text' and 'url' are required."}, status=400)
                    return
                result = asyncio.run(
                    self.analysis_service.analyze_full(
                        text=text,
                        url=url,
                        html_content=str(html_content) if html_content is not None else None,
                    )
                )
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
