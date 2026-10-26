"""Answer questions about Brainlox technical courses.

The course-category page is split into chunks and ranked by word overlap.
A model is called only when GROQ_API_KEY or HF_API_TOKEN is set. With no key,
the reply is the best matching excerpt.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from html.parser import HTMLParser

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

PAGE = "https://brainlox.com/courses/category/technical"
MAX_QUERY = 500
_WORD = re.compile(r"[a-z0-9]+")

_chunks: list[dict] | None = None
_hits: dict[str, list[float]] = {}


class _Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self._skip = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"} and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if self._skip:
            return
        text = " ".join(data.split())
        if text:
            self.parts.append(text)


def page_text(html: str) -> str:
    parser = _Text()
    parser.feed(html)
    return "\n".join(parser.parts)


def chunk_text(text: str, source: str) -> list[dict]:
    words = text.split()
    chunks = []
    size, step = 120, 90
    for i in range(0, max(len(words), 1), step):
        piece = " ".join(words[i : i + size]).strip()
        if len(piece) > 40:
            chunks.append({"text": piece, "source": source})
        if i + size >= len(words):
            break
    return chunks


def tokens(text: str) -> set[str]:
    return {word for word in _WORD.findall(text.lower()) if len(word) > 2}


def rank(query: str, chunks: list[dict]) -> list[dict]:
    wanted = tokens(query)
    scored = [(len(wanted & tokens(chunk["text"])), chunk) for chunk in chunks]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for score, chunk in scored if score][:3]


def load_chunks() -> list[dict]:
    import requests

    response = requests.get(
        PAGE, timeout=20, headers={"User-Agent": "Brainlox-Chatbot/1.0"}
    )
    response.raise_for_status()
    return chunk_text(page_text(response.text), PAGE)


def _post(url: str, payload: dict, headers: dict) -> dict | list | None:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def model_answer(query: str, excerpts: list[dict]) -> str | None:
    context = "\n\n".join(item["text"] for item in excerpts)
    prompt = (
        "Answer using only the excerpts. If they do not contain the answer, say so.\n\n"
        f"Excerpts:\n{context}\n\nQuestion: {query}"
    )
    groq = os.getenv("GROQ_API_KEY")
    if groq:
        data = _post(
            "https://api.groq.com/openai/v1/chat/completions",
            {
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            },
            {"Authorization": f"Bearer {groq}"},
        )
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (TypeError, KeyError, IndexError):
            return None
    token = os.getenv("HF_API_TOKEN")
    if token:
        data = _post(
            "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
            {"inputs": prompt, "parameters": {"max_new_tokens": 300, "temperature": 0.2}},
            {"Authorization": f"Bearer {token}"},
        )
        if isinstance(data, list) and data and "generated_text" in data[0]:
            return data[0]["generated_text"].split(prompt)[-1].strip() or None
    return None


def reply(query: str, chunks: list[dict]) -> dict:
    hits = rank(query, chunks)
    if not hits:
        return {
            "response": "Nothing on the technical-courses page matches that.",
            "sources": [],
            "mode": "retrieve",
        }
    text = model_answer(query, hits)
    return {
        "response": text or hits[0]["text"],
        "sources": [hit["source"] for hit in hits],
        "mode": "model" if text else "retrieve",
    }


def _limited(ip: str) -> bool:
    now = time.time()
    recent = [stamp for stamp in _hits.get(ip, []) if now - stamp < 60]
    if len(recent) >= 5:
        _hits[ip] = recent
        return True
    recent.append(now)
    _hits[ip] = recent
    return False


def chunks() -> list[dict]:
    global _chunks
    if _chunks is None:
        _chunks = load_chunks()
    return _chunks


def create_app():
    from flask import Flask, request, send_from_directory

    app = Flask(__name__)

    @app.route("/")
    def home():
        return "Brainlox course Q&A. Open /ui to chat."

    @app.route("/ui")
    def ui():
        here = os.path.dirname(os.path.abspath(__file__))
        return send_from_directory(here, "chat.html")

    @app.post("/chat")
    def chat():
        if _limited(request.remote_addr or "local"):
            return {"error": "Too many questions. Wait a minute."}, 429
        data = request.get_json(silent=True) or {}
        query = str(data.get("query", "")).strip()
        if not query or len(query) > MAX_QUERY:
            return {"error": "Ask a shorter question."}, 400
        try:
            return reply(query, chunks())
        except Exception:
            return {"error": "The courses page could not be read."}, 502

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=False)
