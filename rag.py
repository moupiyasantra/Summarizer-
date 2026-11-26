"""Core logic: chunking, embeddings, vector search (RAG), local LLM."""
import json
import os
import re
import zlib
from functools import lru_cache

import numpy as np
import requests

# 127.0.0.1 (not "localhost") avoids IPv6 resolution problems on Windows
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "llama3.2")  # any model you have pulled in Ollama
TOP_K = 4        # passages handed to the LLM as context (RAG)


class LLMError(RuntimeError):
    """Friendly, user-facing error from the language model."""


# --------------------------------------------------------------------------
# 1. Chunking (poems -> stanzas, prose -> paragraphs)
# --------------------------------------------------------------------------
def _split_long_block(block: str, max_words: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", block)
    pieces, cur, n = [], [], 0
    for s in sentences:
        w = len(s.split())
        if cur and n + w > max_words:
            pieces.append(" ".join(cur))
            cur, n = [], 0
        cur.append(s)
        n += w
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def chunk_text(text: str, max_words: int = 180) -> list[str]:
    """Split on blank lines (stanzas / paragraphs), merging small ones."""
    text = text.replace("\r\n", "\n")
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    chunks, cur, n = [], [], 0
    for b in blocks:
        parts = _split_long_block(b, max_words) if len(b.split()) > max_words else [b]
        for p in parts:
            w = len(p.split())
            if cur and n + w > max_words:
                chunks.append("\n\n".join(cur))
                cur, n = [], 0
            cur.append(p)
            n += w
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


# --------------------------------------------------------------------------
# 2. Embeddings (MiniLM, with a no-dependency fallback) + vector store
# --------------------------------------------------------------------------
_HASH_DIM = 2048


def _hash_embed(texts: list[str]) -> np.ndarray:
    """Lightweight keyword-based fallback if sentence-transformers is unavailable."""
    out = np.zeros((len(texts), _HASH_DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok in re.findall(r"[a-z0-9']{3,}", t.lower()):
            out[i, zlib.crc32(tok.encode()) % _HASH_DIM] += 1
    out = np.log1p(out)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return out / norms


@lru_cache(maxsize=1)
def _load_embedder():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(EMBED_MODEL)
    except Exception:
        return None  # fall back to keyword embeddings


def embedder_name() -> str:
    return "semantic (MiniLM)" if _load_embedder() is not None else "keyword fallback"


def embed(texts: list[str]) -> np.ndarray:
    m = _load_embedder()
    if m is None:
        return _hash_embed(texts)
    return m.encode(texts, normalize_embeddings=True, show_progress_bar=False)


class VectorStore:
    def __init__(self):
        self.items: list[dict] = []
        self.vecs: np.ndarray | None = None

    def __len__(self):
        return len(self.items)

    def add(self, texts: list[str], source: str):
        if not texts:
            return
        v = embed(texts)
        self.items += [{"text": t, "source": source} for t in texts]
        self.vecs = v if self.vecs is None else np.vstack([self.vecs, v])

    def search(self, query: str, k: int = 4) -> list[dict]:
        if not self.items:
            return []
        scores = self.vecs @ embed([query])[0]
        top = np.argsort(-scores)[:k]
        return [{**self.items[i], "score": float(scores[i])} for i in top]


# --------------------------------------------------------------------------
# 3. Local open-source LLM via Ollama
# --------------------------------------------------------------------------
def chat_stream(messages: list[dict], model: str, temperature: float = 0.3):
    payload = {"model": model, "messages": messages, "stream": True,
               "options": {"temperature": temperature}}
    try:
        with requests.post(f"{OLLAMA_URL}/api/chat", json=payload, stream=True,
                           timeout=(5, 600)) as r:
            if r.status_code == 404:
                raise LLMError(f"Model '{model}' isn't installed. "
                               f"Open a terminal and run:  ollama pull {model}")
            if r.status_code != 200:
                raise LLMError(f"Ollama error {r.status_code}: {r.text[:300]}")
            got = False
            for line in r.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if "error" in data:
                    raise LLMError(f"Ollama: {data['error']}")
                tok = data.get("message", {}).get("content", "")
                if tok:
                    got = True
                    yield tok
            if not got:
                raise LLMError("The model returned an empty response. Try another model.")
    except requests.exceptions.ConnectionError:
        raise LLMError("Can't reach Ollama. Open the Ollama app (or run `ollama serve`) "
                       "and try again.") from None
    except requests.exceptions.ReadTimeout:
        raise LLMError("The model took too long to respond. Try a smaller model "
                       "(e.g. llama3.2) or a shorter text.") from None


def chat(messages, model, temperature=0.3) -> str:
    return "".join(chat_stream(messages, model, temperature))


SYSTEM = ("You are a warm, insightful literature assistant. You summarize and discuss "
          "prose and poetry clearly and simply. Use ONLY the provided context when "
          "given; if the answer is not there, say so. Never invent quotes.")

SUMMARY_STYLES = {
    "Short summary": "Write a clear summary in 3-5 sentences.",
    "Detailed summary": "Write a detailed summary with a short paragraph per major section.",
    "Themes & tone": "List the main themes, mood/tone, and the central message.",
    "Literary devices": "Identify notable literary devices (imagery, metaphor, rhyme, "
                        "structure, symbolism) with brief examples from the text.",
    "Explain in easy words": "Explain what this text is about in very simple language.",
}


# --------------------------------------------------------------------------
# 4. Summarization (map-reduce for long texts)
# --------------------------------------------------------------------------
def summarize(text: str, style: str, kind: str, model: str):
    """Yields streamed output. Long texts are first condensed chunk-by-chunk."""
    instruction = SUMMARY_STYLES[style]
    body = text
    if len(text.split()) > 2000:  # map step
        partials = [chat([{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": f"Summarize this part of a {kind} "
                           f"faithfully in 4-6 sentences:\n\n{c}"}], model)
                    for c in chunk_text(text, max_words=700)]
        body = "\n\n".join(f"Part {i+1}: {p}" for i, p in enumerate(partials))
        kind = f"{kind} (condensed notes)"
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Here is a {kind}:\n\n{body}\n\n{instruction}"}]
    yield from chat_stream(msgs, model)


# --------------------------------------------------------------------------
# 5. RAG answering
# --------------------------------------------------------------------------
def format_context(hits: list[dict]) -> str:
    return "\n\n".join(f"[{i+1}] (source: {h['source']})\n{h['text']}"
                       for i, h in enumerate(hits))


def rag_answer(question: str, hits: list[dict], model: str, history=None):
    msgs = [{"role": "system", "content": SYSTEM}]
    msgs += (history or [])[-6:]
    msgs.append({"role": "user", "content":
                 f"Context:\n{format_context(hits)}\n\nQuestion: {question}\n\n"
                 "Answer using the context. Cite sources like [1], [2]."})
    yield from chat_stream(msgs, model)
