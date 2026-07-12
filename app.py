import json
import os
import re
import threading
import time
import uuid
import hashlib
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, redirect, render_template, request, url_for
from qdrant_client import QdrantClient, models

# Optional heavy imports are loaded lazily.
BGEM3FlagModel = None
FlagReranker = None

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env", override=True)
load_dotenv(override=True)

app = Flask(__name__)

IS_SERVERLESS = bool(os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"))
RUNTIME_BASE_DIR = Path("/tmp") if IS_SERVERLESS else BASE_DIR

DATA_DIR = RUNTIME_BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
COLLECTION_NAME = "vwo_test_cases"

# Tunables
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_N_HYBRID = int(os.getenv("TOP_N_HYBRID", "20"))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", "4"))
RRF_K = int(os.getenv("RRF_K", "60"))
REWRITE_ENABLED = os.getenv("REWRITE_ENABLED", "true").lower() == "true"
INGEST_BATCH = int(os.getenv("INGEST_BATCH", "32"))

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3.1")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "deepseek-r1-distill-llama-70b")
USE_GROQ = os.getenv("LLM_PROVIDER", "openrouter").lower() == "groq"

QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
qdrant = None
qdrant_init_error = ""
try:
    if QDRANT_URL:
        qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)
    else:
        local_qdrant_path = (Path("/tmp") / "qdrant_data") if IS_SERVERLESS else (BASE_DIR / "qdrant_data")
        local_qdrant_path.mkdir(parents=True, exist_ok=True)
        qdrant = QdrantClient(path=str(local_qdrant_path))
except Exception as ex:
    qdrant = None
    qdrant_init_error = str(ex)


@dataclass
class Chunk:
    chunk_id: int
    text: str
    payload: Dict
    dense: List[float]
    sparse_indices: List[int]
    sparse_values: List[float]


state = {
    "upload_path": None,
    "upload_columns": [],
    "text_columns": [],
    "meta_columns": [],
    "latest_preview": [],
    "ingest_events": Queue(),
    "ingest_running": False,
    "chunks_cache": [],
    "latest_chat_used_ids": set(),
}

MODEL_CACHE = {
    "embedder": None,
    "reranker": None,
    "embedder_fallback": False,
    "reranker_fallback": False,
}


def emit_event(stage: str, status: str, detail: Dict):
    payload = {
        "stage": stage,
        "status": status,
        "detail": detail,
        "ts": time.time(),
    }
    state["ingest_events"].put(payload)


def hydrate_chunks_cache_from_qdrant(force: bool = False) -> int:
    if qdrant is None:
        return 0

    if state["chunks_cache"] and not force:
        return len(state["chunks_cache"])

    try:
        existing = [c.name for c in qdrant.get_collections().collections]
        if COLLECTION_NAME not in existing:
            return 0

        rows = []
        offset = None
        while True:
            points, next_offset = qdrant.scroll(
                collection_name=COLLECTION_NAME,
                with_payload=True,
                with_vectors=True,
                limit=512,
                offset=offset,
            )

            for p in points:
                payload = dict(p.payload or {})
                text = payload.get("text", "")
                dense = []
                sparse_preview = []

                vectors = p.vector or {}
                if isinstance(vectors, dict):
                    dense = vectors.get("dense") or []
                    sparse = vectors.get("sparse")
                    if sparse is not None:
                        idxs = list(getattr(sparse, "indices", []) or [])
                        vals = list(getattr(sparse, "values", []) or [])
                        sparse_preview = list(zip(idxs[:8], [round(float(v), 5) for v in vals[:8]]))

                chunk_id = int(payload.get("chunk_id") or p.id)
                payload_no_text = {k: v for k, v in payload.items() if k != "text"}

                rows.append(
                    {
                        "chunk_id": chunk_id,
                        "text": text,
                        "payload": payload_no_text,
                        "dense_preview": [round(float(x), 5) for x in dense[:8]],
                        "sparse_preview": sparse_preview,
                    }
                )

            if next_offset is None:
                break
            offset = next_offset

        rows.sort(key=lambda x: x["chunk_id"])
        state["chunks_cache"] = rows
        return len(rows)
    except Exception:
        return 0


def lazy_load_models():
    global BGEM3FlagModel, FlagReranker
    if BGEM3FlagModel is None or FlagReranker is None:
        try:
            from FlagEmbedding import BGEM3FlagModel as _BGEM3FlagModel
            from FlagEmbedding import FlagReranker as _FlagReranker

            BGEM3FlagModel = _BGEM3FlagModel
            FlagReranker = _FlagReranker
        except Exception as ex:
            MODEL_CACHE["embedder_fallback"] = True
            MODEL_CACHE["reranker_fallback"] = True
            emit_event("Embed", "running", {"message": f"FlagEmbedding import failed; fallback enabled ({ex})"})
            return

    if MODEL_CACHE["embedder"] is None and not MODEL_CACHE["embedder_fallback"]:
        try:
            emit_event("Embed", "running", {"message": "Loading BAAI/bge-m3 (first run can be slow)"})
            MODEL_CACHE["embedder"] = BGEM3FlagModel(
                "BAAI/bge-m3",
                use_fp16=os.getenv("BGE_USE_FP16", "1") == "1",
            )
        except Exception as ex:
            MODEL_CACHE["embedder_fallback"] = True
            emit_event("Embed", "running", {"message": f"bge-m3 unavailable; using fallback embedder ({ex})"})

    if MODEL_CACHE["reranker"] is None and not MODEL_CACHE["reranker_fallback"]:
        try:
            emit_event("Rerank", "running", {"message": "Loading BAAI/bge-reranker-v2-m3"})
            MODEL_CACHE["reranker"] = FlagReranker(
                "BAAI/bge-reranker-v2-m3",
                use_fp16=os.getenv("BGE_USE_FP16", "1") == "1",
            )
        except Exception as ex:
            MODEL_CACHE["reranker_fallback"] = True
            emit_event("Rerank", "running", {"message": f"Reranker unavailable; using fallback rerank ({ex})"})


def _simple_tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+", (text or "").lower())


def _fallback_embed_texts(texts: List[str], dim: int = 256):
    dense_vecs: List[List[float]] = []
    sparse_vecs: List[Tuple[List[int], List[float]]] = []

    for t in texts:
        vec = [0.0] * dim
        counts = {}
        toks = _simple_tokens(t)
        for tok in toks:
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % dim
            sign = 1.0 if (h & 1) == 0 else -1.0
            vec[idx] += sign

            sidx = h % 50000
            counts[sidx] = counts.get(sidx, 0.0) + 1.0

        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = [x / norm for x in vec]

        dense_vecs.append(vec)
        items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:256]
        sparse_vecs.append(([i for i, _ in items], [float(v) for _, v in items]))

    return dense_vecs, sparse_vecs


def _to_sparse(indices_like) -> Tuple[List[int], List[float]]:
    if isinstance(indices_like, dict):
        items = []
        for k, v in indices_like.items():
            try:
                idx = int(k)
            except Exception:
                idx = abs(hash(str(k))) % 2_147_483_647
            items.append((idx, float(v)))
        items.sort(key=lambda x: abs(x[1]), reverse=True)
        return [x[0] for x in items], [x[1] for x in items]

    if isinstance(indices_like, list):
        vals = [float(v) for v in indices_like]
        idxs = list(range(len(vals)))
        return idxs, vals

    return [], []


def embed_texts(texts: List[str]):
    lazy_load_models()
    if MODEL_CACHE["embedder"] is None:
        return _fallback_embed_texts(texts)

    embedder = MODEL_CACHE["embedder"]
    result = embedder.encode(
        texts,
        batch_size=INGEST_BATCH,
        max_length=8192,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )

    dense_vecs = result.get("dense_vecs") or result.get("dense") or []
    sparse_raw = result.get("lexical_weights") or result.get("sparse_vecs") or []

    dense_vecs = np.asarray(dense_vecs).tolist()
    sparse_vecs = [_to_sparse(x) for x in sparse_raw]
    return dense_vecs, sparse_vecs


def split_text(text: str) -> List[str]:
    text = (text or "").strip()
    if len(text) <= CHUNK_SIZE:
        return [text] if text else []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - CHUNK_OVERLAP)
    return chunks


def ensure_collection(dim: int = 1024):
    if qdrant is None:
        raise RuntimeError("Qdrant is not available")

    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME in existing:
        qdrant.delete_collection(COLLECTION_NAME)

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": models.VectorParams(size=dim, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False)
            )
        },
    )


def _safe(val):
    if pd.isna(val):
        return ""
    return str(val)


def run_ingest():
    if state["ingest_running"]:
        return
    state["ingest_running"] = True

    try:
        upload_path = state.get("upload_path")
        text_columns = state.get("text_columns") or []
        meta_columns = state.get("meta_columns") or []

        if not upload_path or not Path(upload_path).exists():
            emit_event("Read", "error", {"message": "No uploaded file selected."})
            return

        emit_event("Read", "running", {"message": f"Reading {Path(upload_path).name}"})
        if upload_path.lower().endswith(".csv"):
            df = pd.read_csv(upload_path)
        else:
            df = pd.read_excel(upload_path)

        emit_event(
            "Read",
            "done",
            {
                "rows": int(len(df)),
                "columns": list(df.columns),
            },
        )

        if not text_columns:
            text_columns = [c for c in df.columns if c.lower() in {"title", "steps", "expected", "tags"}]
            if not text_columns:
                text_columns = [df.columns[0]]

        emit_event(
            "Build docs",
            "running",
            {
                "message": "Assembling row-level documents",
                "text_columns": text_columns,
                "meta_columns": meta_columns,
            },
        )

        docs = []
        for i, row in df.iterrows():
            text = "\n".join([f"{c}: {_safe(row[c])}" for c in text_columns]).strip()
            payload = {c: _safe(row[c]) for c in meta_columns}
            payload["row_idx"] = int(i)
            payload["source_file"] = Path(upload_path).name
            docs.append({"text": text, "payload": payload})

        emit_event("Build docs", "done", {"docs": len(docs)})

        emit_event("Chunk", "running", {"chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP})
        chunks: List[Chunk] = []
        histogram = {"0-250": 0, "251-500": 0, "501-1000": 0, "1001+": 0}
        char_lengths = []
        chunk_id = 1

        for d in docs:
            split = split_text(d["text"])
            for s in split:
                ln = len(s)
                char_lengths.append(ln)
                if ln <= 250:
                    histogram["0-250"] += 1
                elif ln <= 500:
                    histogram["251-500"] += 1
                elif ln <= 1000:
                    histogram["501-1000"] += 1
                else:
                    histogram["1001+"] += 1

                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        text=s,
                        payload={**d["payload"], "chunk_id": chunk_id},
                        dense=[],
                        sparse_indices=[],
                        sparse_values=[],
                    )
                )
                chunk_id += 1

        emit_event(
            "Chunk",
            "done",
            {
                "total_chunks": len(chunks),
                "avg_chars": float(np.mean(char_lengths)) if char_lengths else 0,
                "min_chars": int(min(char_lengths)) if char_lengths else 0,
                "max_chars": int(max(char_lengths)) if char_lengths else 0,
                "histogram": histogram,
                "sample_chunks": [c.text[:300] for c in chunks[:3]],
            },
        )

        emit_event("Embed", "running", {"message": f"Embedding {len(chunks)} chunks"})
        dense_vecs, sparse_vecs = embed_texts([c.text for c in chunks])

        for idx, chunk in enumerate(chunks):
            chunk.dense = dense_vecs[idx]
            chunk.sparse_indices = sparse_vecs[idx][0]
            chunk.sparse_values = sparse_vecs[idx][1]

        dense_preview = chunks[0].dense[:8] if chunks else []
        sparse_preview = (
            list(zip(chunks[0].sparse_indices[:5], chunks[0].sparse_values[:5])) if chunks else []
        )
        emit_event(
            "Embed",
            "done",
            {
                "dense_dim": len(chunks[0].dense) if chunks else 0,
                "dense_preview": dense_preview,
                "sparse_top_5": sparse_preview,
            },
        )

        emit_event("Index", "running", {"message": "Creating collection and upserting points"})
        dense_dim = len(chunks[0].dense) if chunks else 1024
        ensure_collection(dense_dim)

        points = []
        for c in chunks:
            points.append(
                models.PointStruct(
                    id=c.chunk_id,
                    vector={
                        "dense": c.dense,
                        "sparse": models.SparseVector(indices=c.sparse_indices, values=c.sparse_values),
                    },
                    payload={**c.payload, "text": c.text},
                )
            )

        for i in range(0, len(points), 256):
            qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=points[i : i + 256],
                wait=True,
            )
            emit_event("Index", "running", {"progress": i + len(points[i : i + 256]), "total": len(points)})

        collection_info = qdrant.get_collection(COLLECTION_NAME)

        state["chunks_cache"] = [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "payload": c.payload,
                "dense_preview": [round(x, 5) for x in c.dense[:8]],
                "sparse_preview": list(zip(c.sparse_indices[:8], [round(v, 5) for v in c.sparse_values[:8]])),
            }
            for c in chunks
        ]

        emit_event(
            "Index",
            "done",
            {
                "points": len(points),
                "collection": COLLECTION_NAME,
                "status": str(collection_info.status),
            },
        )
        emit_event("Pipeline", "done", {"message": "Ingestion complete"})

    except Exception as ex:
        emit_event("Pipeline", "error", {"message": str(ex)})
    finally:
        state["ingest_running"] = False


def call_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    timeout_sec: int = 45,
) -> str:
    if USE_GROQ and GROQ_API_KEY:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": GROQ_MODEL,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
    else:
        if not OPENROUTER_API_KEY:
            raise RuntimeError("No LLM key configured. Set OPENROUTER_API_KEY or switch to Groq.")
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://127.0.0.1:5050",
            "X-Title": "Advanced RAG Explorer",
        }
        payload = {
            "model": OPENROUTER_MODEL,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_sec)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def rewrite_query(question: str) -> List[str]:
    if not REWRITE_ENABLED:
        return [question]

    prompt = (
        "Create 3 alternate phrasings for retrieval over software test case docs. "
        "Return exactly 3 lines, no bullets, each line a rewrite."
    )

    try:
        out = call_llm(
            "You are a retrieval rewrite engine.",
            f"Question: {question}\n\n{prompt}",
            0.1,
            timeout_sec=int(os.getenv("REWRITE_TIMEOUT_SEC", "10")),
        )
        lines = [x.strip(" -\t") for x in out.splitlines() if x.strip()]
        lines = [x for x in lines if len(x) > 5]
        if not lines:
            return [question]
        return [question] + lines[:2]
    except Exception:
        return [
            question,
            f"Detailed request about: {question}",
            f"Find related VWO test cases for: {question}",
        ]


def dense_sparse_search(query: str, top_n: int = TOP_N_HYBRID):
    if qdrant is None:
        raise RuntimeError("Qdrant is not available")

    dense, sparse = embed_texts([query])
    dense_q = dense[0]
    sparse_idx, sparse_vals = sparse[0]

    dense_hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=("dense", dense_q),
        with_payload=True,
        limit=top_n,
    )

    sparse_hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=models.NamedSparseVector(
            name="sparse",
            vector=models.SparseVector(indices=sparse_idx, values=sparse_vals),
        ),
        with_payload=True,
        limit=top_n,
    )

    return dense_hits, sparse_hits


def rrf_fuse(dense_hits, sparse_hits, k: int = RRF_K):
    scores = {}
    source = {}

    for rank, p in enumerate(dense_hits, start=1):
        pid = int(p.id)
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
        source.setdefault(pid, {})["dense_rank"] = rank

    for rank, p in enumerate(sparse_hits, start=1):
        pid = int(p.id)
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
        source.setdefault(pid, {})["sparse_rank"] = rank

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused, source


def rerank(query: str, ids: List[int]) -> List[Tuple[int, float]]:
    lazy_load_models()
    reranker = MODEL_CACHE["reranker"]

    by_id = {c["chunk_id"]: c for c in state["chunks_cache"]}
    pairs = []
    valid_ids = []
    for pid in ids:
        c = by_id.get(pid)
        if not c:
            continue
        pairs.append([query, c["text"]])
        valid_ids.append(pid)

    if not pairs:
        return []

    if reranker is None:
        q_tokens = set(_simple_tokens(query))
        scores = []
        for pid in valid_ids:
            text = by_id[pid]["text"]
            t_tokens = set(_simple_tokens(text))
            overlap = len(q_tokens.intersection(t_tokens))
            denom = max(1, len(q_tokens.union(t_tokens)))
            scores.append((pid, overlap / denom))
        return sorted(scores, key=lambda x: x[1], reverse=True)

    scores = reranker.compute_score(pairs, normalize=True)
    if isinstance(scores, float):
        scores = [scores]

    ranked = sorted(zip(valid_ids, scores), key=lambda x: x[1], reverse=True)
    return ranked


def is_generate_mode(question: str) -> bool:
    q = question.lower()
    return bool(re.search(r"\b(create|generate|new test case|draft test case)\b", q))


def grounded_answer(question: str, top_chunks: List[Dict]) -> str:
    citations = []
    context_blocks = []
    for i, c in enumerate(top_chunks, start=1):
        context_blocks.append(f"[Chunk {i}]\n{c['text']}")
        citations.append(c["chunk_id"])

    if is_generate_mode(question):
        task_prompt = (
            "Use the retrieved test cases as templates and produce one structured test case with sections: "
            "Title, Preconditions, Steps (numbered), Expected, Priority, Tags. Include realistic VWO language."
        )
    else:
        task_prompt = (
            "Answer only from retrieved chunks. If missing info, state what is missing. "
            "Cite claims as [Chunk N]."
        )

    context_joined = "\n\n".join(context_blocks)
    prompt = (
        f"Question:\n{question}\n\n"
        f"Retrieved Context:\n{context_joined}\n\n"
        f"Instruction:\n{task_prompt}"
    )

    try:
        return call_llm(
            "You are a grounded QA assistant for VWO test cases.",
            prompt,
            0.2,
            timeout_sec=int(os.getenv("ANSWER_TIMEOUT_SEC", "35")),
        )
    except Exception as ex:
        return f"LLM call failed: {ex}.\n\nFallback:\nBased on top chunks {citations}, refine query or configure API key."


def parse_generated_testcase(text: str) -> Dict[str, str]:
    patterns = {
        "summary": r"(?im)^\s*title\s*:\s*(.+)$",
        "preconditions": r"(?im)^\s*preconditions\s*:\s*(.+)$",
        "steps": r"(?ims)^\s*steps\s*:\s*(.+?)(?:\n\s*expected\s*:|\Z)",
        "expected": r"(?ims)^\s*expected\s*:\s*(.+?)(?:\n\s*priority\s*:|\Z)",
        "priority": r"(?im)^\s*priority\s*:\s*(.+)$",
        "tags": r"(?im)^\s*tags\s*:\s*(.+)$",
    }
    out = {}
    for k, p in patterns.items():
        m = re.search(p, text or "")
        out[k] = m.group(1).strip() if m else ""

    if not out["summary"]:
        first_line = (text or "").strip().splitlines()
        out["summary"] = first_line[0][:160] if first_line else "Generated test case"

    if not out["priority"]:
        out["priority"] = "P2"

    return out


@app.route("/")
def home():
    return redirect(url_for("chat_page"))


@app.route("/upload")
def upload_page():
    return render_template("upload.html")


@app.route("/ingest")
def ingest_page():
    return render_template("ingest.html")


@app.route("/chunks")
def chunks_page():
    return render_template("chunks.html")


@app.route("/chat")
def chat_page():
    return render_template("chat.html")


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify({"error": "Missing file"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty file name"}), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in {".csv", ".xlsx", ".xls"}:
        return jsonify({"error": "Only CSV/XLSX/XLS supported"}), 400

    save_name = f"{uuid.uuid4().hex}_{Path(f.filename).name}"
    save_path = UPLOAD_DIR / save_name
    f.save(save_path)

    if ext == ".csv":
        df = pd.read_csv(save_path)
    else:
        df = pd.read_excel(save_path)

    state["upload_path"] = str(save_path)
    state["upload_columns"] = list(df.columns)
    state["latest_preview"] = df.head(5).fillna("").to_dict(orient="records")

    default_text = [c for c in df.columns if c.lower() in {"title", "steps", "expected", "tags"}]
    default_meta = [c for c in df.columns if c.lower() in {"id", "jira_id", "priority", "module"}]

    return jsonify(
        {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "dtypes": {k: str(v) for k, v in df.dtypes.items()},
            "preview": state["latest_preview"],
            "upload_path": str(save_path),
            "default_text": default_text,
            "default_meta": default_meta,
        }
    )


@app.route("/api/selection", methods=["POST"])
def api_selection():
    payload = request.get_json(force=True)
    state["text_columns"] = payload.get("text_columns", [])
    state["meta_columns"] = payload.get("meta_columns", [])
    return jsonify({"ok": True, "text_columns": state["text_columns"], "meta_columns": state["meta_columns"]})


@app.route("/api/ingest/start", methods=["POST"])
def api_ingest_start():
    if qdrant is None:
        return jsonify({"ok": False, "error": f"Qdrant unavailable: {qdrant_init_error or 'init failed'}"}), 503

    if state["ingest_running"]:
        return jsonify({"ok": True, "message": "Ingest already running"})

    # Reset queue by replacing it with a new one.
    state["ingest_events"] = Queue()
    thread = threading.Thread(target=run_ingest, daemon=True)
    thread.start()
    return jsonify({"ok": True})


@app.route("/api/ingest/stream")
def api_ingest_stream():
    def stream():
        while True:
            evt = state["ingest_events"].get()
            yield f"data: {json.dumps(evt)}\n\n"
            if evt.get("stage") == "Pipeline" and evt.get("status") in {"done", "error"}:
                break

    return Response(stream(), mimetype="text/event-stream")


@app.route("/api/chunks")
def api_chunks():
    hydrate_chunks_cache_from_qdrant()
    q = (request.args.get("q") or "").strip().lower()
    priority = (request.args.get("priority") or "").strip().lower()
    module = (request.args.get("module") or "").strip().lower()
    jira_id = (request.args.get("jira_id") or "").strip().lower()

    page = max(1, int(request.args.get("page", "1")))
    page_size = 50

    rows = state["chunks_cache"]
    filtered = []
    for r in rows:
        p = r.get("payload", {})
        text = (r.get("text") or "").lower()
        if q and q not in text:
            continue
        if priority and priority != str(p.get("priority", "")).lower():
            continue
        if module and module != str(p.get("module", "")).lower():
            continue
        if jira_id and jira_id not in str(p.get("jira_id", "")).lower():
            continue
        filtered.append(r)

    total = len(filtered)
    start = (page - 1) * page_size
    end = start + page_size
    out = filtered[start:end]

    used = state.get("latest_chat_used_ids", set())
    for row in out:
        row["highlighted"] = row["chunk_id"] in used

    return jsonify({
        "page": page,
        "page_size": page_size,
        "total": total,
        "rows": out,
    })


@app.route("/api/chat", methods=["POST"])
def api_chat():
    payload = request.get_json(force=True)
    question = (payload.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Question is required"}), 400
    if qdrant is None:
        return jsonify({"error": f"Qdrant unavailable: {qdrant_init_error or 'init failed'}"}), 503

    if not state["chunks_cache"]:
        hydrate_chunks_cache_from_qdrant()
    if not state["chunks_cache"]:
        return jsonify({"error": "No chunks indexed yet. Upload and ingest first."}), 400

    rewrites = rewrite_query(question)

    dense_agg = {}
    sparse_agg = {}

    for q in rewrites[:4]:
        d_hits, s_hits = dense_sparse_search(q)
        for p in d_hits:
            dense_agg[int(p.id)] = max(float(getattr(p, "score", 0.0)), dense_agg.get(int(p.id), -1e9))
        for p in s_hits:
            sparse_agg[int(p.id)] = max(float(getattr(p, "score", 0.0)), sparse_agg.get(int(p.id), -1e9))

    dense_sorted_ids = [x[0] for x in sorted(dense_agg.items(), key=lambda kv: kv[1], reverse=True)[:TOP_N_HYBRID]]
    sparse_sorted_ids = [x[0] for x in sorted(sparse_agg.items(), key=lambda kv: kv[1], reverse=True)[:TOP_N_HYBRID]]

    # Build pseudo points for RRF compatibility.
    dense_points = [type("P", (), {"id": i}) for i in dense_sorted_ids]
    sparse_points = [type("P", (), {"id": i}) for i in sparse_sorted_ids]

    fused, source = rrf_fuse(dense_points, sparse_points)
    candidate_ids = [pid for pid, _ in fused[: max(TOP_N_HYBRID, TOP_K_RERANK * 2)]]

    rerank_before = [{"chunk_id": pid, "rrf": round(score, 6)} for pid, score in fused[:TOP_N_HYBRID]]
    reranked = rerank(question, candidate_ids)
    rerank_after = [{"chunk_id": pid, "score": round(float(score), 6)} for pid, score in reranked[:TOP_K_RERANK]]

    by_id = {c["chunk_id"]: c for c in state["chunks_cache"]}
    selected = [by_id[r["chunk_id"]] for r in rerank_after if r["chunk_id"] in by_id]

    answer = grounded_answer(question, selected)
    state["latest_chat_used_ids"] = {c["chunk_id"] for c in selected}

    return jsonify(
        {
            "rewrites": rewrites[:4],
            "dense_top": dense_sorted_ids[:TOP_N_HYBRID],
            "sparse_top": sparse_sorted_ids[:TOP_N_HYBRID],
            "rrf_top": rerank_before,
            "rerank": {
                "before": rerank_before,
                "after": rerank_after,
                "source": source,
            },
            "answer": answer,
            "citations": [c["chunk_id"] for c in selected],
        }
    )


@app.route("/api/health")
def api_health():
    cached = hydrate_chunks_cache_from_qdrant()
    return jsonify(
        {
            "ok": True,
            "collection": COLLECTION_NAME,
            "qdrant_mode": "remote" if QDRANT_URL else "embedded",
            "chunks_indexed": cached,
            "rewrite_enabled": REWRITE_ENABLED,
            "llm_provider": "groq" if USE_GROQ else "openrouter",
            "qdrant_ready": qdrant is not None,
            "qdrant_init_error": qdrant_init_error,
        }
    )


@app.route("/api/chat/export_jira", methods=["POST"])
def api_chat_export_jira():
    payload = request.get_json(force=True)
    text = (payload.get("text") or "").strip()
    issue_key = (payload.get("issue_key") or "VWO-NEW").strip()
    assignee = (payload.get("assignee") or "qa_team_1").strip()
    module = (payload.get("module") or "Advanced RAG").strip()

    parsed = parse_generated_testcase(text)
    labels = parsed["tags"].replace(",", " ").strip()
    description = (
        f"Preconditions: {parsed['preconditions']}\n"
        f"Steps: {parsed['steps']}\n"
        f"Expected: {parsed['expected']}"
    )

    df = pd.DataFrame(
        [
            {
                "Issue Key": issue_key,
                "Project Key": "VWO",
                "Issue Type": "Test",
                "Summary": parsed["summary"],
                "Description": description,
                "Priority": parsed["priority"],
                "Labels": labels,
                "Status": "Ready",
                "Assignee": assignee,
                "Environment": "QA Sandbox",
                "Module": module,
                "Test Type": "Generated",
            }
        ]
    )
    csv_text = df.to_csv(index=False)
    return Response(
        csv_text,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=generated_jira_row.csv"},
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5050"))
    debug_mode = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="127.0.0.1", port=port, debug=debug_mode, use_reloader=False)
