"""
ingest.py

- Loads crawl4ai JSONL with schema: {"source": "...", "text": "..."}
- Light cleanup for markdown/Wix boilerplate
- Chunks documents
- Builds/persists Chroma DB

Designed to be callable from an API endpoint:
  from ingest import run_ingest
  stats = run_ingest()
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# -----------------------------
# CONFIG (edit here, no CLI args)
# -----------------------------
JSONL_PATH = Path("./data/promtior_docs.jsonl")      # <-- tu jsonl
PERSIST_DIR = Path("../chroma_db")              # <-- carpeta chroma persistente
COLLECTION_NAME = "promtior"

# Rebuild behavior
REBUILD_VECTORSTORE_ON_RUN = True              # True: borra PERSIST_DIR y reconstruye todo

# Chunking
CHUNK_SIZE = 1800
CHUNK_OVERLAP = 200

# Ingest batching
BATCH_SIZE = 128

# Filtering
MIN_CHARS_PER_DOC = 200                        # descarta scraps chicos tipo menús
MAX_DOC_CHARS = 300_000                        # safety (evita páginas enormes)

# Cleaning toggles
CLEAN_MARKDOWN = True
CROP_TOP_BOTTOM_PAGE = True

# Embeddings
EMBEDDINGS_PROVIDER = "openai"                 # "openai" o "st"
OPENAI_EMBED_MODEL = os.getenv("OPENAI_MODEL_EMBEDDINGS", "text-embedding-3-small")
ST_EMBED_MODEL = os.getenv("ST_EMBEDDINGS_MODEL", "all-MiniLM-L6-v2")

# Logging
LOG_LEVEL = os.getenv("INGEST_LOG_LEVEL", "INFO").upper()

# PDF extra source (extra points)
PDF_PATH = Path("./data/AI Engineer.pdf")  # ajustá a tu ruta real
INGEST_PDF = True

# Si querés 1 doc por página (mejor granularidad en PDFs)
PDF_ONE_DOC_PER_PAGE = True
PDF_MIN_CHARS_PER_PAGE = 200


logger = logging.getLogger("ingest")


@dataclass(frozen=True)
class IngestStats:
    loaded_lines: int
    valid_docs: int
    skipped_docs: int
    deduped_docs: int
    chunks: int
    persisted_to: str
    collection: str


def _setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def _iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield line_no, obj
                else:
                    logger.debug("Skipping non-dict JSON at line %d", line_no)
            except json.JSONDecodeError:
                logger.warning("Skipping invalid JSON at line %d", line_no)

def _extract_pdf_documents(pdf_path: Path) -> List[Document]:
    """
    Extracts text from a PDF into Documents.
    - If PDF_ONE_DOC_PER_PAGE=True -> one Document per page
    - Else -> a single Document for the whole PDF
    """
    if not pdf_path.exists():
        logger.warning("PDF not found, skipping: %s", pdf_path)
        return []

    try:
        from pypdf import PdfReader
    except Exception as e:
        raise RuntimeError("Missing dependency 'pypdf'. Install with: pip install pypdf") from e

    reader = PdfReader(str(pdf_path))
    docs: List[Document] = []

    def clean_pdf_text(s: str) -> str:
        # PDF text suele venir con saltos raros
        s = s.replace("\r", "\n")
        s = re.sub(r"\n{3,}", "\n\n", s)
        s = re.sub(r"[ \t]+", " ", s)
        return s.strip()

    if PDF_ONE_DOC_PER_PAGE:
        for i, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            text = clean_pdf_text(raw)
            if len(text) < PDF_MIN_CHARS_PER_PAGE:
                continue

            meta = {
                "url": f"local://{pdf_path.name}",
                "source_type": "pdf",
                "pdf_file": pdf_path.name,
                "page": i,
            }
            meta["content_sha256"] = _sha256_str(text)

            docs.append(Document(page_content=text, metadata=meta))
        return docs

    # Single doc for whole PDF
    all_text = []
    for page in reader.pages:
        all_text.append(page.extract_text() or "")
    merged = clean_pdf_text("\n\n".join(all_text))
    if not merged:
        return []

    meta = {
        "url": f"local://{pdf_path.name}",
        "source_type": "pdf",
        "pdf_file": pdf_path.name,
    }
    meta["content_sha256"] = _sha256_str(merged)
    return [Document(page_content=merged, metadata=meta)]

# -----------------------------
# Cleaning helpers (minimal but high ROI for Wix markdown)
# -----------------------------

_IMG_MD_RE = re.compile(r"!\[[^\]]*]\([^)]+\)")
_LINK_MD_RE = re.compile(r"\[([^\]]+)]\(([^)]+)\)")  # keep label
_MULTISPACE_RE = re.compile(r"[ \t]+")


def _crop_top_bottom(text: str) -> str:
    """
    If the content includes 'top of page' ... 'bottom of page', keep the middle.
    This matches exactly what your sample shows.
    """
    if not CROP_TOP_BOTTOM_PAGE:
        return text

    lower = text.lower()
    top_idx = lower.find("top of page")
    bot_idx = lower.rfind("bottom of page")

    if top_idx != -1 and bot_idx != -1 and bot_idx > top_idx:
        # Keep middle section between them
        mid = text[top_idx:bot_idx]
        # Often the "top of page" area is a nav menu; try to jump after it.
        # Heuristic: keep from first "## " heading if present.
        h2 = mid.find("## ")
        if h2 != -1:
            return mid[h2:].strip()
        return mid.strip()

    return text


def _clean_markdown(text: str) -> str:
    if not CLEAN_MARKDOWN:
        return text

    # 1) remove images: ![alt](url)
    text = _IMG_MD_RE.sub(" ", text)

    # 2) convert [label](url) -> label
    text = _LINK_MD_RE.sub(r"\1", text)

    # 3) remove leftover markdown bullets/symbols that dominate nav
    text = text.replace("*", " ")
    text = text.replace("[]", " ")

    # 4) normalize whitespace
    text = text.replace("\r", "\n")
    text = _MULTISPACE_RE.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _normalize_text(raw: str) -> str:
    raw = raw[:MAX_DOC_CHARS]
    raw = _crop_top_bottom(raw)
    raw = _clean_markdown(raw)
    # final collapse (keep some structure)
    raw = raw.strip()
    return raw


def _extract_doc(obj: Dict[str, Any]) -> Optional[Document]:
    """
    Your schema example:
      {"source": "https://www.promtior.ai/blog", "text": "..."}
    """
    url = obj.get("source") or obj.get("url") or obj.get("page_url")
    text = obj.get("text") or obj.get("content") or obj.get("markdown")

    if not isinstance(text, str) or not text.strip():
        return None

    normalized = _normalize_text(text)
    if len(normalized) < MIN_CHARS_PER_DOC:
        return None

    meta: Dict[str, Any] = {}
    if url:
        meta["url"] = str(url)
    # helpful for dedupe + traceability
    meta["content_sha256"] = _sha256_str(normalized)

    return Document(page_content=normalized, metadata=meta)


def _dedupe(docs: List[Document]) -> List[Document]:
    """
    Dedupe by (url, content_sha256) if url exists, else by content_sha256.
    """
    seen = set()
    out: List[Document] = []
    for d in docs:
        url = d.metadata.get("url", "")
        key = (url, d.metadata.get("content_sha256"))
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def _split_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def _get_embeddings():
    if EMBEDDINGS_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings

        # Requires OPENAI_API_KEY in env
        return OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)

    if EMBEDDINGS_PROVIDER == "st":
        from langchain_community.embeddings import SentenceTransformerEmbeddings

        return SentenceTransformerEmbeddings(model_name=ST_EMBED_MODEL)

    raise ValueError(f"Unknown EMBEDDINGS_PROVIDER: {EMBEDDINGS_PROVIDER}")


def _maybe_rebuild_persist_dir() -> None:
    if not REBUILD_VECTORSTORE_ON_RUN:
        return

    if PERSIST_DIR.exists():
        logger.warning("REBUILD is ON. Removing persist dir: %s", PERSIST_DIR)
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)

    PERSIST_DIR.mkdir(parents=True, exist_ok=True)


def run_ingest() -> IngestStats:
    """
    Call this from your endpoint to rebuild the vectorstore.
    Returns stats you can log/return in the API response.
    """
    _setup_logging()

    if not JSONL_PATH.exists():
        raise FileNotFoundError(f"JSONL_PATH not found: {JSONL_PATH.resolve()}")

    _maybe_rebuild_persist_dir()

    loaded_lines = 0
    docs: List[Document] = []
    skipped_docs = 0

    for line_no, obj in _iter_jsonl(JSONL_PATH):
        loaded_lines += 1
        d = _extract_doc(obj)
        # print("================= BEFORE CLEANING =================")
        # print(obj)
        # print("================= AFTER CLEANING =================")
        # print(d)
        # input("\n\nPress Enter to continue...")
        if d is None:
            skipped_docs += 1
            continue
        with open("cleaned_output.txt", "a", encoding="utf-8") as clean_file:
            clean_file.write(d.page_content + "\n\n====================\n\n")
        docs.append(d)

    logger.info("Loaded lines=%d | docs=%d | skipped=%d", loaded_lines, len(docs), skipped_docs)

        # Add PDF docs (extra source)
    if INGEST_PDF:
        pdf_docs = _extract_pdf_documents(PDF_PATH)
        if pdf_docs:
            logger.info("Loaded %d PDF documents from %s", len(pdf_docs), PDF_PATH)
            docs.extend(pdf_docs)
        else:
            logger.info("No PDF documents extracted (empty or missing).")


    before = len(docs)
    docs = _dedupe(docs)
    deduped_docs = len(docs)
    logger.info("After dedupe: %d (from %d)", deduped_docs, before)

    chunks = _split_docs(docs)
    logger.info("Chunked into %d chunks (chunk_size=%d overlap=%d)", len(chunks), CHUNK_SIZE, CHUNK_OVERLAP)

    embeddings = _get_embeddings()

    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIR),
        embedding_function=embeddings,
    )

    # Stable IDs: based on url + content hash + chunk index within that doc
    # (so reruns overwrite rather than ballooning)
    ids: List[str] = []
    chunk_docs: List[Document] = []

    # We need a stable per-source chunk index. We'll compute it per (url, content_sha256)
    per_source_counter: Dict[Tuple[str, str], int] = {}

    for ch in chunks:
        url = ch.metadata.get("url", "no_url")
        csha = ch.metadata.get("content_sha256", "")
        key = (url, csha)

        idx = per_source_counter.get(key, 0)
        per_source_counter[key] = idx + 1

        stable = _sha256_str(f"{url}::{csha}::chunk_{idx}")
        ids.append(stable)
        chunk_docs.append(ch)

    total = len(chunk_docs)
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        vectordb.add_documents(documents=chunk_docs[start:end], ids=ids[start:end])
        logger.info("Upserted %d/%d", end, total)

    logger.info("Ingest finished. Persisted at %s (collection=%s)", PERSIST_DIR, COLLECTION_NAME)

    return IngestStats(
        loaded_lines=loaded_lines,
        valid_docs=before,
        skipped_docs=skipped_docs,
        deduped_docs=deduped_docs,
        chunks=total,
        persisted_to=str(PERSIST_DIR.resolve()),
        collection=COLLECTION_NAME,
    )


if __name__ == "__main__":
    load_dotenv()
    stats = run_ingest()
    logger.info("STATS: %s", stats)
