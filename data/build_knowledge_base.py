
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterator

DATA_DIR   = Path(__file__).parent
CHROMA_DIR = DATA_DIR / "chroma_db"

EMBED_MODEL    = "pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb"
EMBED_FALLBACK = "all-MiniLM-L6-v2"

CHUNK_SIZE    = 512
CHUNK_OVERLAP = 80
EMBED_BATCH   = 64
DEFAULT_TOP_K = 5

COLLECTIONS: dict[str, str] = {
    "tumor":  "neuroagent_tumor",
    "stroke": "neuroagent_stroke",
    "adhd":   "neuroagent_adhd",
}

JSONL_FILES: dict[str, Path] = {
    cond: DATA_DIR / f"{cond}_papers.jsonl"
    for cond in COLLECTIONS
}

def chunk_text(text: str) -> list[str]:
    if len(text) <= CHUNK_SIZE:
        return [text.strip()]
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start : start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        if start + CHUNK_SIZE >= len(text):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def iter_chunks(jsonl_path: Path) -> Iterator[tuple[str, str, dict]]:
    skipped = 0
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            abstract = (rec.get("abstract") or "").strip()
            title    = (rec.get("title")    or "").strip()
            if not abstract:
                skipped += 1
                continue

            full_text = f"{title}\n\n{abstract}" if title else abstract
            meta = {
                "pmid":    rec.get("pmid",    ""),
                "title":   title,
                "year":    str(rec.get("year", "")),
                "journal": rec.get("journal", ""),
                "authors": rec.get("authors", ""),
                "doi":     rec.get("doi",     ""),
                "mesh":    " | ".join(rec.get("mesh") or []),
                "source":  "PubMed",
            }
            for i, chunk in enumerate(chunk_text(full_text)):
                yield f"{meta['pmid']}_c{i}", chunk, meta

    if skipped:
        print(f"  Skipped {skipped} records (no abstract or bad JSON)")

_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model: {EMBED_MODEL}")
        try:
            _embed_model = SentenceTransformer(EMBED_MODEL)
        except Exception as e:
            print(f"  Primary model failed ({e}), falling back to {EMBED_FALLBACK}")
            _embed_model = SentenceTransformer(EMBED_FALLBACK)
        print("  Model loaded.")
    return _embed_model

def get_chroma_client():
    import chromadb
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))

def get_or_create_collection(client, name: str, recreate: bool = False):
    if recreate:
        try:
            client.delete_collection(name)
        except Exception:
            pass
        return client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
    try:
        return client.get_collection(name)
    except Exception:
        return client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

def _batched(items, size):
    for i in range(0, len(items), size):
        yield items[i : i + size]

def build_condition(condition: str, recreate: bool = True) -> int:
    jsonl = JSONL_FILES[condition]
    if not jsonl.exists():
        print(f"  ERROR: {jsonl} not found. Run pubmed_downloader.py first.")
        return 0

    model  = get_embed_model()
    client = get_chroma_client()
    col    = get_or_create_collection(client, COLLECTIONS[condition], recreate=recreate)

    all_chunks = list(iter_chunks(jsonl))
    total = len(all_chunks)
    print(f"  {total} chunks from {jsonl.name}")

    ingested = 0
    t0 = time.time()

    for batch in _batched(all_chunks, EMBED_BATCH):
        ids    = [c[0] for c in batch]
        texts  = [c[1] for c in batch]
        metas  = [c[2] for c in batch]

        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        col.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metas)
        ingested += len(ids)

        elapsed = time.time() - t0
        rate = ingested / elapsed if elapsed > 0 else 0
        print(f"  {ingested}/{total} chunks  ({rate:.0f}/s)", end="\r")

    print(f"  {ingested}/{total} chunks ingested into '{COLLECTIONS[condition]}'")
    return ingested

def build_all(conditions: list[str] | None = None, recreate: bool = True):
    targets = conditions or list(COLLECTIONS)
    for cond in targets:
        print(f"\n[{cond.upper()}]")
        n = build_condition(cond, recreate=recreate)
        print(f"  Done: {n} chunks")

_retrievers: dict[str, object] = {}

def _get_collection(condition: str):
    if condition not in _retrievers:
        client = get_chroma_client()
        col_name = COLLECTIONS.get(condition)
        if col_name is None:
            raise ValueError(f"Unknown condition: {condition}")
        try:
            _retrievers[condition] = client.get_collection(col_name)
        except Exception as e:
            raise RuntimeError(
                f"ChromaDB collection '{col_name}' not found. "
                "Run: python data/build_knowledge_base.py --build"
            ) from e
    return _retrievers[condition]

def retrieve(
    condition: str,
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    model = get_embed_model()
    col   = _get_collection(condition)

    vec = model.encode([query]).tolist()
    res = col.query(
        query_embeddings=vec,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for doc, meta, dist in zip(
        res["documents"][0],
        res["metadatas"][0],
        res["distances"][0],
    ):
        score = round(1.0 - dist / 2.0, 4)
        output.append({
            "text":    doc,
            "score":   score,
            "pmid":    meta.get("pmid",    ""),
            "title":   meta.get("title",   ""),
            "year":    meta.get("year",    ""),
            "journal": meta.get("journal", ""),
            "authors": meta.get("authors", ""),
            "doi":     meta.get("doi",     ""),
            "mesh":    meta.get("mesh",    ""),
        })
    return output

def format_for_llm(results: list[dict], max_chars: int = 6000) -> str:
    parts, total = [], 0
    for i, r in enumerate(results, 1):
        header = (
            f"[{i}] {r['title']} | {r['journal']} {r['year']} "
            f"| PMID:{r['pmid']} | score={r['score']}"
        )
        block = f"{header}\n{r['text']}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts)

def collection_exists(condition: str) -> bool:
    try:
        client = get_chroma_client()
        col = client.get_collection(COLLECTIONS[condition])
        return col.count() > 0
    except Exception:
        return False

_TEST_QUERIES: dict[str, list[str]] = {
    "tumor": [
        "glioblastoma ring enhancement necrosis grading",
        "eloquent cortex awake craniotomy surgical risk",
        "butterfly glioma corpus callosum invasion",
    ],
    "stroke": [
        "middle cerebral artery stroke aphasia hemiplegia",
        "internal capsule pure motor stroke lenticulostriate",
        "Wallenberg syndrome PICA lateral medullary",
    ],
    "adhd": [
        "caudate volume reduction frontostriatal dopamine",
        "default mode network failure to suppress inattention",
        "right inferior frontal gyrus response inhibition stop-signal",
    ],
}

def run_smoke_test():
    print("Knowledge Base Smoke Test")
    for condition, queries in _TEST_QUERIES.items():
        print(f"\n[{condition.upper()}]")
        for q in queries:
            try:
                results = retrieve(condition, q, top_k=2)
                top = results[0] if results else None
                if top:
                    print(f"  Q: {q[:60]}")
                    print(f"     -> {top['title'][:70]} (score={top['score']}, {top['year']})")
                else:
                    print(f"  Q: {q[:60]}\n     -> No results")
            except RuntimeError as e:
                print(f"  ERROR: {e}")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuroAgent Knowledge Base Builder")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build", action="store_true")
    group.add_argument("--test",  action="store_true")
    group.add_argument("--query", type=str)
    parser.add_argument("--condition", choices=["tumor", "stroke", "adhd", "all"],
                        default="all")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    if args.build:
        conditions = (
            None if args.condition == "all" else [args.condition]
        )
        build_all(conditions=conditions)
        print("\nKnowledge base built. Run with --test to validate.")

    elif args.test:
        run_smoke_test()

    elif args.query:
        if args.condition == "all":
            print("Please specify --condition with --query")
            sys.exit(1)
        results = retrieve(args.condition, args.query, top_k=args.top_k)
        print(f"\nQuery: {args.query}")
        print(f"Condition: {args.condition}  Top-{args.top_k} results:\n")
        print(format_for_llm(results))
