"""Vector retrieval over movie plot chunks using Chroma (in-memory / persistent)."""

from __future__ import annotations

import re

import numpy as np

from src.utils import resolve_path

STOPWORDS = frozenset(
    """
    a an the and or but nor for yet so if then else
    of in on at to from by with about against between into through during before after
    above below up down out off over under again further once here there when where why
    how all any both each few more most other some such no not only own same than too
    very just can will just shall should may might must
    what who whom which this that these those is are was were be been being am has have
    had do does did doing i me my we our you your he him his she her they them their it
    its tell movie movies film films plot story about question queries query
    """.split()
)


def _content_tokens(text: str) -> set[str]:
    """Lowercase alphanumeric tokens with English stopwords removed."""
    if not text:
        return set()
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in tokens if t not in STOPWORDS}


class Retriever:
    """Thin wrapper around a Chroma collection for chunk retrieval.

    Retrieval is hybrid: cosine similarity to the query embedding, plus a
    keyword boost for chunks whose ``Title`` metadata shares significant tokens
    with the query text (e.g. "Poltergeist II" pulls in "Poltergeist II: The
    Other Side" even when the embedding model ranks it low).
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        retr = config["retriever"]
        self.top_k = retr.get("top_k", 3)
        self.similarity_threshold = retr.get("similarity_threshold", 0.0)
        self.title_boost = retr.get("title_boost", 0.4)
        self.store_path = resolve_path(retr["store_path"])
        self.collection_name = retr.get("collection_name", "movie_chunks")

        import chromadb

        self._client = chromadb.PersistentClient(path=str(self.store_path))
        self._collection = self._client.get_or_create_collection(
            self.collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, records: list[dict], embeddings: np.ndarray) -> None:
        """Store chunk records (text + metadata) together with their embeddings."""
        self._collection.add(
            ids=[r["chunk_id"] for r in records],
            documents=[r["chunk_text"] for r in records],
            metadatas=[
                {k: v for k, v in r.items() if k not in ("chunk_id", "chunk_text")}
                for r in records
            ],
            embeddings=embeddings.tolist(),
        )

    def reset(self) -> None:
        """Delete the stored collection (used to rebuild the index)."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.create_collection(
            self.collection_name, metadata={"hnsw:space": "cosine"}
        )

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int | None = None,
        query_text: str | None = None,
    ) -> list[dict]:
        """Return the top-k most relevant chunks for the query embedding.

        When ``query_text`` is given, chunks whose ``Title`` shares significant
        tokens with the query get a similarity boost before ranking, so titles
        mentioned in the question surface even if the embedding similarity is low.
        """
        total = self._collection.count()
        if total == 0:
            return []
        k = min(top_k or self.top_k, total) or 1
        n = total if query_text else k
        result = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        chunks: list[dict] = []
        for cid, doc, meta, dist in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            chunks.append(
                {
                    "chunk_id": cid,
                    "chunk_text": doc,
                    "distance": float(dist),
                    "similarity": float(1 - dist),
                    **meta,
                }
            )

        if query_text and self.title_boost > 0:
            chunks = self._apply_title_boost(chunks, query_text)

        if self.similarity_threshold > 0:
            chunks = [
                c for c in chunks if c["similarity"] >= self.similarity_threshold
            ]
        return chunks[:k]

    def _apply_title_boost(self, chunks: list[dict], query_text: str) -> list[dict]:
        """Bump similarity for chunks whose Title overlaps the query keywords."""
        q_tokens = _content_tokens(query_text)
        if not q_tokens:
            return chunks
        for c in chunks:
            title_tokens = _content_tokens(c.get("Title", ""))
            if not title_tokens:
                continue
            overlap = len(q_tokens & title_tokens)
            if overlap == 0:
                continue
            ratio = overlap / min(len(q_tokens), len(title_tokens))
            c["similarity"] = float(c["similarity"]) + self.title_boost * ratio
        return sorted(chunks, key=lambda c: c["similarity"], reverse=True)

    @property
    def count(self) -> int:
        return self._collection.count()
