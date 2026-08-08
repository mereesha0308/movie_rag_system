"""Chunk long movie plots into smaller overlapping-window chunks.

Chunking is sentence-aware: sentences are accumulated until the chunk
approaches the target word count, avoiding awkward mid-sentence cuts.
"""

from __future__ import annotations

import re

import nltk

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, falling back to a regex split."""
    try:
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        sentences = nltk.sent_tokenize(text)
    except Exception:
        sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return [s for s in sentences if s.strip()]


class Chunker:
    """Split a plot into chunks of roughly `chunk_size` words."""

    def __init__(self, chunk_size: int = 300, overlap: int = 0) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        """Return a list of text chunks for one plot."""
        sentences = _split_sentences(text)
        if not sentences:
            return []

        chunks: list[str] = []
        current: list[str] = []
        current_words = 0

        for sentence in sentences:
            words = sentence.split()
            # Hard-split a single oversized sentence into word windows.
            if len(words) > self.chunk_size:
                if current:
                    chunks.append(" ".join(current))
                    current, current_words = [], 0
                for i in range(0, len(words), self.chunk_size):
                    chunks.append(" ".join(words[i : i + self.chunk_size]))
                continue

            if current_words + len(words) > self.chunk_size and current:
                chunks.append(" ".join(current))
                # Carry the last `overlap` words into the next chunk.
                overlap_words = current[-self.overlap:] if self.overlap > 0 else []
                current = list(overlap_words)
                current_words = len(overlap_words)
            current.append(sentence)
            current_words += len(words)

        if current:
            chunks.append(" ".join(current))

        return chunks

    def chunk_dataframe(self, df, plot_column: str = "Plot") -> list[dict]:
        """Chunk every row and return a flat list of chunk records."""
        records: list[dict] = []
        for idx, row in df.iterrows():
            chunks = self.chunk(row[plot_column])
            for ci, chunk_text in enumerate(chunks):
                records.append(
                    {
                        "chunk_id": f"{idx}-{ci}",
                        "chunk_text": chunk_text,
                        "plot_index": idx,
                        "chunk_index": ci,
                        "Title": row.get("Title"),
                        "Release Year": row.get("Release Year"),
                        "Origin/Ethnicity": row.get("Origin/Ethnicity"),
                        "Genre": row.get("Genre"),
                        "Director": row.get("Director"),
                    }
                )
        return records
