"""Tests for core components (run with: python -m pytest tests/)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.chunker import Chunker
from src.preprocessor import Preprocessor
from src.retriever import Retriever


def make_config(tmp_path):
    return {
        "data": {
            "raw_path": str(tmp_path / "raw.csv"),
            "clean_path": str(tmp_path / "clean.csv"),
        },
        "preprocessing": {
            "drop_duplicates": True,
            "drop_missing": True,
            "drop_unknown": True,
            "unknown_director": "Unknown",
            "unknown_genre": "unknown",
            "unknown_cast": "Unknown",
        },
        "retriever": {
            "vector_store": "chroma",
            "store_path": str(tmp_path / "chroma_store"),
            "collection_name": "test_movie_chunks",
            "top_k": 3,
        },
    }


def sample_df():
    return pd.DataFrame(
        {
            "Title": ["A", "A", "B", "C", "D", "E"],
            "Release Year": [2000, 2000, 2001, 2002, 2003, 2004],
            "Origin/Ethnicity": ["American"] * 6,
            "Director": ["D1", "D1", "D2", "D3", "Unknown", "D5"],
            "Cast": ["C1", "C1", "C2", "C3", "C4", None],
            "Genre": ["drama", "drama", "comedy", "unknown", "action", "horror"],
            "Wiki Page": ["w"] * 6,
            "Plot": ["plot a", "plot a", "plot b", "plot c", "plot d", "plot e"],
        }
    )


def test_clean_removes_duplicates_missing_unknown(tmp_path):
    pre = Preprocessor(make_config(tmp_path))
    cleaned = pre.clean(sample_df())

    assert cleaned["Title"].duplicated().sum() == 0
    assert cleaned.isna().sum().sum() == 0
    assert not cleaned["Director"].eq("Unknown").any()
    assert not cleaned["Genre"].eq("unknown").any()


def test_normalize_plot():
    assert Preprocessor({}).normalize_plot("  Hello   WORLD \n") == "Hello WORLD"


def test_chunker_respects_target_size():
    long_text = " ".join(["This is one sentence." for _ in range(200)])
    chunks = Chunker(chunk_size=50).chunk(long_text)

    assert len(chunks) > 1
    assert all(0 < len(c.split()) <= 60 for c in chunks)
    # Chunks are combined from whole sentences.
    assert all(c.endswith("sentence.") for c in chunks)


def test_chunker_dataframe_records():
    df = pd.DataFrame(
        {"Title": ["T1", "T2"], "Plot": ["Short plot text.", "This is a sentence. " * 100]}
    )
    records = Chunker(chunk_size=100).chunk_dataframe(df)
    assert len(records) >= 3
    assert all("chunk_id" in r and "chunk_text" in r for r in records)
    assert records[0]["Title"] == "T1"


def test_chunker_splits_oversized_sentence():
    long_no_punct = "x " * 1000
    chunks = Chunker(chunk_size=200).chunk(long_no_punct)
    assert len(chunks) == 5
    assert all(len(c.split()) <= 200 for c in chunks)


def test_retriever_add_and_retrieve(tmp_path):
    cfg = make_config(tmp_path)
    retriever = Retriever(cfg)

    records = [
        {"chunk_id": "0-0", "chunk_text": "A spaceship controlled by an AI named HAL.",
         "Title": "2001", "Release Year": 1968},
        {"chunk_id": "1-0", "chunk_text": "Two robots fall in love.",
         "Title": "WALL-E", "Release Year": 2008},
        {"chunk_id": "2-0", "chunk_text": "A farmer fights aliens in Kansas.",
         "Title": "Fields", "Release Year": 1996},
    ]
    embeddings = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    retriever.add_chunks(records, embeddings)

    # Query closest to the AI/spaceship chunk.
    query_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    results = retriever.retrieve(query_emb, top_k=2)

    assert len(results) == 2
    assert results[0]["Title"] == "2001"
    assert results[0]["similarity"] > results[1]["similarity"]


def test_retriever_returns_empty_on_empty_index(tmp_path):
    retriever = Retriever(make_config(tmp_path))
    query_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    assert retriever.retrieve(query_emb, top_k=3) == []
    assert retriever.retrieve(query_emb, top_k=3, query_text="spaceship") == []


def test_retriever_filters_below_similarity_threshold(tmp_path):
    cfg = make_config(tmp_path)
    cfg["retriever"]["similarity_threshold"] = 0.5
    retriever = Retriever(cfg)

    records = [
        {"chunk_id": "0-0", "chunk_text": "A spaceship controlled by an AI named HAL.",
         "Title": "2001", "Release Year": 1968},
        {"chunk_id": "1-0", "chunk_text": "Two robots fall in love.",
         "Title": "WALL-E", "Release Year": 2008},
    ]
    embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    retriever.add_chunks(records, embeddings)

    # Query is identical to the 2001 chunk (sim 1.0), orthogonal to WALL-E (sim 0.0).
    query_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    results = retriever.retrieve(query_emb, top_k=3)

    assert len(results) == 1
    assert results[0]["Title"] == "2001"
