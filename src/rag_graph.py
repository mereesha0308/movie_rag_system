"""High-level RAG pipeline orchestration (build index + structured query)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.agent import MovieAgent
from src.chunker import Chunker
from src.data_loader import DataLoader
from src.embeddings import Embedder
from src.preprocessor import Preprocessor
from src.retriever import Retriever


class RAGPipeline:
    """End-to-end mini RAG: subset -> clean -> chunk -> embed -> retrieve -> generate."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.loader = DataLoader(config)
        self.preprocessor = Preprocessor(config)
        self.chunker = Chunker(
            chunk_size=config["chunking"]["chunk_size"],
            overlap=config["chunking"].get("overlap", 0),
        )
        self.embedder = Embedder(config)
        self.retriever = Retriever(config)
        self.agent = MovieAgent(config, self.retriever, self.embedder)
        self._chunks: list[dict] = []

    # ------------------------------------------------------------------ index
    def build_index(self, force: bool = False) -> np.ndarray | None:
        """Build the vector index once, then reuse it.

        The sampled subset is cached to ``data/subset.csv`` and the embeddings
        are persisted in the Chroma store, so chunking + embedding only happen
        once. Pass ``force=True`` to redo everything from scratch.
        """
        subset_path = self.loader.resolve_subset_path()
        already_built = subset_path.exists() and self.retriever.count > 0

        if already_built and not force:
            print(
                f"Index already built ({self.retriever.count:,} chunks, "
                f"subset cached at {subset_path}). Use --force to rebuild."
            )
            return None

        if subset_path.exists() and not force:
            df = pd.read_csv(subset_path)
            print(f"Loading existing subset from {subset_path}")
        else:
            df = self._make_subset()
            df[["Title", "Plot"]].to_csv(subset_path, index=False)
            print(f"Subset saved: {len(df):,} movies written to {subset_path}")

        self._chunks = self.chunker.chunk_dataframe(df)
        if not self._chunks:
            raise RuntimeError("No chunks were produced from the subset.")

        texts = [c["chunk_text"] for c in self._chunks]
        embeddings = self.embedder.encode(texts)

        self.retriever.reset()
        self.retriever.add_chunks(self._chunks, embeddings)
        return embeddings

    def _make_subset(self) -> pd.DataFrame:
        """Load the raw dataset, clean it (duplicates / missing / unknown) and
        sample `subset_size` rows with a fixed seed. This is the project's
        fixed, final random subset."""
        df = self.loader.load_raw()
        df = self.preprocessor.clean(df)
        df["Plot"] = df["Plot"].map(self.preprocessor.normalize_plot)

        size = int(self.config["data"].get("subset_size", 300))
        seed = int(self.config["data"].get("random_seed", 42))
        if len(df) > size:
            df = df.sample(n=size, random_state=seed)

        return df.reset_index(drop=True)

    # ----------------------------------------------------------------- query
    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """Encode the query and return top-k matching chunks."""
        query_emb = self.embedder.encode([query])[0]
        return self.retriever.retrieve(query_emb, top_k=top_k, query_text=query)

    def answer(self, query: str, top_k: int | None = None) -> dict:
        """Full RAG query returning the structured output JSON.

        Structure: {answer, contexts, reasoning} (+ clarifying_question when
        the agent needs more information from the user).
        """
        if self.retriever.count == 0:
            raise RuntimeError(
                "Index is empty. Run `python main.py build-index` first."
            )

        if top_k is not None:
            self.agent.top_k = top_k

        return self.agent.run(query)
