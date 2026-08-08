"""Preprocess the movie plot dataset (mirrors the Preprocessing notebook)."""

from __future__ import annotations

import re

import pandas as pd

from src.utils import resolve_path


class Preprocessor:
    """Clean the dataset: drop duplicates, missing values and 'unknown' records."""

    def __init__(self, config: dict) -> None:
        self.config = config
        pre = config.get("preprocessing", {})
        self.drop_duplicates = pre.get("drop_duplicates", True)
        self.drop_missing = pre.get("drop_missing", True)
        self.drop_unknown = pre.get("drop_unknown", True)
        self.unknown_director = pre.get("unknown_director", "Unknown")
        self.unknown_genre = pre.get("unknown_genre", "unknown")
        self.unknown_cast = pre.get("unknown_cast", "Unknown")

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full cleaning pipeline and return the cleaned frame."""
        df = df.copy()

        if self.drop_duplicates:
            df = df.drop_duplicates(subset=["Title"], keep="first")

        if self.drop_missing:
            df = df.dropna()

        if self.drop_unknown:
            mask = (
                df["Director"].isin([self.unknown_director])
                | df["Genre"].isin([self.unknown_genre])
                | df["Cast"].isin([self.unknown_cast])
            )
            df = df[~mask]

        return df.reset_index(drop=True)

    def save_clean(self, df: pd.DataFrame) -> None:
        """Persist the cleaned dataset to the configured path."""
        clean_path = self.config["data"].get("clean_path")
        if not clean_path:
            raise ValueError("No 'clean_path' configured; nothing to save.")
        out_path = resolve_path(clean_path)
        df.to_csv(out_path, index=False)

    @staticmethod
    def normalize_plot(text: str) -> str:
        """Basic plot text cleaning: collapse whitespace, keep original case
        (case-preserved text reads better in retrieved contexts)."""
        text = re.sub(r"\s+", " ", text)
        return text.strip()
