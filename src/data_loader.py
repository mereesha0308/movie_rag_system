"""Load the Wiki Movie Plots dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils import resolve_path


class DataLoader:
    """Load raw movie plot data from CSV."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.raw_path = resolve_path(config["data"]["raw_path"])
        self.clean_path = (
            resolve_path(config["data"]["clean_path"]) if "clean_path" in config["data"] else None
        )

    def resolve_subset_path(self) -> Path:
        """Path where the fixed movie subset is cached."""
        subset = self.config["data"].get("subset_path", "data/subset.csv")
        return resolve_path(subset)

    def load_raw(self) -> pd.DataFrame:
        """Load the original (uncleaned) dataset."""
        return pd.read_csv(self.raw_path)

    def load_clean(self) -> pd.DataFrame:
        """Load the cleaned dataset if it exists, otherwise load raw."""
        if self.clean_path is not None and self.clean_path.exists():
            return pd.read_csv(self.clean_path)
        return pd.read_csv(self.raw_path)

    def load(self, clean: bool = True) -> pd.DataFrame:
        """Convenience loader."""
        return self.load_clean() if clean else self.load_raw()
