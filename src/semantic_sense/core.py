# core.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Union, Tuple

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler   # << add this
from semantic_sense.utils import row_to_text
from semantic_sense.model_loader import load_local_model


class AnomalyDetector:
    """
    Semantic anomaly detector using row->text->embedding and centroid distance.

    Modes:
      - text   : use ONLY row_to_text strings for embeddings (default)
      - hybrid : concatenate scaled numeric columns to the text embedding

    Scoring:
      - L2-normalize embeddings
      - centroid = mean(unit vectors)
      - score = 1 - cosine_sim(row, centroid)
      - rank by score; top_percent flagged as anomalies
    """

    def __init__(
        self,
        mode: str = "text",
        numeric_weight: float = 1.0,
        exclude_cols: list[str] | None = None,
    ):
        if mode not in {"text", "hybrid"}:
            raise ValueError("mode must be 'text' or 'hybrid'")
        self.mode = mode
        self.numeric_weight = float(numeric_weight)
        self.exclude_cols = set(exclude_cols or [])

        self.model = load_local_model()
        self._num_scaler = StandardScaler()

    # --------------------------
    # Embedding helpers
    # --------------------------
    def _texts_from_df(self, df: pd.DataFrame) -> List[str]:
        """Convert every row into a single 'Col: val' string."""
        return [
            row_to_text(row)
            for row in tqdm(
                df.itertuples(index=False),
                total=len(df),
                desc="Converting rows to text"
            )
        ]

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of strings -> (n, dim) numpy array."""
        emb = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        if isinstance(emb, np.ndarray) and emb.ndim == 1:
            emb = emb.reshape(1, -1)
        return emb

    def _l2norm(self, X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.clip(norms, eps, None)

    def _numeric_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Pick numeric columns (excluding any in exclude_cols) and scale them."""
        num_cols = (
            df.select_dtypes(include=[np.number])
              .columns.difference(self.exclude_cols)
        )
        if len(num_cols) == 0:
            return np.empty((len(df), 0), dtype=float)
        num_vals = df[num_cols].to_numpy()
        num_scaled = self._num_scaler.fit_transform(num_vals)
        # Weight numeric block (text will be unweighted = 1.0)
        if self.numeric_weight != 1.0 and num_scaled.size:
            num_scaled = num_scaled * self.numeric_weight
        return num_scaled

    # --------------------------
    # Public embedding APIs
    # --------------------------
    def _embed_rows(self, data: Union[pd.DataFrame, pd.Series]) -> np.ndarray:
        """
        Convert DataFrame OR single Series into embeddings.
        - Build row_to_text strings
        - Embed text
        - If mode='hybrid', horizontally concatenate scaled numeric features
        """
        if isinstance(data, pd.Series):
            texts = [row_to_text(data)]
            text_emb = self._embed_texts(texts)
            if self.mode == "text":
                return text_emb
            # For single row hybrid, numeric concat also works:
            df1 = data.to_frame().T
            num_scaled = self._numeric_matrix(df1)
            return np.hstack([text_emb, num_scaled])

        elif isinstance(data, pd.DataFrame):
            texts = self._texts_from_df(data)
            text_emb = self._embed_texts(texts)
            if self.mode == "text":
                return text_emb
            # HYBRID: concat scaled numeric block
            num_scaled = self._numeric_matrix(data)
            return np.hstack([text_emb, num_scaled])

        else:
            raise TypeError("Input must be a pandas DataFrame or Series")

    def embed_series(self, row: pd.Series) -> np.ndarray:
        """Embed a single row (Series) -> (1, dim) (honors mode)."""
        if not isinstance(row, pd.Series):
            raise TypeError(f"Expected pandas.Series, got {type(row)}")
        return self._embed_rows(row)

    def embed_column(self, df: pd.DataFrame, colname: str) -> np.ndarray:
        """Embed one text column directly (bypass row_to_text)."""
        if colname not in df.columns:
            raise ValueError(f"Column {colname} not found in DataFrame")
        texts = df[colname].astype(str).tolist()
        return self._embed_texts(texts)

    # --------------------------
    # Anomaly detection (centroid)
    # --------------------------
    def detect(self, df: pd.DataFrame, top_percent: float = 5.0, return_embeddings: bool = False):

        """
        Rank rows by cosine distance from centroid of (text or hybrid) embeddings.
        """
        # Build embeddings according to mode
        emb = self._embed_rows(df)

        # Normalize and compute centroid
        emb = self._l2norm(emb)
        centroid = emb.mean(axis=0, keepdims=True)
        centroid = centroid / np.linalg.norm(centroid)

        # Cosine distance to centroid
        scores = 1.0 - (emb @ centroid.T).ravel()

        # Rank (1 = most anomalous)
        order = np.argsort(-scores)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(scores) + 1)

        # Threshold by percentile
        threshold = np.percentile(scores, 100 - top_percent)
        is_anomaly = (scores >= threshold).astype(int)

        # Include the exact row_text used for text component
        texts = self._texts_from_df(df)

        out = df.copy()
        out["row_text"] = texts
        out["centroid_distance"] = scores
        out["rank"] = ranks
        out["is_anomaly"] = is_anomaly
        out = out.sort_values("centroid_distance", ascending=False).reset_index(drop=True)

        return (out, emb) if return_embeddings else out
