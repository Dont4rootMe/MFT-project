import logging
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureSelector:
    """Feature selection logic split into per-currency and post-merge stages."""

    def __init__(self, config: Dict | None = None) -> None:
        self.config = config or {}
        self.per_cfg = self.config.get("per_currency", {})
        self.post_cfg = self.config.get("post_merge", {})

    # ------------------------------------------------------------------
    # Per-currency selection
    # ------------------------------------------------------------------
    def select_per_currency(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply feature selection to a single-currency dataframe."""
        if not self.per_cfg.get("enabled", False):
            return data

        method = self.per_cfg.get("method", "")
        if method == "variance_corr":
            return self._variance_corr_filter(data)
        if method == "pca":
            return self._pca_reduction(data)
        if method == "topk":
            return self._topk_features(data)

        logger.warning(f"Unknown per-currency selection method: {method}")
        return data

    def _get_base_columns(self, df: pd.DataFrame):
        date_cols = [c for c in df.columns if "date" in c.lower()]
        core_cols = [
            c
            for c in df.columns
            if any(k in c.lower() for k in ["open", "high", "low", "close", "volume"])
        ]
        feature_cols = [
            c
            for c in df.select_dtypes(include=[np.number]).columns
            if c not in core_cols and c not in date_cols
        ]
        return date_cols, core_cols, feature_cols

    def _variance_corr_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        date_cols, core_cols, feature_cols = self._get_base_columns(df)
        if not feature_cols:
            return df

        vt = VarianceThreshold(
            threshold=self.per_cfg.get("variance_threshold", 0.0)
        )
        feat = df[feature_cols].fillna(0)
        vt.fit(feat)
        feat = feat.loc[:, vt.get_support()]

        corr_threshold = self.per_cfg.get("correlation_threshold", 0.95)
        corr = feat.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if any(upper[c] > corr_threshold)]
        feat = feat.drop(columns=to_drop)

        max_features = self.per_cfg.get("max_features")
        if max_features and feat.shape[1] > max_features:
            variances = feat.var().sort_values(ascending=False)
            feat = feat[variances.index[: max_features]]

        result = pd.concat([df[date_cols + core_cols], feat], axis=1)
        logger.info(
            f"variance_corr selection: {len(feature_cols)} -> {feat.shape[1]} features"
        )
        return result

    def _pca_reduction(self, df: pd.DataFrame) -> pd.DataFrame:
        date_cols, core_cols, feature_cols = self._get_base_columns(df)
        if not feature_cols:
            return df

        feat = df[feature_cols].fillna(0)
        n_components = self.per_cfg.get("n_components", 5)
        n_components = min(n_components, feat.shape[1])
        pca = PCA(n_components=n_components)
        comps = pca.fit_transform(feat)
        comp_cols = [f"pca_{i}" for i in range(comps.shape[1])]
        comps_df = pd.DataFrame(comps, columns=comp_cols, index=df.index)

        result = pd.concat([df[date_cols + core_cols], comps_df], axis=1)
        logger.info(
            f"pca reduction: {len(feature_cols)} -> {comps_df.shape[1]} components"
        )
        return result

    def _topk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        date_cols, core_cols, feature_cols = self._get_base_columns(df)
        if not feature_cols:
            return df

        k = self.per_cfg.get("k") or self.per_cfg.get("max_features")
        if not k:
            return df

        feat = df[feature_cols].fillna(0)
        variances = feat.var().sort_values(ascending=False)
        feat = feat[variances.index[:k]]

        result = pd.concat([df[date_cols + core_cols], feat], axis=1)
        logger.info(f"topk selection: {len(feature_cols)} -> {feat.shape[1]} features")
        return result

    # ------------------------------------------------------------------
    # Post-merge selection
    # ------------------------------------------------------------------
    def select_post_merge(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.post_cfg.get("enabled", False):
            return data

        method = self.post_cfg.get("method", "none")
        logger.warning(f"Post-merge selection method '{method}' not implemented; skipping")
        return data

