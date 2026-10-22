"""Student recommendation providers for CineLab."""
from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

from ..contracts import (
    Algorithm,
    RecommendationItem,
    RecommendationRequest,
    RecommendationResult,
)
from ..data import DATASET, SPLIT, PACKAGE, FrozenData

MODELS_DIR = Path("/app/models/business")


def _load_csv(path: Path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _now():
    return datetime.now(timezone.utc)


def _make_items(movie_scores: list[tuple[str, float]], limit: int, algo: Algorithm, version: str):
    """Build RecommendationResult items, sorted by score desc then movie_id asc."""
    # Deduplicate
    seen = set()
    unique = []
    for mid, score in movie_scores:
        if mid not in seen:
            seen.add(mid)
            unique.append((mid, score))
    unique.sort(key=lambda x: (-x[1], int(x[0])))
    items = []
    for rank, (mid, score) in enumerate(unique[:limit], 1):
        items.append(RecommendationItem(
            movie_id=mid, rank=rank, algorithm=algo,
            model_version=version, score=score,
        ))
    return items


class PopularProvider:
    """Most-rated movies overall."""
    def __init__(self):
        self.model_manifest_path = str(MODELS_DIR / "popular" / "model_manifest.json")
        self._top = None

    def _load(self):
        if self._top is None:
            rows = _load_csv(MODELS_DIR / "popular" / "popular_top.csv")
            self._top = [(r["movie_id"], float(r["score"])) for r in rows]

    def recommend(self, request: RecommendationRequest) -> RecommendationResult:
        self._load()
        items = _make_items(self._top, request.limit, Algorithm.popular, "popular-v1")
        return RecommendationResult(
            algorithm=Algorithm.popular,
            model_version="popular-v1",
            dataset_version=DATASET,
            split_version=SPLIT,
            runtime_mode=request.runtime_mode,
            generated_at=_now(),
            items=items,
        )


class ContentProvider:
    """Genre-based item similarity."""
    def __init__(self):
        self.model_manifest_path = str(MODELS_DIR / "content" / "model_manifest.json")
        self._sim = None

    def _load(self):
        if self._sim is None:
            self._sim = defaultdict(list)
            for r in _load_csv(MODELS_DIR / "content" / "item_similarity.csv"):
                self._sim[r["movie_id"]].append((r["similar_id"], float(r["score"])))

    def recommend(self, request: RecommendationRequest) -> RecommendationResult:
        self._load()
        from ..business import read_history
        history = read_history(request)
        # Score = weighted sum of similarities to user's liked movies
        scores: dict[str, float] = defaultdict(float)
        rated = {fb.movie_id: float(fb.rating) for fb in history.ratings}
        for mid, rating in rated.items():
            if rating < 3.0:
                continue
            weight = rating - 3.0  # only positive signal
            for sim_id, sim_score in self._sim.get(mid, []):
                if sim_id not in rated:
                    scores[sim_id] += sim_score * weight
        items = _make_items(list(scores.items()), request.limit, Algorithm.content, "content-v1")
        return RecommendationResult(
            algorithm=Algorithm.content,
            model_version="content-v1",
            dataset_version=DATASET,
            split_version=SPLIT,
            runtime_mode=request.runtime_mode,
            generated_at=_now(),
            items=items,
        )


class ItemCFProvider:
    """Item-based collaborative filtering."""
    def __init__(self):
        self.model_manifest_path = str(MODELS_DIR / "itemcf" / "model_manifest.json")
        self._sim = None

    def _load(self):
        if self._sim is None:
            self._sim = defaultdict(list)
            for r in _load_csv(MODELS_DIR / "itemcf" / "itemcf_sim.csv"):
                self._sim[r["movie_id"]].append((r["similar_id"], float(r["score"])))

    def recommend(self, request: RecommendationRequest) -> RecommendationResult:
        self._load()
        from ..business import read_history
        history = read_history(request)
        scores: dict[str, float] = defaultdict(float)
        rated = {fb.movie_id: float(fb.rating) for fb in history.ratings}
        for mid, rating in rated.items():
            if rating < 3.0:
                continue
            weight = rating
            for sim_id, sim_score in self._sim.get(mid, []):
                if sim_id not in rated:
                    scores[sim_id] += sim_score * weight
        items = _make_items(list(scores.items()), request.limit, Algorithm.itemcf, "itemcf-v1")
        return RecommendationResult(
            algorithm=Algorithm.itemcf,
            model_version="itemcf-v1",
            dataset_version=DATASET,
            split_version=SPLIT,
            runtime_mode=request.runtime_mode,
            generated_at=_now(),
            items=items,
        )


class ALSProvider:
    """Simplified ALS: genre-vector based."""
    def __init__(self):
        self.model_manifest_path = str(MODELS_DIR / "als" / "model_manifest.json")
        self._factors = None

    def _load(self):
        if self._factors is None:
            self._factors = {}
            rows = _load_csv(MODELS_DIR / "als" / "item_factors.csv")
            for r in rows:
                mid = r["movie_id"]
                vec = [float(v) for k, v in r.items() if k.startswith("g")]
                self._factors[mid] = vec

    def recommend(self, request: RecommendationRequest) -> RecommendationResult:
        self._load()
        from ..business import read_history
        history = read_history(request)
        rated = {fb.movie_id: float(fb.rating) for fb in history.ratings}
        # User vector = weighted average of rated item vectors
        import math
        uvec = None
        total_w = 0.0
        for mid, rating in rated.items():
            if rating < 3.0:
                continue
            v = self._factors.get(mid)
            if v is None:
                continue
            w = rating
            if uvec is None:
                uvec = [0.0] * len(v)
            for i in range(len(v)):
                uvec[i] += v[i] * w
            total_w += w
        if uvec is None or total_w == 0:
            # Fallback to popular
            return PopularProvider().recommend(request)
        unorm = math.sqrt(sum(x*x for x in uvec))
        if unorm > 0:
            uvec = [x/unorm for x in uvec]
        # Score all items by dot product
        scores = []
        for mid, ivec in self._factors.items():
            if mid in rated:
                continue
            dot = sum(a*b for a, b in zip(uvec, ivec))
            scores.append((mid, dot))
        items = _make_items(scores, request.limit, Algorithm.als, "als-v1")
        return RecommendationResult(
            algorithm=Algorithm.als,
            model_version="als-v1",
            dataset_version=DATASET,
            split_version=SPLIT,
            runtime_mode=request.runtime_mode,
            generated_at=_now(),
            items=items,
        )


class ColdStartProvider:
    """Diversified popular for new users."""
    def __init__(self):
        self.model_manifest_path = str(MODELS_DIR / "cold_start" / "model_manifest.json")
        self._list = None

    def _load(self):
        if self._list is None:
            rows = _load_csv(MODELS_DIR / "cold_start" / "cold_start_list.csv")
            self._list = [(r["movie_id"], float(r["score"])) for r in rows]

    def recommend(self, request: RecommendationRequest) -> RecommendationResult:
        self._load()
        items = _make_items(self._list, request.limit, Algorithm.cold_start, "coldstart-v1")
        return RecommendationResult(
            algorithm=Algorithm.cold_start,
            model_version="coldstart-v1",
            dataset_version=DATASET,
            split_version=SPLIT,
            runtime_mode=request.runtime_mode,
            generated_at=_now(),
            items=items,
        )
