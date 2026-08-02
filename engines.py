"""
Recommendation engines.

Strategies implemented, all behind one `RecommendationEngine` facade:

  1. Popularity / weighted rating  -- IMDb's Bayesian formula (great cold-start
     baseline; uses aggregate ratings + vote counts, optional genre filter).
  2. Content-based                 -- item-item cosine similarity over genre /
     numeric / TF-IDF(overview) features ("more like this").
  3. Collaborative filtering       -- per-user ratings via item-item KNN and
     TruncatedSVD matrix factorization ("users who liked X..."; "for you").
  4. Hybrid                        -- normalized blend of content + CF +
     popularity + review-sentiment + likes, with tunable weights and an
     explanation of which signal drove each pick.

Design notes
------------
* Content similarity is computed per-query (one matrix-vector product) rather
  than materializing an N x N matrix, so it scales to the full IMDb catalogue.
* Signals that require data you don't have degrade gracefully: no per-user
  ratings -> CF weight is dropped; no reviews -> sentiment weight is dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer, normalize

from .data import Dataset
from .sentiment import score_text


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def weighted_rating(avg: pd.Series, votes: pd.Series,
                    m: float, C: float) -> pd.Series:
    """IMDb Bayesian weighted rating.

        WR = (v / (v + m)) * R + (m / (v + m)) * C

    Pulls movies with few votes toward the global mean C, so a 9.5 from
    12 voters can't outrank a 8.6 from 200,000.
    """
    v = votes.astype(float)
    return (v / (v + m)) * avg + (m / (v + m)) * C


def _minmax(x: np.ndarray) -> np.ndarray:
    """Scale to [0, 1]; flat arrays map to 0.5."""
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(lo) or hi - lo < 1e-12:
        return np.full_like(x, 0.5)
    return (x - lo) / (hi - lo)


class MatrixFactorization:
    """Biased matrix factorization trained on *observed* ratings only.

        r_ui  ~=  mu + b_u + b_i + p_u . q_i

    Unlike TruncatedSVD on a sparse matrix (which imputes unrated cells as 0 and
    collapses predictions toward the mean), this fits only the ratings that
    exist, so it captures genuine per-user taste. Optimized with vectorized
    full-batch gradient steps (scatter-add), which is fast and dependency-free.
    """

    def __init__(self, n_factors: int = 24, n_epochs: int = 40,
                 lr: float = 0.01, reg: float = 0.05, batch_size: int = 1024,
                 seed: int = 0):
        self.k = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.batch_size = batch_size
        self.seed = seed

    def fit(self, uidx: np.ndarray, iidx: np.ndarray, r: np.ndarray,
            n_users: int, n_items: int) -> "MatrixFactorization":
        rng = np.random.default_rng(self.seed)
        self.mu = float(r.mean())
        self.b_u = np.zeros(n_users)
        self.b_i = np.zeros(n_items)
        self.P = rng.normal(0, 0.1, (n_users, self.k))
        self.Q = rng.normal(0, 0.1, (n_items, self.k))

        n = len(r)
        order = np.arange(n)
        lr = self.lr
        for _ in range(self.n_epochs):
            rng.shuffle(order)
            for s in range(0, n, self.batch_size):
                b = order[s:s + self.batch_size]
                uu, ii, rr = uidx[b], iidx[b], r[b]
                bu, bi = self.b_u[uu], self.b_i[ii]          # pre-update reads
                pu, qi = self.P[uu], self.Q[ii]
                err = rr - (self.mu + bu + bi + np.einsum("ij,ij->i", pu, qi))
                # Per-sample SGD updates (summed via scatter-add within batch).
                np.add.at(self.b_u, uu, lr * (err - self.reg * bu))
                np.add.at(self.b_i, ii, lr * (err - self.reg * bi))
                np.add.at(self.P, uu, lr * (err[:, None] * qi - self.reg * pu))
                np.add.at(self.Q, ii, lr * (err[:, None] * pu - self.reg * qi))
            lr *= 0.98                                       # learning-rate decay
        return self

    def predict_user(self, u: int) -> np.ndarray:
        """Predicted rating for user u against every item."""
        return self.mu + self.b_u[u] + self.b_i + self.Q @ self.P[u]


@dataclass
class RecommendationEngine:
    """Fit once, then query. Missing signals are handled automatically."""

    dataset: Dataset
    vote_percentile: float = 0.80        # quantile of vote counts -> m
    svd_components: int = 40
    _fitted: bool = field(default=False, init=False, repr=False)

    # ---- fit ------------------------------------------------------------- #
    def fit(self) -> "RecommendationEngine":
        m = self.dataset.movies.reset_index(drop=True).copy()
        self.movies = m
        self._id_to_row = {mid: i for i, mid in enumerate(m["movie_id"])}
        self._title_to_id = {t.lower(): mid for t, mid in
                             zip(m["title"], m["movie_id"])}

        # --- Popularity: weighted rating ---
        self._C = float(m["avg_rating"][m["num_votes"] > 0].mean() or m["avg_rating"].mean())
        self._m = float(np.quantile(m["num_votes"], self.vote_percentile))
        m["weighted_rating"] = weighted_rating(m["avg_rating"], m["num_votes"],
                                               self._m, self._C)

        # --- Per-movie review sentiment + likes ---
        self._attach_review_signals()

        # --- Content features ---
        self._build_content_features()

        # --- Collaborative filtering ---
        self._build_collaborative()

        self._fitted = True
        return self

    # ---- feature construction ------------------------------------------- #
    def _attach_review_signals(self) -> None:
        m = self.movies
        rev = self.dataset.reviews
        m["sentiment"] = 0.0
        m["n_reviews"] = 0
        m["total_likes"] = 0
        if len(rev):
            rev = rev.copy()
            rev["s"] = rev["text"].astype(str).map(score_text)
            grp = rev.groupby("movie_id").agg(
                sentiment=("s", "mean"),
                n_reviews=("s", "size"),
                total_likes=("likes", "sum"),
            )
            for col in ("sentiment", "n_reviews", "total_likes"):
                mapped = m["movie_id"].map(grp[col])
                m[col] = mapped.fillna(0)
            m["n_reviews"] = m["n_reviews"].astype(int)
            m["total_likes"] = m["total_likes"].astype(int)

    def _build_content_features(self) -> None:
        m = self.movies
        # Genres -> multi-hot
        self._mlb = MultiLabelBinarizer()
        genre_mat = self._mlb.fit_transform(m["genres"])
        genre_sp = sparse.csr_matrix(genre_mat.astype(float))

        # Numeric (year, runtime) -> scaled, down-weighted vs genres
        year = _minmax(pd.to_numeric(m["year"], errors="coerce").fillna(
            m["year"].median() if m["year"].notna().any() else 0).to_numpy())
        runtime = _minmax(pd.to_numeric(m["runtime"], errors="coerce").fillna(
            np.nanmedian(pd.to_numeric(m["runtime"], errors="coerce"))
            if m["runtime"].notna().any() else 0).to_numpy())
        numeric_sp = sparse.csr_matrix(np.vstack([year, runtime]).T * 0.35)

        # Overview -> TF-IDF (contributes only when overviews exist)
        overviews = m["overview"].fillna("").astype(str)
        if overviews.str.len().sum() > 0:
            self._tfidf = TfidfVectorizer(max_features=4000, stop_words="english")
            text_sp = self._tfidf.fit_transform(overviews) * 0.6
        else:
            self._tfidf = None
            text_sp = sparse.csr_matrix((len(m), 0))

        feats = sparse.hstack([genre_sp, numeric_sp, text_sp]).tocsr()
        # Row-normalize so cosine similarity == dot product.
        self._content_feats = normalize(feats, norm="l2", axis=1)

    def _build_collaborative(self) -> None:
        r = self.dataset.ratings
        self._has_cf = len(r) > 0
        if not self._has_cf:
            return

        self._users = pd.Index(r["user_id"].unique())
        self._user_to_row = {u: i for i, u in enumerate(self._users)}
        # Only keep ratings whose movie is in the catalogue.
        r = r[r["movie_id"].isin(self._id_to_row)]
        rows = r["user_id"].map(self._user_to_row).to_numpy()
        cols = r["movie_id"].map(self._id_to_row).to_numpy()
        vals = r["rating"].astype(float).to_numpy()

        n_users, n_items = len(self._users), len(self.movies)
        self._ui = sparse.csr_matrix((vals, (rows, cols)),
                                     shape=(n_users, n_items))
        # Track what each user has already seen (to exclude from recs).
        self._seen = {u: set(cols[rows == i])
                      for u, i in self._user_to_row.items()}

        # Item-item KNN (cosine) on item vectors (items as rows).
        self._item_vectors = normalize(self._ui.T.tocsr(), norm="l2", axis=1)
        n_neighbors = min(50, n_items)
        self._knn = NearestNeighbors(metric="cosine", n_neighbors=n_neighbors)
        self._knn.fit(self._item_vectors)

        self._global_mean = float(vals.mean())

        # Primary CF model: biased matrix factorization on observed ratings.
        self._mf = MatrixFactorization(
            n_factors=min(self.svd_components, min(n_users, n_items) - 1),
        ).fit(rows, cols, vals, n_users, n_items)

        # TruncatedSVD kept as a fast alternative (method='svd').
        centered = self._ui.copy()
        centered.data = centered.data - self._global_mean
        k = min(self.svd_components, min(self._ui.shape) - 1)
        self._svd = TruncatedSVD(n_components=max(2, k), random_state=0)
        self._user_factors = self._svd.fit_transform(centered)     # users x k
        self._item_factors = self._svd.components_.T               # items x k

    # ---- id / title resolution ------------------------------------------ #
    def resolve(self, title_or_id: str) -> str:
        """Accept a movie_id or a (case-insensitive) title; return movie_id."""
        if title_or_id in self._id_to_row:
            return title_or_id
        key = str(title_or_id).lower()
        if key in self._title_to_id:
            return self._title_to_id[key]
        # Fall back to a forgiving substring match.
        hits = self.movies[self.movies["title"].str.lower().str.contains(
            key, regex=False)]
        if len(hits):
            return hits.iloc[0]["movie_id"]
        raise KeyError(f"No movie matching {title_or_id!r}")

    def _row(self, mid: str) -> int:
        return self._id_to_row[mid]

    def _present(self, rows: np.ndarray, cols=None) -> pd.DataFrame:
        cols = cols or ["title", "year", "genres", "avg_rating",
                        "num_votes", "weighted_rating"]
        out = self.movies.iloc[rows][cols].copy()
        return out.reset_index(drop=True)

    # ================================================================== #
    # Public recommendation API
    # ================================================================== #
    def top(self, genre: str | None = None, n: int = 10,
            min_votes: int | None = None) -> pd.DataFrame:
        """Best movies by weighted rating, optionally within a genre."""
        self._check()
        df = self.movies
        if genre:
            mask = df["genres"].apply(lambda gs: genre.lower() in
                                      [g.lower() for g in gs])
            df = df[mask]
        if min_votes:
            df = df[df["num_votes"] >= min_votes]
        cols = ["title", "year", "genres", "avg_rating", "num_votes",
                "weighted_rating"]
        return df.sort_values("weighted_rating", ascending=False)[cols].head(n).reset_index(drop=True)

    def similar(self, title_or_id: str, n: int = 10,
                method: str = "content") -> pd.DataFrame:
        """Movies similar to a given one.

        method='content' uses feature similarity (always available).
        method='cf' uses co-rating patterns (needs per-user ratings).
        """
        self._check()
        mid = self.resolve(title_or_id)
        row = self._row(mid)

        if method == "cf":
            if not self._has_cf:
                raise ValueError("No per-user ratings loaded; use method='content'.")
            sims, idx = self._knn.kneighbors(self._item_vectors[row], n_neighbors=n + 1)
            order = [i for i in idx.ravel() if i != row][:n]
            scores = (1 - sims.ravel())[[list(idx.ravel()).index(i) for i in order]]
            out = self._present(order)
            out.insert(0, "similarity", np.round(scores, 3))
            return out

        # content
        q = self._content_feats[row]
        sims = (self._content_feats @ q.T).toarray().ravel()
        sims[row] = -np.inf
        order = np.argsort(-sims)[:n]
        out = self._present(order)
        out.insert(0, "similarity", np.round(sims[order], 3))
        return out

    def for_user(self, user_id: str, n: int = 10,
                 method: str = "mf") -> pd.DataFrame:
        """Personalized recommendations for an existing user.

        method='mf'  : biased matrix factorization (default, most accurate)
        method='svd' : TruncatedSVD (faster, mean-biased)
        method='knn' : item-item collaborative filtering
        """
        self._check()
        if not self._has_cf:
            raise ValueError("No per-user ratings loaded; try recommend(liked=[...]).")
        if user_id not in self._user_to_row:
            raise KeyError(f"Unknown user {user_id!r}")
        u = self._user_to_row[user_id]
        seen = self._seen.get(user_id, set())

        if method == "mf":
            preds = self._mf.predict_user(u)
        elif method == "svd":
            preds = self._user_factors[u] @ self._item_factors.T + self._global_mean
        elif method == "knn":
            preds = self._knn_user_scores(u)
        else:
            raise ValueError("method must be 'mf', 'svd', or 'knn'")

        preds = preds.copy()
        preds[list(seen)] = -np.inf
        order = np.argsort(-preds)[:n]
        out = self._present(order)
        out.insert(0, "pred_score", np.round(preds[order], 2))
        return out

    def recommend(self, liked=None, user_id: str | None = None, n: int = 10,
                  weights: dict | None = None,
                  explain: bool = False) -> pd.DataFrame:
        """Hybrid recommendation — the main entry point.

        Provide either `liked` (list of titles/ids for a new user, cold-start)
        or `user_id` (an existing user). Blends, per candidate:
            content : similarity to the liked/seed items
            cf      : SVD predicted rating (existing user only)
            popular : weighted rating
            sentiment : mean review sentiment
            likes   : total helpfulness "likes" (log-scaled)
        Weights default sensibly and auto-zero for unavailable signals.
        """
        self._check()
        n_items = len(self.movies)

        default_w = {"content": 1.0, "cf": 1.0, "popular": 0.4,
                     "sentiment": 0.3, "likes": 0.2}
        w = {**default_w, **(weights or {})}

        seed_rows, seen = [], set()
        if liked:
            for t in liked:
                r = self._row(self.resolve(t))
                seed_rows.append(r)
                seen.add(r)
        if user_id is not None and self._has_cf and user_id in self._user_to_row:
            u = self._user_to_row[user_id]
            seen |= set(self._seen.get(user_id, set()))
        else:
            u = None

        comp = {}

        # content: max similarity to any seed item
        if seed_rows:
            seed_mat = self._content_feats[seed_rows]
            sims = (self._content_feats @ seed_mat.T).toarray()     # items x seeds
            comp["content"] = sims.max(axis=1)
        else:
            comp["content"] = np.zeros(n_items)
            w["content"] = 0.0

        # cf: matrix-factorization prediction for the user
        if u is not None:
            comp["cf"] = self._mf.predict_user(u)
        else:
            comp["cf"] = np.zeros(n_items)
            w["cf"] = 0.0

        comp["popular"] = self.movies["weighted_rating"].to_numpy()

        if self.movies["n_reviews"].sum() > 0:
            comp["sentiment"] = self.movies["sentiment"].to_numpy()
        else:
            comp["sentiment"] = np.zeros(n_items)
            w["sentiment"] = 0.0

        if self.movies["total_likes"].sum() > 0:
            comp["likes"] = np.log1p(self.movies["total_likes"].to_numpy())
        else:
            comp["likes"] = np.zeros(n_items)
            w["likes"] = 0.0

        # Normalize each active component to [0,1] and blend.
        total_w = sum(v for k, v in w.items() if v > 0) or 1.0
        score = np.zeros(n_items)
        norm_comp = {}
        for k, vec in comp.items():
            nv = _minmax(vec) if w[k] > 0 else np.zeros(n_items)
            norm_comp[k] = nv
            score += w[k] * nv
        score /= total_w

        score[list(seen)] = -np.inf
        order = np.argsort(-score)[:n]

        out = self._present(order)
        out.insert(0, "score", np.round(score[order], 3))
        if explain:
            for k in ("content", "cf", "popular", "sentiment", "likes"):
                if w[k] > 0:
                    out[k] = np.round(norm_comp[k][order], 2)
        return out

    # ---- internals ------------------------------------------------------- #
    def _knn_user_scores(self, u: int) -> np.ndarray:
        """Item-item KNN prediction: weight items by similarity to the user's
        rated items, scaled by their (mean-centered) ratings."""
        user_row = self._ui[u]
        rated = user_row.indices
        if len(rated) == 0:
            return np.full(len(self.movies), self._global_mean)
        ratings = user_row.data - self._global_mean
        # similarity of every item to each rated item, then weighted sum
        sim_block = (self._item_vectors @ self._item_vectors[rated].T).toarray()
        num = sim_block @ ratings
        den = np.abs(sim_block).sum(axis=1) + 1e-9
        return self._global_mean + num / den

    def _check(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call .fit() before requesting recommendations.")
