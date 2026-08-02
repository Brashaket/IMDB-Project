"""
Offline evaluation for the collaborative-filtering recommender.

* rating RMSE   : how well SVD predicts held-out ratings (accuracy)
* precision@k   : of the k recommended items, how many were actually relevant
* recall@k      : of the relevant held-out items, how many were recommended

"Relevant" = a held-out item the user rated >= `like_threshold`. We hold out a
fraction of each user's ratings, refit on the rest, and check whether the
recommender surfaces the held-out likes.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from .data import Dataset
from .engines import RecommendationEngine


def train_test_split_ratings(ratings: pd.DataFrame, test_frac: float = 0.2,
                             seed: int = 0, min_train: int = 3):
    """Per-user holdout split. Users with too few ratings stay entirely in train."""
    rng = np.random.default_rng(seed)
    test_idx = []
    for _, grp in ratings.groupby("user_id"):
        idx = grp.index.to_numpy()
        if len(idx) <= min_train:
            continue
        k = max(1, int(round(len(idx) * test_frac)))
        test_idx.extend(rng.choice(idx, size=min(k, len(idx) - min_train),
                                   replace=False))
    test_mask = ratings.index.isin(test_idx)
    return ratings[~test_mask].copy(), ratings[test_mask].copy()


def evaluate(dataset: Dataset, k: int = 10, test_frac: float = 0.2,
             like_threshold: float = 7.0, seed: int = 0,
             method: str = "mf") -> dict:
    """Run a holdout evaluation. Returns a metrics dict."""
    if len(dataset.ratings) == 0:
        raise ValueError("Evaluation needs per-user ratings.")

    train_r, test_r = train_test_split_ratings(dataset.ratings, test_frac, seed)

    train_ds = copy.copy(dataset)
    train_ds.ratings = train_r
    eng = RecommendationEngine(train_ds).fit()

    # --- rating RMSE on held-out ratings ---
    preds, actuals = [], []
    for u, mid, r in test_r[["user_id", "movie_id", "rating"]].itertuples(index=False):
        if u in eng._user_to_row and mid in eng._id_to_row:
            ui, mi = eng._user_to_row[u], eng._id_to_row[mid]
            if method == "mf":
                p = float(eng._mf.predict_user(ui)[mi])
            else:
                p = float(eng._user_factors[ui] @ eng._item_factors[mi]
                          + eng._global_mean)
            preds.append(np.clip(p, 0.5, 10.0))
            actuals.append(r)
    rmse = float(np.sqrt(np.mean((np.array(preds) - np.array(actuals)) ** 2))) \
        if preds else float("nan")

    # --- precision@k / recall@k ---
    relevant = (test_r[test_r["rating"] >= like_threshold]
                .groupby("user_id")["movie_id"].apply(set).to_dict())
    precisions, recalls, covered = [], [], 0
    for u, rel_items in relevant.items():
        if u not in eng._user_to_row:
            continue
        try:
            recs = eng.for_user(u, n=k, method=method)
        except Exception:
            continue
        rec_ids = set()
        for t in recs["title"]:
            try:
                rec_ids.add(eng.resolve(t))
            except KeyError:
                pass
        hits = len(rec_ids & rel_items)
        precisions.append(hits / k)
        recalls.append(hits / len(rel_items))
        covered += 1

    return {
        "method": method,
        "k": k,
        "n_test_ratings": len(actuals),
        "rmse": round(rmse, 3),
        "precision_at_k": round(float(np.mean(precisions)), 3) if precisions else 0.0,
        "recall_at_k": round(float(np.mean(recalls)), 3) if recalls else 0.0,
        "users_evaluated": covered,
    }


if __name__ == "__main__":
    from .data import generate_synthetic
    ds = generate_synthetic(n_movies=300, n_users=600, density=0.12, seed=3)
    print("DATA:", ds.summary())
    for m in ("mf", "svd", "knn"):
        print(evaluate(ds, k=10, method=m))
