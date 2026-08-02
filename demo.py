"""Narrated end-to-end demo on the built-in synthetic dataset."""

from __future__ import annotations

import pandas as pd

from .data import generate_synthetic
from .engines import RecommendationEngine
from .evaluate import evaluate


def _show(title: str, df: pd.DataFrame) -> None:
    pd.set_option("display.width", 160)
    pd.set_option("display.max_colwidth", 26)
    df = df.copy()
    if "genres" in df.columns:
        df["genres"] = df["genres"].apply(
            lambda g: ", ".join(g) if isinstance(g, list) else g)
    print(f"\n{'-' * 78}\n{title}\n{'-' * 78}")
    print(df.to_string(index=False))


def run() -> None:
    print("Building synthetic dataset (no internet needed)...")
    ds = generate_synthetic()
    print("  ", ds.summary())

    eng = RecommendationEngine(ds).fit()
    print(f"   weighted-rating params:  C(global mean)={eng._C:.2f}  "
          f"m(vote threshold)={eng._m:.0f}")

    _show("1) POPULARITY  —  top movies by IMDb weighted rating",
          eng.top(n=8))

    _show("2) POPULARITY  —  best in 'Sci-Fi'",
          eng.top(genre="Sci-Fi", n=6))

    seed = eng.movies.iloc[0]["title"]
    _show(f"3) CONTENT-BASED  —  more like {seed!r} "
          f"(genres: {', '.join(eng.movies.iloc[0]['genres'])})",
          eng.similar(seed, n=6, method="content"))

    _show(f"4) COLLABORATIVE  —  people who liked {seed!r} also liked...",
          eng.similar(seed, n=6, method="cf"))

    _show("5) PERSONALIZED  —  recommendations for existing user 'u5' "
          "(matrix factorization)",
          eng.for_user("u5", n=6))

    liked = [eng.movies.iloc[0]["title"], eng.movies.iloc[2]["title"]]
    _show(f"6) HYBRID (cold-start)  —  new user who liked {liked} "
          "(with signal breakdown)",
          eng.recommend(liked=liked, n=6, explain=True))

    print(f"\n{'-' * 78}\n7) EVALUATION  —  holdout metrics on a larger, "
          f"denser dataset\n{'-' * 78}")
    big = generate_synthetic(n_movies=300, n_users=600, density=0.12, seed=3)
    print("   dataset:", big.summary())
    for m in ("mf", "svd", "knn"):
        r = evaluate(big, k=10, method=m)
        print(f"   {m:4s}  RMSE={r['rmse']:<6} "
              f"precision@10={r['precision_at_k']:<6} "
              f"recall@10={r['recall_at_k']}")
    print("\n   (Matrix factorization wins; on real MovieLens-100k these "
          "metrics are substantially higher.)")

    print(f"\n{'=' * 78}\nDone. Try the CLI, e.g.:\n"
          "   python -m recommender.cli top --genre Drama -n 10\n"
          "   python -m recommender.cli recommend --liked \"Wild Mirage\" --explain\n"
          f"{'=' * 78}")


if __name__ == "__main__":
    run()
