"""
Command-line interface.

Examples
--------
  # Run everything on the built-in synthetic dataset:
  python -m recommender.cli demo

  # Top movies overall / by genre (uses the IMDb weighted-rating formula):
  python -m recommender.cli top --genre Sci-Fi -n 10

  # "More like this":
  python -m recommender.cli similar "Wild Mirage" -n 10

  # Personalized for an existing user (needs per-user ratings):
  python -m recommender.cli for-user u5 -n 10

  # Hybrid recommendations for a NEW user from a few liked titles:
  python -m recommender.cli recommend --liked "Wild Mirage" "Savage Odyssey" --explain

  # Evaluate the collaborative filter:
  python -m recommender.cli evaluate

Data sources (default is --synthetic):
  --imdb BASICS RATINGS      IMDb title.basics.tsv[.gz] + title.ratings.tsv[.gz]
  --movielens FOLDER         MovieLens folder (movies.csv, ratings.csv)
  --csv FOLDER               folder with movies.csv[, ratings.csv, reviews.csv]
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import (
    RecommendationEngine,
    generate_synthetic,
    load_csv_folder,
    load_imdb,
    load_movielens,
    load_tmdb_reviews,
)
from .evaluate import evaluate as run_eval


def _load_dataset(args):
    if args.imdb:
        ds = load_imdb(args.imdb[0], args.imdb[1], min_votes=args.min_votes)
    elif args.movielens:
        ds = load_movielens(args.movielens)
    elif args.csv:
        ds = load_csv_folder(args.csv)
    else:
        ds = generate_synthetic()
    if args.reviews:
        ds.reviews = load_tmdb_reviews(args.reviews)
    return ds


def _show(df: pd.DataFrame) -> None:
    pd.set_option("display.width", 160)
    pd.set_option("display.max_colwidth", 30)
    pd.set_option("display.max_rows", 100)
    if "genres" in df.columns:
        df = df.copy()
        df["genres"] = df["genres"].apply(
            lambda g: ", ".join(g) if isinstance(g, list) else g)
    print(df.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    # Data-source flags live on a parent so every subcommand accepts them,
    # e.g.  `recommender top --csv data`  or  `recommender for-user u5 --movielens ml/`.
    src = argparse.ArgumentParser(add_help=False)
    src.add_argument("--imdb", nargs=2, metavar=("BASICS", "RATINGS"))
    src.add_argument("--movielens", metavar="FOLDER")
    src.add_argument("--csv", metavar="FOLDER")
    src.add_argument("--reviews", metavar="CSV", help="attach a reviews/comments CSV")
    src.add_argument("--synthetic", action="store_true",
                     help="use the built-in synthetic dataset (default)")
    src.add_argument("--min-votes", type=int, default=0)

    p = argparse.ArgumentParser(
        prog="recommender",
        description="Movie recommendation engine (ratings + reviews + likes).")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("top", parents=[src],
                       help="best movies by weighted rating")
    s.add_argument("--genre", default=None)
    s.add_argument("-n", type=int, default=10)

    s = sub.add_parser("similar", parents=[src],
                       help="movies similar to a given title")
    s.add_argument("title")
    s.add_argument("-n", type=int, default=10)
    s.add_argument("--method", choices=["content", "cf"], default="content")

    s = sub.add_parser("for-user", parents=[src],
                       help="personalized recs for an existing user")
    s.add_argument("user_id")
    s.add_argument("-n", type=int, default=10)
    s.add_argument("--method", choices=["mf", "svd", "knn"], default="mf")

    s = sub.add_parser("recommend", parents=[src],
                       help="hybrid recs (new user from liked titles)")
    s.add_argument("--liked", nargs="+", default=None)
    s.add_argument("--user", default=None)
    s.add_argument("-n", type=int, default=10)
    s.add_argument("--explain", action="store_true")

    sub.add_parser("evaluate", parents=[src],
                   help="holdout precision@k / recall@k / RMSE")
    sub.add_parser("demo", help="run a full narrated demo on synthetic data")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "demo":
        from .demo import run as run_demo
        run_demo()
        return 0

    ds = _load_dataset(args)
    print(f"Loaded: {ds.summary()}\n")
    eng = RecommendationEngine(ds).fit()

    try:
        if args.cmd == "top":
            _show(eng.top(genre=args.genre, n=args.n,
                          min_votes=args.min_votes or None))
        elif args.cmd == "similar":
            _show(eng.similar(args.title, n=args.n, method=args.method))
        elif args.cmd == "for-user":
            _show(eng.for_user(args.user_id, n=args.n, method=args.method))
        elif args.cmd == "recommend":
            _show(eng.recommend(liked=args.liked, user_id=args.user,
                                n=args.n, explain=args.explain))
        elif args.cmd == "evaluate":
            for m in ("mf", "svd", "knn"):
                print(run_eval(ds, k=10, method=m))
    except (KeyError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
