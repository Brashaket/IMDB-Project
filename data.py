"""
Data loading and normalization.

The engine works on three normalized tables (pandas DataFrames):

  movies   : movie_id, title, year, genres (list[str]), avg_rating, num_votes,
             overview (str), runtime (float, minutes)
  ratings  : user_id, movie_id, rating (float 0.5-10)          [per-user, optional]
  reviews  : movie_id, user_id, text (str), likes (int)         [comments, optional]

Real-world adapters convert common public formats into these tables:

  * IMDb Non-Commercial Datasets  -> load_imdb()   (aggregate ratings + metadata)
      https://developer.imdb.com/non-commercial-datasets/
      Files: title.basics.tsv.gz, title.ratings.tsv.gz
      NOTE: IMDb does NOT distribute per-user ratings, "likes", or review text,
      and scraping the site violates its Terms of Service. Use MovieLens for
      per-user ratings and the TMDB API for review text.

  * MovieLens                     -> load_movielens()  (per-user ratings + links)
      https://grouplens.org/datasets/movielens/  (ml-latest-small etc.)
      Links file maps MovieLens movieId <-> imdbId <-> tmdbId.

  * TMDB API export               -> load_tmdb_reviews()  (review text = "comments")
      https://developer.themoviedb.org/  (GET /movie/{id}/reviews)

If you have none of the above, `generate_synthetic()` builds a coherent dataset
so the engine is demonstrable immediately and offline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    """Bundle of the three normalized tables."""
    movies: pd.DataFrame
    ratings: pd.DataFrame            # may be empty
    reviews: pd.DataFrame            # may be empty

    def summary(self) -> str:
        return (
            f"{len(self.movies):,} movies | "
            f"{len(self.ratings):,} user ratings"
            f"{' from ' + str(self.ratings['user_id'].nunique()) + ' users' if len(self.ratings) else ''} | "
            f"{len(self.reviews):,} reviews/comments"
        )


# --------------------------------------------------------------------------- #
# Real-data adapters
# --------------------------------------------------------------------------- #
def load_imdb(basics_path: str, ratings_path: str,
              title_types=("movie",), min_votes: int = 0) -> Dataset:
    """Load IMDb Non-Commercial Datasets (title.basics + title.ratings)."""
    basics = pd.read_csv(
        basics_path, sep="\t", na_values="\\N", low_memory=False,
        dtype={"startYear": "string", "runtimeMinutes": "string"},
    )
    ratings = pd.read_csv(ratings_path, sep="\t", na_values="\\N")

    if title_types:
        basics = basics[basics["titleType"].isin(title_types)]

    df = basics.merge(ratings, on="tconst", how="inner")
    movies = pd.DataFrame({
        "movie_id": df["tconst"],
        "title": df["primaryTitle"],
        "year": pd.to_numeric(df["startYear"], errors="coerce"),
        "genres": df["genres"].fillna("").apply(
            lambda g: [x for x in g.split(",") if x] if g else []),
        "avg_rating": df["averageRating"].astype(float),
        "num_votes": df["numVotes"].astype(int),
        "overview": "",                       # not in IMDb datasets
        "runtime": pd.to_numeric(df["runtimeMinutes"], errors="coerce"),
    })
    movies = movies[movies["num_votes"] >= min_votes].reset_index(drop=True)
    empty_r = pd.DataFrame(columns=["user_id", "movie_id", "rating"])
    empty_rev = pd.DataFrame(columns=["movie_id", "user_id", "text", "likes"])
    return Dataset(movies, empty_r, empty_rev)


def load_movielens(folder: str) -> Dataset:
    """Load a MovieLens export folder (movies.csv, ratings.csv, [links.csv])."""
    movies_raw = pd.read_csv(os.path.join(folder, "movies.csv"))
    ratings_raw = pd.read_csv(os.path.join(folder, "ratings.csv"))

    # MovieLens titles look like "Toll (1995)"; split off the year.
    year = movies_raw["title"].str.extract(r"\((\d{4})\)").iloc[:, 0]
    title = movies_raw["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)

    movies = pd.DataFrame({
        "movie_id": movies_raw["movieId"].astype(str),
        "title": title,
        "year": pd.to_numeric(year, errors="coerce"),
        "genres": movies_raw["genres"].apply(
            lambda g: [] if g == "(no genres listed)" else g.split("|")),
        "overview": "",
        "runtime": np.nan,
    })

    ratings = pd.DataFrame({
        "user_id": ratings_raw["userId"].astype(str),
        "movie_id": ratings_raw["movieId"].astype(str),
        # MovieLens is 0.5-5; scale to a 0.5-10 axis to match IMDb-style ratings.
        "rating": ratings_raw["rating"].astype(float) * 2.0,
    })

    # Derive aggregate rating/vote count from the per-user ratings.
    agg = ratings.groupby("movie_id")["rating"].agg(["mean", "count"])
    movies = movies.merge(
        agg.rename(columns={"mean": "avg_rating", "count": "num_votes"}),
        left_on="movie_id", right_index=True, how="left")
    movies["avg_rating"] = movies["avg_rating"].fillna(0.0)
    movies["num_votes"] = movies["num_votes"].fillna(0).astype(int)

    empty_rev = pd.DataFrame(columns=["movie_id", "user_id", "text", "likes"])
    return Dataset(movies.reset_index(drop=True), ratings, empty_rev)


def load_tmdb_reviews(csv_path: str) -> pd.DataFrame:
    """Load a reviews CSV (columns: movie_id, [user_id], content/text, [likes]).

    Returns a `reviews` table you can attach to an existing Dataset via
    `dataset.reviews = load_tmdb_reviews(...)`.
    """
    raw = pd.read_csv(csv_path)
    text_col = "content" if "content" in raw.columns else "text"
    return pd.DataFrame({
        "movie_id": raw["movie_id"].astype(str),
        "user_id": raw.get("user_id", pd.Series([""] * len(raw))).astype(str),
        "text": raw[text_col].astype(str),
        "likes": pd.to_numeric(raw.get("likes", 0), errors="coerce").fillna(0).astype(int),
    })


def load_csv_folder(folder: str) -> Dataset:
    """Load a folder written by `generate_synthetic().save()` (or your own CSVs)."""
    movies = pd.read_csv(os.path.join(folder, "movies.csv"))
    movies["genres"] = movies["genres"].fillna("").apply(
        lambda g: [x for x in str(g).split("|") if x] if g else [])
    movies["movie_id"] = movies["movie_id"].astype(str)

    def _read(name, cols):
        p = os.path.join(folder, name)
        if os.path.exists(p):
            d = pd.read_csv(p)
            for c in ("user_id", "movie_id"):
                if c in d.columns:
                    d[c] = d[c].astype(str)
            return d
        return pd.DataFrame(columns=cols)

    ratings = _read("ratings.csv", ["user_id", "movie_id", "rating"])
    reviews = _read("reviews.csv", ["movie_id", "user_id", "text", "likes"])
    return Dataset(movies, ratings, reviews)


# --------------------------------------------------------------------------- #
# Synthetic data generator (coherent so recommendations are non-random)
# --------------------------------------------------------------------------- #
_GENRES = ["Action", "Adventure", "Comedy", "Crime", "Drama", "Fantasy",
           "Horror", "Mystery", "Romance", "Sci-Fi", "Thriller", "Animation"]

_TITLE_A = ["Silent", "Crimson", "Broken", "Eternal", "Hidden", "Final",
            "Distant", "Savage", "Golden", "Frozen", "Burning", "Last",
            "Neon", "Velvet", "Iron", "Midnight", "Pale", "Wild"]
_TITLE_B = ["Horizon", "Empire", "Shadow", "Requiem", "Legacy", "Voyage",
            "Protocol", "Kingdom", "Paradox", "Echo", "Covenant", "Mirage",
            "Reckoning", "Ascent", "Descent", "Sanctuary", "Odyssey", "Gambit"]

_POS_WORDS = ["brilliant", "gripping", "beautiful", "unforgettable", "superb",
              "captivating", "loved it", "a masterpiece", "thrilling", "moving"]
_NEG_WORDS = ["boring", "disappointing", "predictable", "a mess", "dull",
              "overrated", "forgettable", "tedious", "weak", "avoid it"]
_NEU_WORDS = ["watchable", "fine", "okay", "average", "nothing special",
              "decent enough", "passable", "middling"]


def generate_synthetic(n_movies: int = 220, n_users: int = 350,
                       density: float = 0.06, seed: int = 7) -> Dataset:
    """Create a coherent synthetic dataset.

    Each user has latent genre affinities; ratings, reviews and "likes" are all
    generated consistently with those affinities and each movie's quality, so
    collaborative filtering, content similarity and sentiment all find real
    structure (recommendations won't be random noise).
    """
    rng = np.random.default_rng(seed)

    # --- Movies ---
    titles, used = [], set()
    while len(titles) < n_movies:
        t = f"{rng.choice(_TITLE_A)} {rng.choice(_TITLE_B)}"
        if t not in used:
            used.add(t)
            titles.append(t)

    movie_genres, quality = [], []
    for _ in range(n_movies):
        k = rng.integers(1, 4)
        movie_genres.append(sorted(rng.choice(_GENRES, size=k, replace=False).tolist()))
        quality.append(np.clip(rng.normal(6.4, 1.3), 1.0, 9.5))  # latent quality
    quality = np.array(quality)

    # Vote counts follow a heavy-tailed (log-normal) distribution.
    num_votes = np.clip(rng.lognormal(mean=8.0, sigma=1.4, size=n_movies),
                        20, None).astype(int)
    avg_rating = np.clip(quality + rng.normal(0, 0.25, n_movies), 1.0, 10.0).round(1)

    movies = pd.DataFrame({
        "movie_id": [f"m{1000 + i}" for i in range(n_movies)],
        "title": titles,
        "year": rng.integers(1972, 2025, n_movies),
        "genres": movie_genres,
        "avg_rating": avg_rating,
        "num_votes": num_votes,
        "overview": ["A " + " ".join(g).lower() + " story." for g in movie_genres],
        "runtime": rng.integers(82, 168, n_movies).astype(float),
    })

    # --- Users with latent genre affinities ---
    #  affinity[user, genre] in [0, 1]
    genre_idx = {g: i for i, g in enumerate(_GENRES)}
    affinity = rng.beta(1.4, 3.0, size=(n_users, len(_GENRES)))
    movie_gvec = np.zeros((n_movies, len(_GENRES)))
    for mi, gl in enumerate(movie_genres):
        for g in gl:
            movie_gvec[mi, genre_idx[g]] = 1.0

    # --- Ratings ---
    rating_rows = []
    n_interactions = int(n_users * n_movies * density)
    user_pick = rng.integers(0, n_users, n_interactions)
    movie_pick = rng.integers(0, n_movies, n_interactions)
    seen = set()
    for u, mi in zip(user_pick, movie_pick):
        if (u, mi) in seen:
            continue
        seen.add((u, mi))
        genres_here = movie_gvec[mi] > 0
        taste = affinity[u, genres_here].mean() if genres_here.any() else 0.3
        # Rating blends movie quality with the user's taste for its genres.
        base = 0.55 * quality[mi] + 0.45 * (2.0 + 8.0 * taste)
        r = float(np.clip(base + rng.normal(0, 0.8), 0.5, 10.0))
        r = round(r * 2) / 2.0  # half-star resolution
        rating_rows.append((f"u{u}", movies.at[mi, "movie_id"], r))
    ratings = pd.DataFrame(rating_rows, columns=["user_id", "movie_id", "rating"])

    # --- Reviews (comments) + likes, sentiment-correlated with rating ---
    review_rows = []
    for u, mid, r in rating_rows:
        if rng.random() > 0.4:            # only ~40% of ratings get a written review
            continue
        if r >= 7.5:
            phrase = rng.choice(_POS_WORDS)
        elif r <= 4.5:
            phrase = rng.choice(_NEG_WORDS)
        else:
            phrase = rng.choice(_NEU_WORDS)
        text = f"Honestly {phrase}. " + rng.choice(
            ["Would watch again.", "Not for everyone.", "Glad I saw it.",
             "Your mileage may vary.", "The cast carries it.", "Ending stuck with me."])
        # "likes" = helpfulness votes; more extreme opinions get more engagement.
        likes = int(rng.poisson(2 + 6 * abs(r - 6) / 5))
        review_rows.append((mid, u, text, likes))
    reviews = pd.DataFrame(review_rows, columns=["movie_id", "user_id", "text", "likes"])

    return Dataset(movies, ratings, reviews)


def save_dataset(ds: Dataset, folder: str) -> None:
    """Persist a Dataset to CSVs (genres joined with '|')."""
    os.makedirs(folder, exist_ok=True)
    m = ds.movies.copy()
    m["genres"] = m["genres"].apply(lambda g: "|".join(g))
    m.to_csv(os.path.join(folder, "movies.csv"), index=False)
    if len(ds.ratings):
        ds.ratings.to_csv(os.path.join(folder, "ratings.csv"), index=False)
    if len(ds.reviews):
        ds.reviews.to_csv(os.path.join(folder, "reviews.csv"), index=False)


if __name__ == "__main__":
    ds = generate_synthetic()
    print(ds.summary())
    print("\nSample movies:")
    print(ds.movies[["title", "year", "genres", "avg_rating", "num_votes"]].head())
    print("\nSample reviews:")
    print(ds.reviews.head()[["movie_id", "text", "likes"]].to_string(index=False))
