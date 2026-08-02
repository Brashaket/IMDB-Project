# Movie Recommendation Engine

A multi-strategy movie recommender that turns three signals — **ratings**,
**reviews/comments** (via sentiment), and **likes/interactions** — into
recommendations. It runs immediately on a built-in synthetic dataset and has
clean adapters for real IMDb, MovieLens, and TMDB data.

## An honest note on IMDb data

The engine is designed around IMDb-style signals, but it's worth being clear
about what IMDb actually makes available:

| Signal | Available from IMDb? | Legitimate source used here |
|---|---|---|
| Aggregate rating + vote count + metadata | ✅ Yes | [IMDb Non-Commercial Datasets](https://developer.imdb.com/non-commercial-datasets/) (`title.basics`, `title.ratings`) |
| Per-user ratings (for collaborative filtering) | ❌ Not published | [MovieLens](https://grouplens.org/datasets/movielens/) (links to IMDb IDs) |
| Review text / "comments" | ❌ Not in datasets | [TMDB API](https://developer.themoviedb.org/) `GET /movie/{id}/reviews` |
| "Likes" / helpfulness votes | ❌ Not published | TMDB / your own interaction logs |

IMDb does **not** distribute per-user ratings, likes, or review text, and
scraping the site violates its [Terms of Service](https://www.imdb.com/conditions).
So this project **does not scrape** — it uses the official IMDb datasets for
ratings/metadata and standard public sources (MovieLens, TMDB) for the
per-user, comment, and like signals. Everything is wired so you can drop in
whichever of these you have; missing signals degrade gracefully.

## Quickstart

```bash
pip install -r requirements.txt
python demo.py                 # full narrated demo on synthetic data
```

No internet or downloads needed — a coherent synthetic dataset is generated so
every strategy is demonstrable out of the box.

## What it does

Five strategies behind one `RecommendationEngine`:

1. **Popularity / weighted rating** — IMDb's Bayesian formula
   `WR = (v/(v+m))·R + (m/(v+m))·C`, so a 9.5 from 12 voters can't outrank an
   8.6 from 200,000. Great cold-start baseline; filterable by genre.
2. **Content-based** — cosine similarity over genre / year / runtime /
   TF-IDF(overview) features. "More like this." Computed per-query, so it
   scales to the full IMDb catalogue without an N×N matrix.
3. **Collaborative filtering** — learns from per-user ratings via biased
   **matrix factorization** (`r ≈ μ + b_u + b_i + pᵤ·qᵢ`, trained with
   mini-batch SGD on observed ratings only) plus item-item KNN. "Users who
   liked X…" and personalized "for you."
4. **Sentiment** — a built-in lexicon scorer turns review/comment text into a
   per-movie sentiment signal (this is how "comments" feed the ranking).
5. **Hybrid** — a normalized, tunable blend of all of the above, with an
   `--explain` mode that shows which signal drove each pick.

## CLI

```bash
# Top movies overall / by genre
python -m recommender.cli top --genre Sci-Fi -n 10

# "More like this" (content) or co-rating patterns (cf)
python -m recommender.cli similar "Wild Mirage" -n 10
python -m recommender.cli similar "Wild Mirage" --method cf

# Personalized for an existing user (needs per-user ratings)
python -m recommender.cli for-user u5 -n 10 --method mf

# Hybrid for a NEW user, from a few liked titles, with a signal breakdown
python -m recommender.cli recommend --liked "Wild Mirage" "Savage Odyssey" --explain

# Offline evaluation of the collaborative filter
python -m recommender.cli evaluate
```

Any command accepts a data-source flag (default is the synthetic set):

```bash
python -m recommender.cli top --imdb title.basics.tsv.gz title.ratings.tsv.gz --min-votes 1000
python -m recommender.cli for-user 42 --movielens ml-latest-small/
python -m recommender.cli top --csv data/          # your own movies.csv/ratings.csv/reviews.csv
```

## Python API

```python
from recommender import RecommendationEngine, generate_synthetic

ds = generate_synthetic()                      # or load_imdb(...), load_movielens(...)
eng = RecommendationEngine(ds).fit()

eng.top(genre="Drama", n=10)                   # weighted-rating leaderboard
eng.similar("Wild Mirage", n=10)               # content-based neighbours
eng.for_user("u5", n=10)                       # personalized (matrix factorization)
eng.recommend(liked=["Wild Mirage"], n=10, explain=True)   # hybrid, cold-start
```

### Plugging in real data

```python
from recommender import load_imdb, load_movielens, load_tmdb_reviews

# IMDb aggregate ratings + metadata
ds = load_imdb("title.basics.tsv.gz", "title.ratings.tsv.gz", min_votes=1000)

# MovieLens per-user ratings (enables collaborative filtering)
ds = load_movielens("ml-latest-small/")

# Attach TMDB review text as the "comments" signal
ds.reviews = load_tmdb_reviews("tmdb_reviews.csv")   # cols: movie_id, text[, likes]

eng = RecommendationEngine(ds).fit()
```

The internal schema is three DataFrames — `movies` (`movie_id, title, year,
genres, avg_rating, num_votes, overview, runtime`), `ratings` (`user_id,
movie_id, rating`), and `reviews` (`movie_id, user_id, text, likes`). Any
adapter that produces these tables works. See `data/` for the exact CSV format
(`python generate_sample_data.py` writes it).

## Graceful degradation

The engine adapts to whatever signals you actually have:

- **No per-user ratings** → collaborative filtering is skipped; content +
  popularity (+ sentiment) still work. This is the IMDb-datasets-only case.
- **No reviews** → sentiment and likes drop out of the hybrid automatically.
- **No overviews** (as in raw IMDb/MovieLens) → content similarity leans on
  genres + year + runtime.

## Evaluation

`evaluate` does a per-user holdout and reports rating **RMSE**, **precision@k**,
and **recall@k**:

```
mf    RMSE=1.02   precision@10=0.022  recall@10=0.129
svd   RMSE=1.21   precision@10=0.010  recall@10=0.058
knn   RMSE=1.21   precision@10=0.006  recall@10=0.039
```

Matrix factorization clearly beats plain TruncatedSVD (which imputes unrated
cells as zero and collapses toward the mean) and item-item KNN. **These numbers
are on the synthetic dataset and are illustrative** — the synthetic per-user
signal is deliberately noisy, and the catalogue is tiny, so absolute
precision is modest. On real **MovieLens-100k**, the same matrix-factorization
model reaches substantially higher precision/recall. The point the numbers
demonstrate is the *relative* ordering and that CF genuinely learns per-user
taste (recall ≈3.5× a random baseline here).

## Project structure

```
movie_recommender/
├── recommender/
│   ├── data.py        # schema, IMDb/MovieLens/TMDB adapters, synthetic generator
│   ├── sentiment.py   # dependency-free lexicon sentiment for reviews/comments
│   ├── engines.py     # weighted rating, content, CF (MF + KNN + SVD), hybrid
│   ├── evaluate.py    # holdout precision@k / recall@k / RMSE
│   ├── cli.py         # command-line interface
│   └── demo.py        # narrated end-to-end demo
├── demo.py                  # -> recommender.demo.run()
├── generate_sample_data.py  # writes the synthetic set to data/*.csv
├── data/                    # example CSVs (movies, ratings, reviews)
└── requirements.txt
```

## Notes & extension points

- **Sentiment**: the built-in scorer is a compact lexicon+rules model (zero
  dependencies). For production, swap `recommender/sentiment.py:score_text` for
  VADER or a transformer — the rest of the engine only needs a float in
  `[-1, 1]`.
- **Scale**: content similarity is per-query (no N×N materialization). For very
  large catalogues, back it with an approximate-nearest-neighbour index
  (e.g. FAISS) using the same feature vectors.
- **Cold start**: `recommend(liked=[...])` needs no user history — it seeds
  from a few titles the person likes.
