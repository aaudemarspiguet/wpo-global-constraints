# Datasets

Amazon's homepage logs are proprietary, and no public dataset has everything
the paper used (which rows were shown together on a page, to whom, under which
purchase option, and what got streamed). So this project uses three sources,
each for what it's good at.

| | 1. Simulator (built in) | 2. MovieLens + TMDB providers | 3. ContentWise Impressions |
|---|---|---|---|
| What's real | nothing (by design) | catalog, genres, user tastes, activity timing, stream/rent/buy availability | carousel rows actually shown to users of a streaming service, their screen position, and linked interactions |
| What's simulated | everything | page impressions and clicks | nothing, but you can't evaluate pages that were never shown |
| Ground truth for any page? | **yes** | yes (simulated clicks) | no |
| Use it for | the core replication (Tables 1–3, Figs. 2–3) | realism check: does the story survive real catalogs and tastes? | calibration: real category mix by row position, real engagement by category |
| Effort | none | ~15 minutes (+ free TMDB key) | ~1 hour (large files) |

Start with 1, finish the exercises, then add 2. Use 3 if you want your
simulator's parameters to come from real carousel behavior.

---

## 1. The built-in simulator

`src/wpo/world.py`. Four categories mirroring the paper (Prime movies, Prime
TV, rent/buy, add-on channels), 60 widgets with genre mixes, and a pool of
20,000 users with hidden tastes and category affinities. Many users never rent
or subscribe to channels; a minority love them. Weekend days over-sample
channel and TV fans. The true click model penalizes repeated categories, less
so for users who love that category.

Because every probability is known, you can compute the **true** value of any
page a policy composes. That's what lets you check claims the paper could only
estimate.

---

## 2. MovieLens + TMDB watch providers

### MovieLens (GroupLens, University of Minnesota)

- **Get it:** `python scripts/download_movielens.py` (default `ml-latest-small`),
  or browse https://grouplens.org/datasets/movielens/.
- **Size:** `ml-latest-small` has about 100,000 ratings and 3,600 tag
  applications on 9,000 movies by 600 users (last updated 9/2018).
- **Important:** GroupLens notes that the "latest" datasets change over time
  and are **not appropriate for reporting research results**. For anything you
  publish, use a fixed benchmark version (e.g., ml-1m, ml-20m, ml-25m).
- **Files used:** `movies.csv` (movieId, title, genres), `ratings.csv`
  (userId, movieId, rating, timestamp), `links.csv` (movieId, imdbId, tmdbId).
  `links.csv` is the bridge to TMDB.
- **License:** non-commercial use; cite Harper & Konstan (2015) [X6]. Read the
  README inside the zip.

### TMDB watch providers

- **What:** for each movie, which services carry it to stream ("flatrate"),
  rent, or buy, per country. The endpoint is
  `GET /3/movie/{tmdb_id}/watch/providers`. TMDB sources this data from
  JustWatch.
- **Get it:** create a free TMDB account, copy your API Read Access Token, then
  ```bash
  export TMDB_TOKEN=...
  python scripts/fetch_tmdb_providers.py --limit 3000
  ```
  Responses are cached in `data/tmdb_cache/`, so re-runs are free. Respect the
  rate limits in TMDB's current terms.
- **Attribution required** in anything you publish: "This product uses the
  TMDB API but is not endorsed or certified by TMDB."
- **Caveat:** availability is *today's*, not 2018's, and varies by region.

### How it maps onto the paper

`wpo/data/movielens.py` builds a `World`:

| Paper concept | Built from |
|---|---|
| Category (content type × purchase option) | `prime_movie` = on the house service by subscription; `rent_buy` = rentable or buyable; `channel` = subscription on another service (3P-like). Movies only, so m = 3. |
| Widgets | one per (category, genre) with ≥ 6 titles, plus "Popular …" per category; items are the most-rated titles |
| User taste | rating-weighted genre profile (mean-centered ratings) |
| Category affinity | how over-represented a category is among the titles the user rated, vs. the population. **This is an assumption:** rating a title isn't buying it. State it. |
| Weekend behavior | share of the user's ratings made on weekends |

Without `--providers`, categories are assigned pseudo-randomly and the run
prints a warning. That mode is only for checking that the pipeline runs.

---

## 3. ContentWise Impressions

- **Paper:** Pérez Maurera, Ferrari Dacrema, Saule, Scriminaci & Cremonesi.
  *ContentWise Impressions: An Industrial Dataset with Impressions Included.*
  CIKM 2020. https://doi.org/10.1145/3340531.3412774 (preprint: arXiv:2008.01212) [X7].
- **What:** implicit interactions and **impressions** (the recommendation rows
  actually shown) of movies and TV series from an over-the-top (OTT) media
  streaming service. The authors describe it as the first open-source dataset
  of its kind with impressions. Impressions are organized as rows on a screen,
  with the most relevant rows near the top and the most relevant items at the
  left of each row.
- **Size:** about 10.46M interactions, 307K impressions directly linked to an
  interaction, and 23.3M impressions with no linked interaction.
- **Get it:** https://github.com/ContentWise/contentwise-impressions. The README
  links an optional form that returns the download link. Unzip into
  `data/contentwise/`; you'll find `interactions`, `impressions-direct-link`,
  and `impressions-non-direct-link` inside its `data` folder.
- **License:** data released under **CC BY-NC-SA 4.0** (non-commercial,
  share-alike); the authors' code repo is AGPL-3.0. Cite the CIKM paper.
- **Resources:** the authors' split-generation script can use up to 16 GiB of
  RAM. Our loader subsamples (`--sample`) so it runs on a laptop.

### Schema (verify against the README in your download)

`wpo/data/contentwise.py` expects the columns below. They come from the
dataset's documentation; check them with `check_schema()` and edit the
constants at the top of the file if your copy differs.

| Table | Columns used |
|---|---|
| interactions | `user_id`, `item_id`, `series_id`, `item_type`, `interaction_type`, `recommendation_id` |
| impressions-direct-link | `recommendation_id`, `row_position`, `recommendation_list_length`, `recommended_series_list` |
| impressions-non-direct-link | `user_id`, `row_position`, `recommendation_list_length`, `recommended_series_list` |

### How it maps onto the paper

| Paper concept | ContentWise stand-in |
|---|---|
| Widget | one impressed recommendation row |
| Slot $i$ | `row_position` |
| Category | majority `item_type` of the series in the row (movies, TV series, episodes, …) |
| Target balance $\mathbf{b}$ (paper: "current production balance") | category mix of impressed rows by position (`calibration_report`) |
| Engagement signal | whether a row's impression is linked to an interaction (biased: you only see outcomes for what was shown) |

What it can't do: rows aren't grouped into full pages with timestamps in
every table, and you never observe pages that weren't shown. So use it to
**set** the simulator's category shares and targets, then run the experiments
in the simulator.

```bash
pip install pyarrow     # if the files are parquet
python scripts/contentwise_calibrate.py --data-dir data/contentwise --sample 2000000
```

For evaluating carousel interfaces offline, see also the follow-up work by the
same group on carousel-aware evaluation [X8].

---

## Optional: quick category labels without an API key

Kaggle hosts title listings per service, e.g., "Amazon Prime Movies and TV
Shows" (about 9.7K titles) and a merged "Movies on Netflix, Prime Video, Hulu
and Disney+" (about 16.7K movies, last updated May 2020). They're handy for a
no-key prototype of the category mapping, but they're stale snapshots; prefer
TMDB providers for anything you report.
