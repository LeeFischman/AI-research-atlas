# AI Research Atlas

A daily-updated interactive semantic atlas of recent AI research papers from arXiv (cs.AI), visualised using [Apple Embedding Atlas](https://apple.github.io/embedding-atlas/). Each point is a paper. Nearby points share similar research topics. Clusters are labelled automatically using KeyBERT. Points are coloured by reputation tier or author count.

---

## How It Works

### 1. Paper Collection
Each run fetches up to 250 recent `cs.AI` submissions from the [arXiv API](https://arxiv.org/help/api/index) covering the last 2 days (5 days on first run). Papers are stored in a rolling 4-day window in `database.parquet`, committed back to the repository after each run. Duplicate arXiv IDs are overwritten with the newest version.

### 2. Embeddings — SPECTER2
Each paper is represented as a 768-dimensional vector using [`allenai/specter2_base`](https://huggingface.co/allenai/specter2_base), a transformer model purpose-built for scientific text. Unlike general-purpose sentence encoders, SPECTER2 was trained on citation graphs — papers that cite each other are pulled closer in the embedding space, so semantic proximity in the atlas reflects genuine intellectual relationships rather than just surface word overlap.

In **incremental mode**, only newly fetched papers are embedded. Existing papers reuse their stored vectors, saving significant runner time. Embeddings are stored in `database.parquet` alongside the paper metadata.

### 3. Dimensionality Reduction — UMAP
The 768-dimensional vectors are projected to 2D using [UMAP](https://umap-learn.readthedocs.io/) (Uniform Manifold Approximation and Projection). UMAP preserves both local structure (nearby papers stay near each other) and global structure (clusters of related topics remain separated). Parameters used:

| Parameter | Value | Effect |
|-----------|-------|--------|
| `n_neighbors` | 15 | Balances local vs global structure |
| `min_dist` | 0.1 | Controls how tightly points cluster |
| `metric` | cosine | Appropriate for high-dimensional text vectors |
| `random_state` | 42 | Reproducible layout |

UMAP runs over the full corpus on every run so the layout is always globally coherent, even as new papers are added incrementally.

### 4. Cluster Labelling — HDBSCAN + KeyBERT
Cluster labels are generated in two steps:

**HDBSCAN** (Hierarchical Density-Based Spatial Clustering of Applications with Noise) identifies natural groupings in the 2D projection. Unlike k-means, HDBSCAN doesn't require a pre-specified number of clusters and handles clusters of varying density. Points that don't belong to any cluster are marked as noise and receive no label.

**KeyBERT** then extracts the most representative keyword phrase for each cluster. It works by encoding candidate phrases from the cluster's paper titles using SPECTER2 and finding the phrase whose embedding is closest to the cluster's centroid. This means labels reflect semantic content rather than raw word frequency — so "federated learning" or "medical imaging" beats generic terms that happen to appear often.

The title text is used for label extraction (not the full abstract) because titles are noun-dense and already capture the paper's specific contribution. Generic AI boilerplate ("model", "approach", "results") is filtered out via an extended stop word list before extraction.

### 5. Reputation Scoring
Each paper is scored on three signals and assigned to one of two tiers:

| Signal | Points |
|--------|--------|
| Affiliation with a top institution (MIT, Stanford, DeepMind, etc.) | +3 |
| Public code on GitHub or HuggingFace | +2 |
| 8+ authors (large consortium / industrial lab) | +3 |
| 4–7 authors (mid-sized collaboration) | +1 |

Papers scoring ≥ 3 are labelled **Reputation Enhanced**. All others are **Reputation Std**. Select "Reputation" in the color picker to see this overlay.

### 6. Visualisation — Apple Embedding Atlas
The final site is built using [Apple Embedding Atlas](https://apple.github.io/embedding-atlas/), an open-source tool for interactive embedding visualisation. It renders up to several million points smoothly using a modern WebGL stack backed by [DuckDB-WASM](https://duckdb.org/docs/api/wasm/overview) for in-browser SQL queries. The pre-computed 2D coordinates and KeyBERT labels are passed in directly; Embedding Atlas handles rendering, zoom, pan, search, and the data table.

---

## Setup (New Repo)

### 1. Create the repository
Create a new **public** GitHub repository. Clone it locally and copy in these files:

```
.github/workflows/daily_update.yml
update_map.py
requirements.txt
stop_words.csv
README.md
```

Commit and push to `main`.

### 2. Enable GitHub Actions write permissions
**Settings → Actions → General → Workflow permissions → Read and write permissions**. Save.

### 3. Run the workflow for the first time
**Actions → Update AI Research Atlas → Run workflow**

The first run takes 20–35 minutes (SPECTER2 model download is cached after that). It will:
- Fetch the last 5 days of cs.AI papers from arXiv
- Embed all papers with SPECTER2
- Project to 2D with UMAP
- Generate cluster labels with HDBSCAN + KeyBERT
- Build and export the Embedding Atlas site
- Commit `database.parquet` back to `main`
- Deploy the site to the `gh-pages` branch

### 4. Enable GitHub Pages
After the first run completes: **Settings → Pages → Source: Deploy from a branch → gh-pages / (root)**

Your atlas will be live at:
```
https://<your-username>.github.io/<repo-name>/
```

---

## Embedding Modes

Set `EMBEDDING_MODE` at the top of `update_map.py`:

| Mode | How it works | When to use |
|------|-------------|-------------|
| `incremental` *(default)* | Embeds only new papers; UMAP and KeyBERT run over the full corpus | Daily runs — faster, KeyBERT labels active |
| `full` | CLI handles SPECTER2 + UMAP internally on every run | Full reset — slower, TF-IDF labels (weaker) |

---

## Tuning

### Cluster granularity
In `generate_keybert_labels()`:
```python
clusterer = HDBSCAN(min_cluster_size=5, min_samples=4, metric="euclidean")
```
- Decrease `min_cluster_size` → more clusters, more labels
- Increase `min_cluster_size` → fewer, broader clusters

### Label style
```python
keywords = kw_model.extract_keywords(
    combined,
    keyphrase_ngram_range=(1, 2),  # (1,1) for single words, (1,2) for phrases
    use_mmr=True,
    diversity=0.3,                 # 0.0–1.0, higher = more diverse terms
    top_n=3,
)
```

### Retention window
Change `RETENTION_DAYS` (currently 4) to keep more or fewer days of papers.

### arXiv category
Change `cat:cs.AI` in the search query to any arXiv category, e.g. `cat:cs.LG` for machine learning or `cat:cs.CV` for computer vision.

### Reputation criteria
Edit `calculate_reputation()` and `INSTITUTION_PATTERN` to adjust which institutions and signals contribute to the reputation score.
