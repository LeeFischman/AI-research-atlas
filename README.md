# AI Research Atlas

A daily-updated interactive semantic atlas of recent AI research papers from arXiv (cs.AI), visualised using [Apple Embedding Atlas](https://apple.github.io/embedding-atlas/). Each point is a paper. Nearby points share similar research topics. Clusters are labelled automatically using KeyBERT. Points are coloured by reputation tier or author count.

---

## How It Works

### 1. Paper Collection
Each run fetches up to 250 recent `cs.AI` submissions from the [arXiv API](https://arxiv.org/help/api/index) covering the last 2 days (5 days on first run). Papers are stored in a rolling 4-day window in `database.parquet`, committed back to the repository after each run. Duplicate arXiv IDs are overwritten with the newest version.

### 2. Embeddings — SPECTER2
Each paper is represented as a 768-dimensional vector using [`allenai/specter2_base`](https://huggingface.co/allenai/specter2_base), a transformer model purpose-built for scientific text. Unlike general-purpose sentence encoders, SPECTER2 was trained on citation graphs — papers that cite each other are pulled closer in the embedding space, so semantic proximity in the atlas reflects genuine intellectual relationships rather than just surface word overlap.

In **incremental mode** (the default), only newly fetched papers are embedded. Existing papers reuse their stored vectors, saving significant runner time. Both the full 768D embedding and the 50D intermediate projection are stored in `database.parquet` alongside the paper metadata.

### 3. Dimensionality Reduction — Two-Stage UMAP
UMAP runs twice per build using two different objectives:

**Stage 1 — 768D → 50D (for clustering)**
The full embedding vectors are reduced to 50 dimensions using cosine metric. This intermediate representation preserves far more semantic structure than a direct 2D projection, giving HDBSCAN a richer space to find meaningful clusters. The 50D vectors are stored in `database.parquet` and reused on subsequent runs.

**Stage 2 — 768D → 2D (for display)**
A separate UMAP pass produces the 2D coordinates used to position points on screen. This projection prioritises visual separation and layout quality rather than clustering fidelity.

| Parameter | Value | Effect |
|-----------|-------|--------|
| `n_neighbors` | 15 | Balances local vs global structure |
| `min_dist` | 0.1 | Controls how tightly points cluster visually (2D only) |
| `metric` | cosine | Appropriate for high-dimensional text vectors |
| `random_state` | 42 | Reproducible layout |

### 4. Cluster Labelling — HDBSCAN + KeyBERT

**HDBSCAN** clusters papers in the 50D cosine space. Unlike k-means, HDBSCAN requires no pre-specified number of clusters and handles clusters of varying density naturally. Clustering in 50D rather than 2D means the groupings reflect genuine semantic similarity rather than visual proximity alone. Points that don't belong to any cluster are marked as noise and receive no label.

**KeyBERT** then extracts the most representative keyword phrase for each cluster. It encodes candidate phrases from the cluster's paper titles using SPECTER2 and finds the phrase whose embedding is closest to the cluster centroid. Labels reflect semantic content rather than raw word frequency — "federated learning" or "medical imaging" wins over generic terms that happen to appear often across all clusters.

Paper titles (not abstracts) are used as KeyBERT input because titles are noun-dense and capture each paper's specific contribution concisely. Generic AI boilerplate ("model", "approach", "results") is filtered out via an extended stop word list before extraction.

### 5. Reputation Scoring
Each paper is scored on three signals and assigned to one of two tiers:

| Signal | Points |
|--------|--------|
| Institution name found in title or abstract | +3 |
| Public code on GitHub or HuggingFace | +2 |
| 8+ authors (large consortium / industrial lab) | +3 |
| 4–7 authors (mid-sized collaboration) | +1 |

Papers scoring ≥ 3 are labelled **Reputation Enhanced**. All others are **Reputation Std**. Select "Reputation" in the color picker to see this overlay.

### 6. Visualisation — Apple Embedding Atlas
The final site is built using [Apple Embedding Atlas](https://apple.github.io/embedding-atlas/), an open-source tool for interactive embedding visualisation. It renders up to several million points smoothly using a WebGL stack backed by [DuckDB-WASM](https://duckdb.org/docs/api/wasm/overview) for in-browser queries. The pre-computed 2D coordinates and KeyBERT labels are passed in directly; Embedding Atlas handles rendering, zoom, pan, search, and the data table.

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

The first run takes 30–45 minutes (SPECTER2 model download is cached after that, and subsequent incremental runs are significantly faster). It will:
- Fetch the last 5 days of cs.AI papers from arXiv
- Embed all papers with SPECTER2 (768D)
- Project to 50D with UMAP for clustering
- Project to 2D with UMAP for display
- Cluster in 50D cosine space with HDBSCAN
- Generate cluster labels with KeyBERT
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

Controlled by the `EMBEDDING_MODE` environment variable, defaulting to `incremental`:

| Mode | How it works | When to use |
|------|-------------|-------------|
| `incremental` *(default)* | Embeds only new papers; both UMAP stages, HDBSCAN, and KeyBERT run over the full corpus | Daily runs — faster, KeyBERT labels active |
| `full` | CLI handles SPECTER2 + UMAP internally on every run | Full reset — slower, TF-IDF labels (weaker) |

---

## Tuning

### Cluster granularity
All HDBSCAN settings take effect immediately on the next build and do not affect `database.parquet`. See the detailed settings guide in `generate_keybert_labels()` in `update_map.py`. Key parameters:

```python
clusterer = HDBSCAN(min_cluster_size=5, min_samples=4, metric=cluster_metric)
```

| Parameter | Effect |
|-----------|--------|
| `min_cluster_size` | Lower → more clusters; higher → fewer, broader clusters |
| `min_samples` | Lower → fewer noise points; higher → stricter assignment |
| `cluster_selection_method="leaf"` | Finer-grained clusters; try if results feel too coarse |
| `min_cluster_size=max(5, len(df) // 40)` | Adaptive sizing that scales with corpus size |
| `cluster_selection_epsilon=0.5` | Merges clusters closer than this threshold |

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

---

## Known Limitations

### Institution Detection
The arXiv API does not return structured affiliation data — there is no institution field in the API response. Institution matching in the reputation scoring works by searching the **title and abstract text** for institution names appearing inline (e.g. an abstract mentioning "...conducted at MIT..." or a title referencing "DeepMind").

This means reputation scoring will miss papers where the authors are from top institutions but never mention them in the title or abstract, which is common — affiliations typically appear only in the author byline on the PDF, which the API does not expose.

The current scoring compensates with two proxy signals that don't require affiliation data: GitHub/HuggingFace links (indicating a public codebase) and author count (larger teams correlate loosely with institutional backing). The result is a reasonable approximation, but not a precise institutional ranking.

If more accurate affiliation data is needed, options include:
- **OpenAlex API** — returns structured per-author affiliation data parsed from submission metadata, cross-referenceable by arXiv ID. Free, well-documented, and indexes arXiv papers within 1–2 days of submission.
- **Semantic Scholar API** — similar structured affiliation data, also free but with slightly slower arXiv coverage.
- **arXiv bulk metadata (OAI-PMH)** — includes affiliations for some papers but inconsistently populated and more complex to integrate.

### Citation Counts
Any citation-based quality signal (from OpenAlex, Semantic Scholar, etc.) is effectively zero for papers less than a week old. The atlas focuses on very recent preprints, so citation counts are not a useful signal for this corpus. Author-level citation history (a proxy for researcher seniority) is the more practical alternative.
