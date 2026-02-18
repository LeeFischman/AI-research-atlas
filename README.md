# AI Research Map

A daily-updated interactive map of recent AI research papers from arXiv (cs.AI),
visualised using [Embedding Atlas](https://apple.github.io/embedding-atlas/).

Each point is a paper. Nearby points share similar research topics. Cluster labels
are generated automatically from the paper text. Points are coloured by reputation tier.

---

## Setup (new repo)

### 1. Create the repository

Create a new **public** GitHub repository. Clone it locally, then copy in these files:

```
.github/
  workflows/
    daily_update.yml
update_map.py
requirements.txt
stop_words.csv
README.md
```

Commit and push to `main`.

### 2. Enable GitHub Actions write permissions

Go to **Settings → Actions → General → Workflow permissions**
and select **Read and write permissions**. Save.

### 3. Run the workflow for the first time

Go to **Actions → Update AI Research Map → Run workflow**.
This will:
- Fetch the latest papers from arXiv
- Compute SPECTER2 embeddings and UMAP projection
- Build the Embedding Atlas site
- Commit `database.parquet` back to `main`
- Deploy the site to the `gh-pages` branch

The first run takes 15–25 minutes (model download is cached after that).

### 4. Enable GitHub Pages

After the first run completes, go to **Settings → Pages**.
- Source: **Deploy from a branch**
- Branch: **gh-pages** / `/ (root)`

Your map will be live at:
`https://<your-username>.github.io/<repo-name>/`

---

## How it works

| Component | Detail |
|-----------|--------|
| Papers | Last 5 days of `cs.AI` submissions, up to 250 per run |
| Embeddings | `allenai/specter2_base` — purpose-built for scientific text |
| Projection | UMAP to 2D (computed by Embedding Atlas internally) |
| Cluster labels | TF-IDF on paper text, with stop words to suppress generic AI terms |
| Point colour | **Reputation Enhanced** — papers from top institutions or with public code |
| Schedule | 14:00 UTC daily |

---

## Customisation

**Stop words** — Edit `stop_words.csv` (one word per row under the `word` header)
to suppress additional terms from cluster labels.

**Reputation criteria** — Edit `calculate_reputation()` in `update_map.py`
to add institutions or scoring rules.

**Date range** — Change `timedelta(days=5)` in `update_map.py` to widen or narrow
the arXiv query window.

**arXiv category** — Change `cat:cs.AI` in the search query to any arXiv category,
e.g. `cat:cs.LG` for machine learning or `cat:stat.ML`.
