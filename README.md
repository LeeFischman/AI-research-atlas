# AI Research Atlas

A daily-updated interactive atlas of recent AI research papers from arXiv (cs.AI),
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

Go to **Actions → Update AI Research Atlas → Run workflow**.
This will:
- Fetch the last 5 days of papers from arXiv (first-run pre-fill)
- Compute SPECTER2 embeddings and UMAP projection
- Build the Embedding Atlas site
- Commit `database.parquet` back to `main`
- Deploy the site to the `gh-pages` branch

The first run takes 15–25 minutes (model download is cached after that).

### 4. Enable GitHub Pages

After the first run completes, go to **Settings → Pages**.
- Source: **Deploy from a branch**
- Branch: **gh-pages** / `/ (root)`

Your atlas will be live at:
`https://<your-username>.github.io/<repo-name>/`

---

## Embedding modes

The workflow supports two modes, selectable from a dropdown when triggering a manual run
(**Actions → Update AI Research Atlas → Run workflow → embedding_mode**).
Scheduled daily runs always use the default (`full`).

| Mode | How it works | Best for |
|------|-------------|----------|
| **full** *(default)* | Hands the entire DB to the CLI; SPECTER2 and UMAP run internally on every run | Guaranteed layout coherence; simpler |
| **incremental** | Python embeds only *new* papers with SPECTER2; UMAP re-projects all stored vectors | Faster runner times as the DB grows |

In incremental mode, raw embedding vectors are stored in `database.parquet` alongside the
2D projection coordinates. UMAP still runs over the full corpus each time so the layout
remains globally coherent — only the expensive SPECTER2 step is skipped for existing papers.

---

## How it works

| Component | Detail |
|-----------|--------|
| Papers | Rolling 4-day window of `cs.AI` submissions, up to 250 per run |
| First run | Pre-fills with the last 5 days automatically |
| Deduplication | Duplicate arXiv IDs are overwritten with the newest version |
| Embeddings | `allenai/specter2_base` — purpose-built for scientific text |
| Projection | UMAP to 2D |
| Cluster labels | TF-IDF on paper text, with stop words to suppress generic AI terms |
| Point colour | **Reputation Enhanced** — papers from top institutions or with public code |
| Schedule | 14:00 UTC daily |

---

## Customisation

**Stop words** — Edit `stop_words.csv` (one word per row under the `word` header)
to suppress additional terms from cluster labels.

**Reputation criteria** — Edit `calculate_reputation()` in `update_map.py`
to add institutions or scoring rules.

**Retention window** — Change `RETENTION_DAYS` in `update_map.py` (currently 4 days).

**arXiv category** — Change `cat:cs.AI` in the search query to any arXiv category,
e.g. `cat:cs.LG` for machine learning or `cat:stat.ML`.
