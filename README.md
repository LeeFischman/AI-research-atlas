# AI Research Atlas

A live, daily-updating semantic map of AI research. Papers from arXiv's `cs.AI` category are fetched, embedded, grouped by an AI taxonomist, and rendered as an interactive 2D scatter plot — making it easy to navigate the research landscape, spot emerging themes, and find related work.

**Live site:** https://leefischman.github.io/ai-research-atlas

---

## What it does

Every day at 14:00 UTC, a GitHub Actions workflow:

1. Fetches the last 2 days of `cs.AI` papers from arXiv (rolling 4-day window, ~200–250 papers)
2. Embeds each paper with SPECTER2 (768-dimensional citation-aware vectors)
3. Sends all papers to Claude Haiku in a single API call — Haiku assigns each paper to a thematic group and names it in one shot
4. Lays out the groups spatially using MDS on SPECTER2 inter-group distances, then scatters papers within each group using their SPECTER2 geometry as a direction hint
5. Renders the result as an interactive map using Apple's Embedding Atlas library and deploys to GitHub Pages

Each dot is a paper. Dots near each other share research methodology or framing. Group labels float over their clusters.

---

## Repository structure

```
update_map_v2.py        — v2 pipeline (current)
update_map.py           — v1 pipeline (kept as rollback)
atlas_utils.py          — shared utilities imported by both pipelines
database.parquet        — rolling paper database (committed, updated daily)
group_names_v2.json     — Haiku group name cache (committed, updated on full runs)
labels_v2.parquet       — label positions for Atlas CLI (committed, updated daily)
requirements.txt        — Python dependencies
docs/                   — built Atlas output, deployed to GitHub Pages
.github/workflows/
  update_atlas.yml      — daily workflow definition
```

---

## Pipeline in depth

### Stage 1 — Fetch

Papers are pulled from arXiv via the `arxiv` Python client, querying `cat:cs.AI` with a sliding date window. On the first run, 5 days of papers are fetched to pre-fill the database. Subsequent runs fetch the last 2 days and merge with the existing rolling database, pruning papers older than 4 days. Weekend dry spells (when arXiv returns nothing) rebuild from the existing database without fetching.

Each paper is stored with title, abstract, URL, author count, author tier, and a reputation score derived from institutional affiliation signals.

### Stage 2 — Embed

Papers are embedded with [SPECTER2](https://huggingface.co/allenai/specter2), a model trained on citation graphs. SPECTER2 places papers near each other in embedding space if researchers who cite one tend to cite the other — capturing methodological and intellectual proximity rather than surface keyword overlap.

Two projections are produced:
- **768D raw embeddings** — used for cosine distance computation throughout the pipeline
- **2D UMAP projection** (`projection_x/y`) — used only as direction hints for within-group scatter

Embeddings are cached in `database.parquet` and computed incrementally (only new papers are embedded on each run).

### Stage 3 — Haiku grouping

All papers are sent to Claude Haiku in a **single API call** (~22,000 tokens). The prompt asks Haiku to:
- Assign every paper to exactly one group (target: 12–18 groups)
- Name each group with a 3–6 word noun phrase capturing the shared intellectual thread
- Return a JSON array with one entry per paper

Haiku returns both the group assignment and the name in the same call — no second labeling pass needed.

**If Haiku returns more groups than `GROUP_COUNT_MAX`:** excess groups are merged down by repeatedly finding the closest pair (smallest mean inter-group SPECTER2 cosine distance) and absorbing the smaller group into the larger one. The larger group's name is kept.

**If Haiku fails:** the pipeline falls back to HDBSCAN clustering on the 50D SPECTER2 embeddings, with generic "Group N" names.

Group names are cached to `group_names_v2.json` after every successful Haiku call.

**Cost:** ~$0.04 per run (22K input tokens + ~4K output tokens at Haiku pricing).

### Stage 4 — Layout

**4a: MDS between-group layout**

A `(n_groups × n_groups)` distance matrix is built from mean pairwise SPECTER2 cosine distances between all papers in each group pair. Classical metric MDS projects this to 2D, placing groups that are semantically far apart further from each other on the canvas.

Raw MDS coordinates are multiplied by `LAYOUT_SCALE = 20.0` to produce a human-readable coordinate range (typically ±3 to ±5 units after scaling).

**4b: Within-group scatter**

Each paper is placed around its group's MDS centroid:

```
direction    = unit vector from SPECTER2 group centroid → paper's UMAP position
base_radius  = median_nearest_neighbour_centroid_distance × SCATTER_FRACTION
eff_radius   = base_radius × (1 + group_variance × VARIANCE_AMPLIFIER)
scatter_dist = eff_radius × (paper_mean_intra_dist / group_mean_intra_dist)
final_pos    = mds_centroid + direction × scatter_dist
```

The key design choice: scatter radius is expressed as a **fraction of the median nearest-neighbour centroid distance** (`SCATTER_FRACTION = 0.35`). This means each group's cloud extends at most 35% of the way toward its nearest neighbouring cluster, regardless of the absolute scale of the layout or how similar the groups are. This prevents clouds from overlapping even when groups are semantically close.

Papers at the edge of their group (high intra-group cosine distance) drift proportionally further from the centroid than papers at the core.

### Stage 5 — Build & deploy

The Atlas CLI is invoked with `projection_v2_x/y` as coordinates and `labels_v2.parquet` for group label positions. Label positions are computed as the **mean paper position** within each group (not the MDS centroid), so labels always sit at the visual centre of the dot cloud regardless of how scatter shifted the papers.

The built output is deployed to GitHub Pages via the `JamesIves/github-pages-deploy-action`.

---

## Design decisions and paths not taken

### Why Haiku for grouping instead of HDBSCAN?

The v1 pipeline used HDBSCAN on SPECTER2 embeddings. SPECTER2 groups papers by citation-graph proximity — papers cluster together if the same researchers tend to cite both. This produces semantically valid groups, but the clusters are opaque: a group might contain papers on seemingly unrelated surface topics because they share a methodological niche. The resulting cluster labels were hard to generate meaningfully because the groups didn't have obvious surface-level descriptions.

Haiku reads the titles and abstracts directly and groups by intellectual meaning. The groups it forms are immediately intuitive to a human reader, and it names them in the same pass — eliminating the need for a separate labeling call.

### Why not use Haiku's grouping and SPECTER2 geometry independently?

The spatial layout still uses SPECTER2 distances, not Haiku's grouping, for two reasons:

1. SPECTER2 captures continuous semantic distance. Haiku produces discrete assignments. The MDS layout preserves the relative distances between groups rather than treating all group boundaries as equally sharp.
2. Within-group scatter uses the SPECTER2 UMAP position as a direction hint, placing papers at the edge of a group (semantically distant from their group centroid) further from the label — a natural representation of within-group heterogeneity.

### Why MDS instead of UMAP for the group layout?

UMAP is excellent for large datasets (hundreds to thousands of points) but can distort global distance relationships in favour of preserving local structure. With only 12–18 group centroids, classical metric MDS is fast, stable, and preserves the global distance ratios faithfully. The MDS stress metric in the logs gives a direct measure of how well the 2D layout reflects the true distances.

### Why not just use the SPECTER2 UMAP directly as the final layout?

The SPECTER2 UMAP projection was tried as the final layout in v1. The problem is that UMAP's global structure is not reliable at the scale of the full corpus — groups that are semantically distant can end up adjacent due to UMAP's manifold assumptions. MDS on the group-level distance matrix is more geometrically honest about inter-group relationships.

### Label placement: mean paper position vs MDS centroid

Early builds placed labels at the MDS centroid. The scatter function pushes papers significantly away from the centroid (by design), so the label ended up floating in empty space between dot clouds. Switching to the mean of the actual `projection_v2_x/y` positions ensures labels always sit at the visual centre of their cluster, regardless of how scatter distributed the papers.

### Scatter radius: absolute vs fraction-of-centroid-spacing

The first scatter implementation used an absolute radius:
```
radius = mean_cosine_dist × SCATTER_SCALE_BASE × LAYOUT_SCALE
```
This worked until LAYOUT_SCALE changed or the number of groups changed, at which point clouds would either collapse to points or completely overlap. The current approach expresses radius as a fraction of the median nearest-neighbour centroid distance, making it self-calibrating. `SCATTER_FRACTION = 0.35` means clouds extend at most 35% toward the nearest neighbour, regardless of absolute scale.

---

## Configuration reference

All tuning knobs are at the top of `update_map_v2.py`:

| Parameter | Default | Effect |
|---|---|---|
| `GROUP_COUNT_MIN` | 12 | Minimum groups Haiku must produce; fewer triggers a retry |
| `GROUP_COUNT_MAX` | 18 | Maximum groups; excess are merged automatically |
| `ABSTRACT_GROUPING_CHARS` | 300 | Abstract chars sent to Haiku; shorter = cheaper, longer = better grouping |
| `GROUPING_MAX_RETRIES` | 5 | Haiku retry attempts (covers 529 overload + parse failures) |
| `GROUPING_RETRY_BASE_WAIT` | 60s | Base wait between retries, doubles each attempt |
| `LAYOUT_SCALE` | 20.0 | Multiplier on all MDS coordinates; increase if canvas feels cramped |
| `SCATTER_FRACTION` | 0.35 | Scatter radius as fraction of nearest centroid distance (0.25–0.50) |
| `VARIANCE_AMPLIFIER` | 2.0 | Extra spread for loosely-coupled groups (1.0–3.0) |

### Quick tuning guide

**Clusters overlap / labels still central** → lower `SCATTER_FRACTION` (0.35 → 0.25)

**Clusters too sparse / papers far from label** → raise `SCATTER_FRACTION` (0.35 → 0.45)

**Merge step absorbing groups that feel distinct** → raise `GROUP_COUNT_MAX` (18 → 20)

**Haiku consistently returns too few groups** → lower `GROUP_COUNT_MIN` or reduce `ABSTRACT_GROUPING_CHARS`

---

## Offline mode

Offline mode skips the arXiv fetch, SPECTER2 embedding, and Haiku API call entirely. It re-runs only the MDS layout, scatter, and Atlas build using data already in `database.parquet` and `group_names_v2.json`. Useful for testing layout parameter changes without incurring API costs or wait time.

**Requirements before using offline mode:**
- `database.parquet` must contain `group_id_v2`, `embedding`, and `projection_x/y` columns
- `group_names_v2.json` must exist (written automatically after every successful Haiku call and committed to the repo)

**Local:**
```bash
OFFLINE_MODE=true python update_map_v2.py
```

**In GitHub Actions workflow YAML:**
```yaml
- name: Run Update Script
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
    OFFLINE_MODE: "true"
  run: python update_map_v2.py
```

Remove `OFFLINE_MODE` from the YAML when you want a full production run (fresh fetch + re-grouping).

---

## Setup

### Prerequisites

- Python 3.12+
- GitHub repository with Pages enabled
- Anthropic API key (for Haiku grouping)
- Optionally: Hugging Face token (for higher rate limits when downloading SPECTER2)

### GitHub secrets required

| Secret | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Haiku grouping call |
| `HF_TOKEN` | (Optional) Hugging Face model downloads at higher rate limits |

### First run

On the first run, the pipeline pre-fills the database with the last 5 days of papers. This takes longer than normal daily runs due to the larger fetch and initial embedding of all papers.

---

## Rollback

`update_map.py` (v1 pipeline) is kept in the repo. To roll back:

1. Change `python update_map_v2.py` to `python update_map.py` in the workflow YAML
2. Push

The v1 pipeline reads from the same `database.parquet` but uses its own projection columns (`projection_x/y`) rather than the v2 columns (`projection_v2_x/y`), so the databases are fully compatible.
