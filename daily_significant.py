#!/usr/bin/env python3
# daily_significant.py
# ──────────────────────────────────────────────────────────────────────────────
# AI Research Atlas — Daily significant-papers feed writer.
#
# Runs every day after the main atlas build (daily_update.yml, 03:00 UTC).
#
# What this does:
#   1. Loads database.parquet (the live 14-day rolling window of all papers).
#   2. Filters to papers added yesterday (UTC) by date_added.
#      Falls back to a 2-day window if fewer than SIG_JSON_MIN_PAPERS qualify.
#   3. Scores every paper by composite score:
#        max_author_hindex * 1_000
#        + ss_influential_citations * 100
#        + ss_citation_count
#      No hard Prominence filter — ranking is purely signal-based so the feed
#      works from day one regardless of author-cache enrichment state.
#   4. Writes the top SIG_JSON_MAX_PAPERS by composite score to
#      significant_papers.json for the LinkedIn extension.
#      The file is replaced on every run — it represents yesterday's harvest
#      only and is not cumulative.
#
# Relationship to weekly_significant.yml:
#   The weekly job maintains significant.parquet (citation-rank pool) for
#   Atlas display. This script is fully independent — it sources directly
#   from database.parquet so the LinkedIn feed always reflects recent papers.
#
# Normal run:
#   python daily_significant.py
# ──────────────────────────────────────────────────────────────────────────────

import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from atlas_utils import DB_PATH, load_ss_cache

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

SIGNIFICANT_PAPERS_JSON_PATH = "significant_papers.json"

# Score weights
HINDEX_WEIGHT      = 1_000   # max author h-index — primary signal
INFLUENTIAL_WEIGHT = 100     # influential citation count — strong signal
CITATION_WEIGHT    = 1       # raw citation count — tiebreaker

# Minimum papers before expanding window to 2 days
SIG_JSON_MIN_PAPERS = 5

# Maximum papers written to the feed per day
SIG_JSON_MAX_PAPERS = 20


# ══════════════════════════════════════════════════════════════════════════════
# SCORING
# ══════════════════════════════════════════════════════════════════════════════

def max_hindex(val) -> int:
    """Extract the maximum h-index from an author_hindices array/list/None."""
    if val is None:
        return 0
    try:
        arr = np.asarray(val, dtype=float)
        arr = arr[~np.isnan(arr)]
        return int(arr.max()) if len(arr) > 0 else 0
    except Exception:
        return 0


def composite_score(row) -> float:
    h   = max_hindex(row.get("author_hindices"))
    inf = int(row.get("ss_influential_citations") or 0)
    cit = int(row.get("ss_citation_count") or 0)
    return h * HINDEX_WEIGHT + inf * INFLUENTIAL_WEIGHT + cit * CITATION_WEIGHT


# ══════════════════════════════════════════════════════════════════════════════
# CORE
# ══════════════════════════════════════════════════════════════════════════════

def load_database() -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"{DB_PATH} not found — has the main pipeline run?")
    df = pd.read_parquet(DB_PATH)
    print(f"  Loaded {len(df)} papers from {DB_PATH}.")
    return df


def filter_by_date(df: pd.DataFrame, today_str: str, yesterday_str: str, two_days_str: str) -> tuple[pd.DataFrame, str]:
    """Filter df to yesterday's papers; expand window progressively if too few.

    Order of preference:
      1. Yesterday only (normal steady-state case)
      2. Yesterday + today (catches manual runs or bootstrapping days)
      3. Yesterday + today + 2 days ago (wide fallback for weekends / gaps)
    """
    if "date_added" not in df.columns:
        raise ValueError("database.parquet is missing 'date_added' column.")

    dates = df["date_added"].astype(str).str[:10]

    # Try yesterday first
    mask = dates == yesterday_str
    if mask.sum() >= SIG_JSON_MIN_PAPERS:
        print(f"  {mask.sum()} paper(s) added on {yesterday_str}.")
        return df[mask].copy(), yesterday_str

    # Expand to include today (handles bootstrapping / manual runs)
    mask = dates.isin([yesterday_str, today_str])
    if mask.sum() >= SIG_JSON_MIN_PAPERS:
        print(f"  Fewer than {SIG_JSON_MIN_PAPERS} papers yesterday — "
              f"expanding to include today ({mask.sum()} papers).")
        return df[mask].copy(), f"{yesterday_str}–{today_str}"

    # Widen to 3-day window
    mask = dates.isin([two_days_str, yesterday_str, today_str])
    print(f"  Fewer than {SIG_JSON_MIN_PAPERS} papers in 2-day window — "
          f"expanding to 3 days ({mask.sum()} papers).")
    return df[mask].copy(), f"{two_days_str}–{today_str}"


def build_json(df: pd.DataFrame, ss_cache: dict) -> list[dict]:
    """Score, sort, cap, and build output records."""
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    papers = []

    for _, row in df.iterrows():
        arxiv_id = str(row.get("id", "")).strip()

        # Best available TLDR: live ss_cache first, then parquet column
        ss_entry = ss_cache.get(arxiv_id, {})
        tldr     = (ss_entry.get("tldr") or "").strip()
        if not tldr:
            tldr = str(row.get("ss_tldr", "") or "").strip()

        authors = row.get("authors_list", [])
        if not isinstance(authors, list):
            authors = []

        cite_count = int(row.get("ss_citation_count", 0) or 0)
        inf_count  = int(row.get("ss_influential_citations", 0) or 0)
        h_max      = max_hindex(row.get("author_hindices"))
        score      = h_max * HINDEX_WEIGHT + inf_count * INFLUENTIAL_WEIGHT + cite_count * CITATION_WEIGHT

        papers.append({
            "_score":                   score,
            "id":                       arxiv_id,
            "title":                    str(row.get("title", "") or "").strip(),
            "url":                      str(row.get("url", "") or f"https://arxiv.org/pdf/{arxiv_id}"),
            "authors_list":             authors,
            "author_count":             int(row.get("author_count", len(authors)) or len(authors)),
            "ss_tldr":                  tldr,
            "abstract":                 str(row.get("abstract", "") or "").strip(),
            "ss_citation_count":        cite_count,
            "ss_influential_citations": inf_count,
            "prominence_tier":          str(row.get("Prominence", "Unverified")),
            "publication_date":         str(row.get("publication_date", today_str) or today_str)[:10],
        })

    # Sort descending by composite score, cap at SIG_JSON_MAX_PAPERS
    papers.sort(key=lambda p: p["_score"], reverse=True)
    papers = papers[:SIG_JSON_MAX_PAPERS]

    # Strip internal field before writing
    for p in papers:
        del p["_score"]

    return papers


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    now           = datetime.now(timezone.utc)
    yesterday_str = (now.date() - timedelta(days=1)).strftime("%Y-%m-%d")
    two_days_str  = (now.date() - timedelta(days=2)).strftime("%Y-%m-%d")
    run_date      = now.strftime("%B %d, %Y")

    print("=" * 60)
    print("  AI Research Atlas — Daily Significant Papers Feed")
    print(f"  {run_date} UTC")
    print(f"  Sourcing from : {DB_PATH}")
    print(f"  Target date   : {yesterday_str}")
    print(f"  Scoring       : max_hindex × {HINDEX_WEIGHT:,}"
          f" + influential × {INFLUENTIAL_WEIGHT}"
          f" + citations × {CITATION_WEIGHT}")
    print(f"  Output cap    : top {SIG_JSON_MAX_PAPERS} papers")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────────────────────────
    db = load_database()
    ss_cache = load_ss_cache()

    # ── Filter to yesterday (with progressive fallback) ───────────────────────
    today_str_filter = now.strftime("%Y-%m-%d")
    filtered, window_label = filter_by_date(db, today_str_filter, yesterday_str, two_days_str)



    if filtered.empty:
        print(f"\n  No papers found for {window_label} — "
              f"{SIGNIFICANT_PAPERS_JSON_PATH} not written.")
        raise SystemExit(0)

    print(f"\n  {len(filtered)} paper(s) in window '{window_label}'.")

    # ── Score and build ───────────────────────────────────────────────────────
    papers = build_json(filtered, ss_cache)

    if not papers:
        print(f"  No papers to write — {SIGNIFICANT_PAPERS_JSON_PATH} not written.")
        raise SystemExit(0)

    # ── Write ─────────────────────────────────────────────────────────────────
    with open(SIGNIFICANT_PAPERS_JSON_PATH, "w") as f:
        json.dump(papers, f, indent=2)

    print(f"\n  Wrote {len(papers)} paper(s) → {SIGNIFICANT_PAPERS_JSON_PATH}")

    # ── Summary ───────────────────────────────────────────────────────────────
    top = papers[0]
    print(f"  Top paper     : {top['title'][:70]}")
    print(f"  Prominence    : {top['prominence_tier']}")
    print(f"  Citations     : {top['ss_citation_count']} "
          f"({top['ss_influential_citations']} influential)")

    tiers = {}
    for p in papers:
        tiers[p["prominence_tier"]] = tiers.get(p["prominence_tier"], 0) + 1
    print(f"  Tier breakdown: " + ", ".join(f"{t}: {n}" for t, n in sorted(tiers.items())))

    print("\n✓  daily_significant.py complete.")
