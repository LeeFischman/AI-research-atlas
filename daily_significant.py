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
#   3. Retains only papers with Prominence >= Emerging (Elite / Enhanced /
#      Emerging). Unverified papers are excluded.
#   4. Sorts by composite score (tier_weight + ss_citation_count) descending.
#   5. Writes significant_papers.json for the LinkedIn extension.
#      The file is replaced on every run — it represents yesterday's harvest
#      only and is not cumulative.
#
# Relationship to weekly_significant.yml:
#   The weekly job maintains significant.parquet (citation-rank pool) for
#   Atlas display. This script is fully independent of that pool — it sources
#   directly from database.parquet so the LinkedIn feed always reflects
#   recent papers, not historically highly-cited ones.
#
# Normal run:
#   python daily_significant.py
# ──────────────────────────────────────────────────────────────────────────────

import json
import os
from datetime import datetime, timedelta, timezone

import pandas as pd

from atlas_utils import DB_PATH, load_ss_cache

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

SIGNIFICANT_PAPERS_JSON_PATH = "significant_papers.json"

# Tier weights for composite score (must match LinkedIn extension logic)
TIER_WEIGHTS = {
    "Elite":      100_000,
    "Enhanced":    50_000,
    "Emerging":    10_000,
    "Unverified":       0,
}

# Tiers that qualify as significant
SIGNIFICANT_TIERS = {"Elite", "Enhanced", "Emerging"}

# Minimum papers before expanding window to 2 days
SIG_JSON_MIN_PAPERS = 5


# ══════════════════════════════════════════════════════════════════════════════
# CORE
# ══════════════════════════════════════════════════════════════════════════════

def load_database() -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"{DB_PATH} not found — has the main pipeline run?")
    df = pd.read_parquet(DB_PATH)
    print(f"  Loaded {len(df)} papers from {DB_PATH}.")
    return df


def filter_by_date(df: pd.DataFrame, yesterday_str: str, two_days_str: str) -> tuple[pd.DataFrame, str]:
    """Filter df to yesterday's papers; expand to 2-day window if too few."""
    if "date_added" not in df.columns:
        raise ValueError("database.parquet is missing 'date_added' column.")

    dates = df["date_added"].astype(str).str[:10]

    mask = dates == yesterday_str
    if mask.sum() >= SIG_JSON_MIN_PAPERS:
        print(f"  {mask.sum()} paper(s) added on {yesterday_str}.")
        return df[mask].copy(), yesterday_str

    # Expand to 2-day window
    mask = dates.isin([yesterday_str, two_days_str])
    print(f"  Fewer than {SIG_JSON_MIN_PAPERS} papers yesterday — "
          f"expanding to 2-day window ({mask.sum()} papers).")
    return df[mask].copy(), f"{two_days_str}–{yesterday_str}"


def filter_significant(df: pd.DataFrame) -> pd.DataFrame:
    """Retain only papers with Prominence in SIGNIFICANT_TIERS."""
    if "Prominence" not in df.columns:
        print("  WARNING: 'Prominence' column missing — cannot filter by tier.")
        return df

    before = len(df)
    df = df[df["Prominence"].isin(SIGNIFICANT_TIERS)].copy()
    print(f"  {len(df)} significant paper(s) after Prominence filter "
          f"(dropped {before - len(df)} Unverified).")
    return df


def build_json(df: pd.DataFrame, ss_cache: dict) -> list[dict]:
    """Build the output records sorted by composite score descending."""
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    papers = []

    for _, row in df.iterrows():
        arxiv_id = str(row.get("id", "")).strip()
        tier     = str(row.get("Prominence", "Unverified"))

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
        composite  = TIER_WEIGHTS.get(tier, 0) + cite_count

        papers.append({
            "_composite_score":         composite,
            "id":                       arxiv_id,
            "title":                    str(row.get("title", "") or "").strip(),
            "url":                      str(row.get("url", "") or f"https://arxiv.org/pdf/{arxiv_id}"),
            "authors_list":             authors,
            "author_count":             int(row.get("author_count", len(authors)) or len(authors)),
            "ss_tldr":                  tldr,
            "abstract":                 str(row.get("abstract", "") or "").strip(),
            "ss_citation_count":        cite_count,
            "ss_influential_citations": inf_count,
            "prominence_tier":          tier,
            "publication_date":         str(row.get("publication_date", today_str) or today_str)[:10],
        })

    papers.sort(key=lambda p: p["_composite_score"], reverse=True)
    for p in papers:
        del p["_composite_score"]

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
    print(f"  Sourcing from: {DB_PATH}")
    print(f"  Target date  : {yesterday_str}")
    print(f"  Threshold    : Prominence >= Emerging")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    db = load_database()
    ss_cache = load_ss_cache()

    # ── Filter to yesterday's papers ──────────────────────────────────────────
    filtered, window_label = filter_by_date(db, yesterday_str, two_days_str)

    if filtered.empty:
        print(f"\n  No papers found for {window_label} — "
              f"{SIGNIFICANT_PAPERS_JSON_PATH} not written.")
        raise SystemExit(0)

    # ── Apply significance threshold ──────────────────────────────────────────
    significant = filter_significant(filtered)

    if significant.empty:
        print(f"\n  No significant papers (Prominence >= Emerging) found for "
              f"{window_label} — {SIGNIFICANT_PAPERS_JSON_PATH} not written.")
        raise SystemExit(0)

    # ── Build and write JSON ──────────────────────────────────────────────────
    papers = build_json(significant, ss_cache)

    with open(SIGNIFICANT_PAPERS_JSON_PATH, "w") as f:
        json.dump(papers, f, indent=2)

    print(f"\n  Wrote {len(papers)} paper(s) → {SIGNIFICANT_PAPERS_JSON_PATH}")
    if papers:
        top = papers[0]
        print(f"  Top paper : [{top['prominence_tier']}] {top['title'][:70]}")

    print("\n  Tier breakdown:")
    for tier in ["Elite", "Enhanced", "Emerging"]:
        count = sum(1 for p in papers if p["prominence_tier"] == tier)
        if count:
            print(f"    {tier}: {count}")

    print("\n✓  daily_significant.py complete.")
