"""Post the daily Atlas snapshot to LinkedIn as the authenticated member.

Flow: resolve the member URN from the access token (no second secret needed)
-> register an image upload -> PUT the PNG bytes -> create the post with the
fixed text. Runs as a continue-on-error step in the daily workflow, after
atlas_snapshot.py renders and before the commit step.

Guards (all exit 0, never fail the pipeline):
  - LINKEDIN_ACCESS_TOKEN not set         -> setup pending, skip
  - atlas_snapshot.png missing            -> nothing to post, skip
  - not the posting weekday (LINKEDIN_POST_DAY, New York time) -> skip
  - linkedin_last_post.txt == this ISO week -> already posted (e.g. a
                                             workflow re-run), skip
Real API failures exit 1: the workflow step is continue-on-error, so the
pipeline still completes, but the step shows red in the run summary.

Env:
  LINKEDIN_ACCESS_TOKEN  member token with scopes: openid profile w_member_social
                         (expires every ~60 days -- regenerate in the LinkedIn
                         Developer Portal and update the GitHub secret)
  LINKEDIN_VERSION       optional LinkedIn-Version header (YYYYMM) to try
                         first. Not needed: LinkedIn retires versions after
                         ~12 months, so the script probes from the current
                         month backwards and uses the newest active one.
  DRY_RUN                "true" -> validate the token and print what would be
                         posted, but post nothing.
"""

import datetime
import os
import sys

import requests

API = "https://api.linkedin.com"
TITLE = "AI Research Atlas Weekly Report"
SITE_URL = "https://airesearchatlas.com"
HASHTAGS = ["AI", "MachineLearning", "arXiv"]
DATABASE = "database.parquet"
TAXONOMY = "dynamic_taxonomy.json"
# "New on the map": dynamic groups created in the last NEW_GROUP_DAYS days,
# still active, with at least NEW_GROUP_MIN_PAPERS papers. Days on which more
# than BULK_CREATION_LIMIT groups were created are treated as a taxonomy
# rebuild (not genuine emergence) and skipped. At most NEW_GROUP_MAX shown.
NEW_GROUP_DAYS = 7
NEW_GROUP_MIN_PAPERS = 5
NEW_GROUP_MAX = 2
BULK_CREATION_LIMIT = 3
ALT_TEXT = "Semantic map of today's arXiv cs.AI papers - airesearchatlas.com"
IMAGE = "atlas_snapshot.png"
STAMP_FILE = "linkedin_last_post.txt"   # holds the ISO week last posted, e.g. 2026-W46
# Post once a week, on the run that falls on this weekday in New York time.
# The nightly run (03:00 UTC) lands at 10-11pm ET the previous evening, so the
# Monday post is made by the run that starts Tuesday 03:00 UTC. Override with
# LINKEDIN_POST_DAY (Mon..Sun), or set it to "daily" to post every run.
POST_DAY = os.environ.get("LINKEDIN_POST_DAY", "Mon").strip()
POST_TZ = "America/New_York"


# LinkedIn "little text" reserved characters must be backslash-escaped in
# the post commentary, or the post can be rejected or silently truncated.
_RESERVED = set("\\|{}@[]()<>#*_~")


def escape_little_text(text):
    return "".join("\\" + c if c in _RESERVED else c for c in text)


def map_stats():
    """(paper_count, group_count) from database.parquet, or None."""
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(DATABASE, columns=["group_id_v3"])
        col = table.column("group_id_v3").to_pylist()
        groups = {g for g in col if g is not None}
        return table.num_rows, len(groups)
    except Exception as e:
        print(f"Post text: map stats unavailable ({type(e).__name__}: {e})")
        return None


def new_groups(today):
    """Names of genuinely new research directions, largest first."""
    try:
        import json
        from collections import Counter
        with open(TAXONOMY) as fh:
            groups = json.load(fh).get("groups", {})
        per_day = Counter(g.get("created") for g in groups.values())
        cutoff = today - datetime.timedelta(days=NEW_GROUP_DAYS)
        active_since = today - datetime.timedelta(days=1)
        picks = []
        for g in groups.values():
            created = datetime.date.fromisoformat(g["created"])
            last_seen = datetime.date.fromisoformat(g["last_seen"])
            papers = (g.get("paper_count_history") or [0])[-1]
            if (created > cutoff and last_seen >= active_since
                    and papers >= NEW_GROUP_MIN_PAPERS
                    and per_day[g["created"]] <= BULK_CREATION_LIMIT):
                picks.append((papers, g["name"]))
        picks.sort(reverse=True)
        return [name for _, name in picks[:NEW_GROUP_MAX]]
    except Exception as e:
        print(f"Post text: new-group detection unavailable ({type(e).__name__}: {e})")
        return []


def build_post_text(today):
    """Daily post text. Every part beyond the title and link is optional and
    is simply left out if its data can't be read."""
    stats = map_stats()
    if stats:
        papers, groups = stats
        first = f"{TITLE}: {papers:,} papers across {groups} research directions"
    else:
        first = TITLE
    lines = [escape_little_text(first)]
    fresh = new_groups(today)
    if fresh:
        lines.append(escape_little_text("New on the map: " + "; ".join(fresh)))
    lines.append(escape_little_text(f"Explore further: {SITE_URL}"))
    lines.append(" ".join("{hashtag|\\#|" + t + "}" for t in HASHTAGS))
    return "\n".join(lines)


def candidate_versions():
    """Newest-first YYYYMM strings: optional override, then this month back 12."""
    out = []
    override = os.environ.get("LINKEDIN_VERSION", "").strip()
    if override:
        out.append(override)
    d = datetime.date.today().replace(day=1)
    for _ in range(13):
        v = d.strftime("%Y%m")
        if v not in out:
            out.append(v)
        d = (d - datetime.timedelta(days=1)).replace(day=1)
    return out


def versioned_headers(token, version):
    return {
        "Authorization": f"Bearer {token}",
        "X-Restli-Protocol-Version": "2.0.0",
        "LinkedIn-Version": version,
    }


def is_version_error(r):
    return r.status_code == 426 or "NONEXISTENT_VERSION" in r.text


def main():
    token = os.environ.get("LINKEDIN_ACCESS_TOKEN", "").strip()
    if not token:
        print("LINKEDIN SKIP: LINKEDIN_ACCESS_TOKEN not set (setup pending).")
        return 0
    if not os.path.exists(IMAGE):
        print(f"LINKEDIN SKIP: {IMAGE} not found (snapshot step failed?).")
        return 0

    from zoneinfo import ZoneInfo
    local_now = datetime.datetime.now(ZoneInfo(POST_TZ))
    daily = POST_DAY.lower() == "daily"
    # Grace window: a scheduled run delayed past midnight still counts for the
    # posting day if it starts before 06:00 the next morning.
    effective = local_now - datetime.timedelta(hours=6) if local_now.hour < 6 else local_now
    if not daily and effective.strftime("%a").lower() != POST_DAY[:3].lower():
        print(f"LINKEDIN SKIP: posts go out weekly on {POST_DAY} ({POST_TZ}); "
              f"today is {local_now.strftime('%a %Y-%m-%d')}.")
        return 0

    # One post per period: per ISO week when weekly, per day when daily.
    iso = local_now.isocalendar()
    period = (local_now.strftime("%Y-%m-%d") if daily
              else f"{iso[0]}-W{iso[1]:02d}")
    if os.path.exists(STAMP_FILE):
        with open(STAMP_FILE) as fh:
            if fh.read().strip() == period:
                print(f"LINKEDIN SKIP: already posted for {period}.")
                return 0

    # 1. Resolve the member URN from the token (needs openid + profile scopes).
    r = requests.get(f"{API}/v2/userinfo",
                     headers={"Authorization": f"Bearer {token}"}, timeout=30)
    if r.status_code != 200:
        print(f"LINKEDIN FAIL: userinfo returned {r.status_code}: {r.text[:300]}")
        print("Most likely cause: token expired (60-day lifetime). "
              "Regenerate it in the Developer Portal and update the GitHub secret.")
        return 1
    author = f"urn:li:person:{r.json()['sub']}"
    print(f"Posting as: {author}")

    post_text = build_post_text(
        datetime.datetime.now(datetime.timezone.utc).date())
    print("Post text:\n" + post_text)

    if os.environ.get("DRY_RUN", "").lower() == "true":
        print(f"DRY_RUN: would post the text above with {IMAGE} "
              f"({os.path.getsize(IMAGE)} bytes). Posting nothing.")
        return 0

    # 2. Register the image upload, probing for the newest active API version
    #    (brand-new months are often not live yet; old ones get retired).
    version = None
    for v in candidate_versions():
        r = requests.post(f"{API}/rest/images?action=initializeUpload",
                          headers={**versioned_headers(token, v),
                                   "Content-Type": "application/json"},
                          json={"initializeUploadRequest": {"owner": author}},
                          timeout=30)
        if is_version_error(r):
            print(f"API version {v} not active; trying older...")
            continue
        version = v
        break
    if version is None:
        print("LINKEDIN FAIL: no active LinkedIn API version found in the last "
              "13 months. Set LINKEDIN_VERSION (YYYYMM) in the workflow env.")
        return 1
    if r.status_code not in (200, 201):
        print(f"LINKEDIN FAIL: initializeUpload {r.status_code} "
              f"(version {version}): {r.text[:300]}")
        return 1
    print(f"Using LinkedIn API version {version}")
    value = r.json()["value"]
    upload_url, image_urn = value["uploadUrl"], value["image"]

    # 3. Upload the PNG bytes.
    with open(IMAGE, "rb") as fh:
        r = requests.put(upload_url, data=fh,
                         headers={"Authorization": f"Bearer {token}",
                                  "Content-Type": "application/octet-stream"},
                         timeout=120)
    if r.status_code not in (200, 201):
        print(f"LINKEDIN FAIL: image upload {r.status_code}: {r.text[:300]}")
        return 1
    print(f"Image uploaded: {image_urn}")

    # 4. Create the post.
    payload = {
        "author": author,
        "commentary": post_text,
        "visibility": "PUBLIC",
        "distribution": {
            "feedDistribution": "MAIN_FEED",
            "targetEntities": [],
            "thirdPartyDistributionChannels": [],
        },
        "content": {"media": {"id": image_urn, "altText": ALT_TEXT}},
        "lifecycleState": "PUBLISHED",
        "isReshareDisabledByAuthor": False,
    }
    r = requests.post(f"{API}/rest/posts",
                      headers={**versioned_headers(token, version),
                               "Content-Type": "application/json"},
                      json=payload, timeout=60)
    if r.status_code != 201:
        print(f"LINKEDIN FAIL: create post {r.status_code}: {r.text[:300]}")
        return 1

    post_id = r.headers.get("x-restli-id", "(id not returned)")
    print(f"LINKEDIN OK: post created {post_id}")
    with open(STAMP_FILE, "w") as fh:
        fh.write(period + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
