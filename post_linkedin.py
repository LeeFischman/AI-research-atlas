"""Post the daily Atlas snapshot to LinkedIn as the authenticated member.

Flow: resolve the member URN from the access token (no second secret needed)
-> register an image upload -> PUT the PNG bytes -> create the post with the
fixed text. Runs as a continue-on-error step in the daily workflow, after
atlas_snapshot.py renders and before the commit step.

Guards (all exit 0, never fail the pipeline):
  - LINKEDIN_ACCESS_TOKEN not set         -> setup pending, skip
  - atlas_snapshot.png missing            -> nothing to post, skip
  - linkedin_last_post.txt == today (UTC) -> already posted today (e.g. a
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
POST_TEXT = "AI Research Atlas Daily Report"
ALT_TEXT = "Semantic map of today's arXiv cs.AI papers - airesearchatlas.com"
IMAGE = "atlas_snapshot.png"
STAMP_FILE = "linkedin_last_post.txt"


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

    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    if os.path.exists(STAMP_FILE):
        with open(STAMP_FILE) as fh:
            if fh.read().strip() == today:
                print(f"LINKEDIN SKIP: already posted on {today}.")
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

    if os.environ.get("DRY_RUN", "").lower() == "true":
        print(f"DRY_RUN: would post '{POST_TEXT}' with {IMAGE} "
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
        "commentary": POST_TEXT,
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
        fh.write(today + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
