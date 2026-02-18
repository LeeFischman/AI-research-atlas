import arxiv
import pandas as pd
import subprocess
import os
import re
import json
import time
import shutil
import random
import urllib.error
from datetime import datetime, timedelta, timezone

DB_PATH = "database.parquet"
STOP_WORDS_PATH = "stop_words.csv"

# ──────────────────────────────────────────────
# 1. TEXT SCRUBBER
#    Strips "model" variants so they don't dominate
#    TF-IDF cluster labels. The full original abstract
#    is preserved separately for display purposes.
# ──────────────────────────────────────────────
def scrub_model_words(text):
    pattern = re.compile(r'\bmodel(?:s|ing|ed|er|ers)?\b', re.IGNORECASE)
    cleaned = pattern.sub("", text)
    return " ".join(cleaned.split())


# ──────────────────────────────────────────────
# 2. DOCS CLEANUP
# ──────────────────────────────────────────────
def clear_docs_contents(target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        return
    for filename in os.listdir(target_dir):
        file_path = os.path.join(target_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Skipped {file_path}: {e}")


# ──────────────────────────────────────────────
# 3. REPUTATION SCORING
#    Used only for point color — not for cluster labels.
# ──────────────────────────────────────────────
INSTITUTION_PATTERN = re.compile(r"\b(" + "|".join([
    "MIT", "Stanford", "CMU", "UC Berkeley", "Harvard",
    "DeepMind", "OpenAI", "Anthropic", "FAIR", "Meta AI"
]) + r")\b", re.IGNORECASE)

def calculate_reputation(row):
    score = 0
    full_text = f"{row['title']} {row['abstract']}".lower()
    if INSTITUTION_PATTERN.search(full_text):
        score += 3
    if any(k in full_text for k in ['github.com', 'huggingface.co']):
        score += 2
    return "Reputation Enhanced" if score >= 4 else "Reputation Std"


# ──────────────────────────────────────────────
# 3b. AUTHOR COUNT TIER
#    Buckets raw author count into a display label.
#    "8+ Authors" flags large consortia / industrial labs.
# ──────────────────────────────────────────────
def categorize_authors(n: int) -> str:
    if n <= 3:
        return "1–3 Authors"
    elif n <= 7:
        return "4–7 Authors"
    else:
        return "8+ Authors"


# ──────────────────────────────────────────────
# 4. ARXIV FETCH WITH EXPONENTIAL BACKOFF

BASE_WAIT    = 15    # seconds before first retry
MAX_WAIT     = 480   # cap per attempt (~8 min)
MAX_RETRIES  = 7

def fetch_results_with_retry(client, search):
    last_exception = None

    for attempt in range(MAX_RETRIES):
        try:
            return list(client.results(search))

        except Exception as e:
            last_exception = e
            err_str = str(e).lower()
            is_rate_limit = (
                "429" in err_str
                or "too many requests" in err_str
                or (isinstance(e, urllib.error.HTTPError) and e.code == 429)
            )

            # Exponential backoff with full jitter
            wait = min(BASE_WAIT * (2 ** attempt), MAX_WAIT)
            jitter = random.uniform(0, wait * 0.25)   # up to 25 % extra
            total_wait = wait + jitter

            if is_rate_limit:
                print(
                    f"⚠️  Rate limited (429) by arXiv — "
                    f"attempt {attempt + 1}/{MAX_RETRIES}. "
                    f"Waiting {total_wait:.0f}s before retry..."
                )
            else:
                print(
                    f"⚠️  arXiv error: {e} — "
                    f"attempt {attempt + 1}/{MAX_RETRIES}. "
                    f"Waiting {total_wait:.0f}s before retry..."
                )

            time.sleep(total_wait)

    raise Exception(
        f"arXiv fetch failed after {MAX_RETRIES} attempts. "
        f"Last error: {last_exception}"
    )


# ──────────────────────────────────────────────
# 5. HTML POP-OUT PANEL
#    Injected into <body> of the generated index.html.
#    A tab sits at left-center; clicking it slides open
#    an info panel with map description and legend.
# ──────────────────────────────────────────────
def build_panel_html(run_date: str) -> str:
    return f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
  :root {{
    --arm-font: 'Inter', system-ui, -apple-system, sans-serif;
    --arm-accent: #60a5fa;
    --arm-accent-dim: rgba(96, 165, 250, 0.12);
    --arm-border: rgba(255, 255, 255, 0.08);
    --arm-text: #e2e8f0;
    --arm-muted: #94a3b8;
    --arm-bg: rgba(15, 23, 42, 0.82);
    --arm-panel-w: 300px;
  }}

  /* ── Tab ── */
  #arm-tab {{
    position: fixed;
    left: 0;
    top: 50%;
    transform: translateY(-50%);
    z-index: 1000000;
    display: flex;
    align-items: center;
    justify-content: center;
    writing-mode: vertical-rl;
    text-orientation: mixed;
    background: rgba(15, 23, 42, 0.90);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    color: var(--arm-accent);
    border: 1px solid var(--arm-border);
    border-left: none;
    padding: 18px 9px;
    border-radius: 0 10px 10px 0;
    cursor: pointer;
    font-family: var(--arm-font);
    font-weight: 600;
    font-size: 12px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    box-shadow: 3px 0 20px rgba(0, 0, 0, 0.4);
    transition: background 0.2s, color 0.2s, box-shadow 0.2s;
    user-select: none;
  }}
  #arm-tab:hover {{
    background: rgba(30, 41, 59, 0.95);
    color: #93c5fd;
    box-shadow: 4px 0 24px rgba(96, 165, 250, 0.2);
  }}

  /* ── Panel ── */
  #arm-panel {{
    position: fixed;
    left: 0;
    top: 50%;
    transform: translateY(-50%) translateX(-110%);
    z-index: 999999;
    width: var(--arm-panel-w);
    max-height: 88vh;
    display: flex;
    flex-direction: column;
    background: var(--arm-bg);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid var(--arm-border);
    border-left: none;
    border-radius: 0 16px 16px 0;
    box-shadow: 6px 0 40px rgba(0, 0, 0, 0.6),
                inset 1px 0 0 rgba(255, 255, 255, 0.04);
    font-family: var(--arm-font);
    font-size: 13px;
    color: var(--arm-text);
    line-height: 1.65;
    transition: transform 0.32s cubic-bezier(0.4, 0, 0.2, 1);
    overflow: hidden;
  }}
  #arm-panel.arm-open {{
    transform: translateY(-50%) translateX(0);
  }}

  /* scrollable body */
  #arm-body {{
    overflow-y: auto;
    overflow-x: hidden;
    padding: 22px 20px 16px 20px;
    flex: 1;
    scrollbar-width: thin;
    scrollbar-color: #334155 transparent;
  }}
  #arm-body::-webkit-scrollbar {{ width: 4px; }}
  #arm-body::-webkit-scrollbar-track {{ background: transparent; }}
  #arm-body::-webkit-scrollbar-thumb {{ background: #334155; border-radius: 4px; }}

  /* ── Header ── */
  .arm-header {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 4px;
  }}
  .arm-title {{
    font-size: 15px;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.01em;
    line-height: 1.3;
    margin: 0;
  }}
  .arm-title span {{ color: var(--arm-accent); }}
  #arm-close {{
    background: none;
    border: none;
    color: var(--arm-muted);
    cursor: pointer;
    font-size: 17px;
    line-height: 1;
    padding: 2px 4px;
    border-radius: 4px;
    transition: color 0.15s, background 0.15s;
    flex-shrink: 0;
    margin-left: 8px;
  }}
  #arm-close:hover {{ color: #f1f5f9; background: rgba(255, 255, 255, 0.07); }}

  /* ── Byline ── */
  .arm-byline {{
    font-size: 12px;
    color: var(--arm-muted);
    margin-bottom: 16px;
  }}
  .arm-byline a {{
    color: var(--arm-accent);
    text-decoration: none;
    font-weight: 500;
    transition: color 0.15s;
  }}
  .arm-byline a:hover {{ color: #93c5fd; text-decoration: underline; }}

  /* ── Divider ── */
  .arm-divider {{
    border: none;
    border-top: 1px solid var(--arm-border);
    margin: 14px 0;
  }}

  /* ── Section label ── */
  .arm-section {{
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #475569;
    margin: 0 0 8px 0;
  }}

  /* ── Body text ── */
  .arm-p {{
    color: #94a3b8;
    margin: 0 0 10px 0;
    font-size: 12.5px;
  }}
  .arm-p a {{
    color: var(--arm-accent);
    text-decoration: none;
    transition: color 0.15s;
  }}
  .arm-p a:hover {{ color: #93c5fd; text-decoration: underline; }}

  /* ── Tip card ── */
  .arm-tip {{
    background: var(--arm-accent-dim);
    border: 1px solid rgba(96, 165, 250, 0.22);
    border-radius: 8px;
    padding: 10px 12px;
    font-size: 12px;
    color: #bfdbfe;
    margin-bottom: 12px;
    display: flex;
    gap: 8px;
    align-items: flex-start;
  }}
  .arm-tip-icon {{ flex-shrink: 0; font-size: 14px; margin-top: 1px; }}

  /* ── Legend ── */
  .arm-legend-row {{
    display: flex;
    align-items: flex-start;
    gap: 9px;
    margin-bottom: 8px;
    font-size: 12px;
    color: #94a3b8;
  }}
  .arm-dot {{
    width: 9px;
    height: 9px;
    border-radius: 50%;
    flex-shrink: 0;
    margin-top: 4px;
  }}
  .arm-dot-enhanced {{ background: #f59e0b; box-shadow: 0 0 6px rgba(245, 158, 11, 0.5); }}
  .arm-dot-std      {{ background: #6366f1; box-shadow: 0 0 6px rgba(99, 102, 241, 0.4); }}
  .arm-legend-label {{ font-weight: 600; color: #cbd5e1; }}

  /* ── Book card ── */
  .arm-book {{
    display: flex;
    align-items: center;
    gap: 10px;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid var(--arm-border);
    border-radius: 8px;
    padding: 10px 12px;
    text-decoration: none;
    transition: background 0.2s, border-color 0.2s;
    margin-bottom: 8px;
  }}
  .arm-book:hover {{
    background: rgba(96, 165, 250, 0.07);
    border-color: rgba(96, 165, 250, 0.3);
  }}
  .arm-book-icon  {{ font-size: 22px; flex-shrink: 0; }}
  .arm-book-text  {{ display: flex; flex-direction: column; }}
  .arm-book-title {{ font-size: 12px; font-weight: 600; color: #e2e8f0; line-height: 1.3; margin-bottom: 2px; }}
  .arm-book-sub   {{ font-size: 11px; color: var(--arm-accent); }}

  /* ── Status badge ── */
  #arm-footer {{
    padding: 10px 20px 14px 20px;
    border-top: 1px solid var(--arm-border);
    display: flex;
    align-items: center;
    gap: 7px;
    flex-shrink: 0;
  }}
  .arm-status-dot {{
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 6px rgba(34, 197, 94, 0.7);
    flex-shrink: 0;
    animation: arm-pulse 2.5s ease-in-out infinite;
  }}
  @keyframes arm-pulse {{
    0%, 100% {{ opacity: 1; }}
    50%       {{ opacity: 0.35; }}
  }}
  .arm-status-text {{ font-size: 11px; color: #475569; font-family: var(--arm-font); }}
  .arm-status-text strong {{ color: #64748b; font-weight: 500; }}
</style>

<!-- Left-center tab -->
<button id="arm-tab"
  onclick="document.getElementById('arm-panel').classList.toggle('arm-open')"
  aria-label="Open info panel">
  About
</button>

<!-- Slide-out panel -->
<div id="arm-panel" role="complementary" aria-label="About the Atlas">

  <div id="arm-body">

    <!-- Header -->
    <div class="arm-header">
      <p class="arm-title">The <span>AI Research</span> Atlas</p>
      <button id="arm-close"
        onclick="document.getElementById('arm-panel').classList.remove('arm-open')"
        aria-label="Close panel">✕</button>
    </div>

    <div class="arm-byline">
      By <a href="https://www.linkedin.com/in/lee-fischman/" target="_blank" rel="noopener">Lee Fischman</a>
    </div>

    <!-- About -->
    <p class="arm-section">About</p>
    <p class="arm-p">
      A live semantic atlas of recent AI research from arXiv (cs.AI) over the last 5 days, rebuilt daily.
      Each point is a paper. Nearby points share similar topics — clusters surface
      naturally from the embedding space and are labelled by their most distinctive terms.
    </p>
    <p class="arm-p">
      Powered by <a href="https://apple.github.io/embedding-atlas/" target="_blank" rel="noopener">Apple Embedding Atlas</a>
      and SPECTER2 scientific embeddings.
    </p>

    <hr class="arm-divider">

    <!-- Tip -->
    <div class="arm-tip">
      <span class="arm-tip-icon">💡</span>
      <span>Set color to <strong>Reputation</strong> to mark papers with higher reputation scoring.</span>
    </div>

    <!-- Legend -->
    <p class="arm-section">Reputation coloring</p>
    <div class="arm-legend-row">
      <div class="arm-dot arm-dot-enhanced"></div>
      <div>
        <span class="arm-legend-label">Reputation Enhanced</span><br>
        Papers from MIT, Stanford, CMU, DeepMind, OpenAI, Anthropic &amp; similar, or with public code on GitHub / HuggingFace.
      </div>
    </div>
    <div class="arm-legend-row">
      <div class="arm-dot arm-dot-std"></div>
      <div>
        <span class="arm-legend-label">Reputation Std</span><br>
        All other papers.
      </div>
    </div>

    <hr class="arm-divider">

    <!-- Books -->
    <p class="arm-section">Check out my books</p>
    <a class="arm-book" href="https://www.amazon.com/dp/B0GMVH6P2W" target="_blank" rel="noopener">
      <span class="arm-book-icon">📘</span>
      <span class="arm-book-text">
        <span class="arm-book-title">Building AI-Powered Products and Agents</span>
        <span class="arm-book-sub">Available on Amazon →</span>
      </span>
    </a>

    <hr class="arm-divider">

    <!-- How to use -->
    <p class="arm-section">How to use</p>
    <p class="arm-p">
      Click any point to read its abstract and open the PDF on arXiv.
      Use the search bar to find papers by keyword or phrase.
      Drag to pan; scroll or pinch to zoom.
    </p>

  </div><!-- /#arm-body -->

  <!-- Status badge -->
  <div id="arm-footer">
    <div class="arm-status-dot"></div>
    <span class="arm-status-text">Last updated <strong>{run_date} UTC</strong></span>
  </div>

</div><!-- /#arm-panel -->
"""


# ──────────────────────────────────────────────
# 6. MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    clear_docs_contents("docs")

    now = datetime.now(timezone.utc)
    run_date = now.strftime("%B %d, %Y")

    client = arxiv.Client(page_size=100, delay_seconds=10)
    search = arxiv.Search(
        query=(
            f"cat:cs.AI AND submittedDate:"
            f"[{(now - timedelta(days=5)).strftime('%Y%m%d%H%M')} TO {now.strftime('%Y%m%d%H%M')}]"
        ),
        max_results=250
    )

    results = fetch_results_with_retry(client, search)

    if not results:
        print("⚠️  No results returned from arXiv. Skipping build.")
        exit(0)

    data_list = []
    for r in results:
        title        = r.title
        abstract     = r.summary
        author_count = len(r.authors)
        # Repeat title twice to up-weight specific topic nouns vs. abstract boilerplate
        scrubbed = scrub_model_words(f"{title}. {title}. {abstract}")
        data_list.append({
            "title":        title,
            "abstract":     abstract,                        # preserved for display / tooltip
            "text":         scrubbed,                        # used for embeddings and TF-IDF labels
            "url":          r.pdf_url,
            "id":           r.entry_id.split("/")[-1],
            "author_count": author_count,                    # raw integer, visible in table
            "author_tier":  categorize_authors(author_count), # categorical, used for coloring
        })

    df = pd.DataFrame(data_list)

    # Reputation — point color only, never drives cluster labels
    df["group"] = df.apply(calculate_reputation, axis=1)

    df.to_parquet(DB_PATH, index=False)
    print(f"📄 Saved {len(df)} papers to {DB_PATH}")

    # ── Build the atlas ──────────────────────────────
    print("🧠 Building Atlas...")
    subprocess.run([
        "embedding-atlas", DB_PATH,
        "--text",       "text",
        "--model",      "allenai/specter2_base",
        "--stop-words", STOP_WORDS_PATH,
        "--export-application", "site.zip"
    ], check=True)

    os.system("unzip -o site.zip -d docs/ && touch docs/.nojekyll")

    # ── Config override ──────────────────────────────
    config_path = "docs/data/config.json"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            conf = json.load(f)

        # Display paper title in tooltip and table
        conf["name_column"]  = "title"
        conf["label_column"] = "title"

        # Color points by reputation (selectable in the UI)
        conf["color_by"] = "group"

        # Cluster floating labels are driven by TF-IDF on the 'text' column.
        # We intentionally do NOT set topic_label_column here — doing so would
        # override the auto-labels with our binary reputation strings.

        # Expose useful columns in the side panel / table
        conf.setdefault("column_mappings", {}).update({
            "title":        "title",
            "abstract":     "abstract",
            "group":        "group",
            "author_count": "author_count",
            "author_tier":  "author_tier",
            "url":          "url",
        })

        with open(config_path, "w") as f:
            json.dump(conf, f, indent=4)

        print("✅ Config updated: title labels, reputation coloring, columns mapped.")
    else:
        print("⚠️  docs/data/config.json not found — skipping config override.")

    # ── Inject pop-out panel into index.html ─────────
    index_file = "docs/index.html"
    if os.path.exists(index_file):
        panel_html = build_panel_html(run_date)
        with open(index_file, "r", encoding="utf-8") as f:
            content = f.read()
        # Inject just before </body> so the panel sits on top of everything
        if "</body>" in content:
            content = content.replace("</body>", panel_html + "\n</body>")
        else:
            content += panel_html
        with open(index_file, "w", encoding="utf-8") as f:
            f.write(content)
        print("✅ Info panel injected into index.html")
    else:
        print("⚠️  docs/index.html not found — skipping panel injection.")

    print("✨ Sync Complete!")
