"""Render a static PNG snapshot of the built Atlas (docs/) for LinkedIn posting.

Serves the freshly built docs/ folder on localhost and screenshots it with
headless Chromium via Playwright. The embedding-atlas pin (0.20.0) still has
the WebGL2 fallback, which renders in headless CI through SwiftShader.

Failure philosophy: this step must never take down the daily pipeline, and a
bad render must never overwrite the previous good snapshot. The screenshot is
written to a temp name and only promoted to atlas_snapshot.png if it passes a
minimum-size sanity check (blank/uniform renders compress far smaller than a
real scatter plot). Any exception logs and exits 0.

After a good render, the image is auto-cropped to the map content: UI chrome
(top bar, color picker, side tabs, status line) is ignored, the bounding box
of the dots + labels is found, padded, and widened/heightened to LinkedIn's
1.91:1 feed ratio on a white background. If cropping fails for any reason
the uncropped screenshot is kept.

Tunables via env:
  SNAPSHOT_SETTLE_SECONDS  extra wait after canvas appears (default 20)
  SNAPSHOT_MIN_BYTES       sanity threshold for a real render (default 80000)
  SNAPSHOT_CROP            "false" to disable the content crop (default on)
"""

import http.server
import os
import socketserver
import sys
import threading
import time

OUTPUT = "atlas_snapshot.png"
TEMP_OUTPUT = "atlas_snapshot_new.png"
DOCS_DIR = "docs"
PORT = 8123
VIEWPORT = {"width": 1200, "height": 627}  # LinkedIn feed aspect (1.91:1)
SCALE = 2                                  # -> 2400x1254 output
SETTLE_SECONDS = int(os.environ.get("SNAPSHOT_SETTLE_SECONDS", "20"))
MIN_BYTES = int(os.environ.get("SNAPSHOT_MIN_BYTES", "80000"))
CROP = os.environ.get("SNAPSHOT_CROP", "true").lower() != "false"
CROP_ASPECT = 1.91      # LinkedIn feed landscape ratio
CROP_PAD_FRAC = 0.04    # padding around content, as a fraction of its size
INK_THRESHOLD = 215     # grayscale value below which a pixel counts as content
# Fractions of the screenshot treated as the plot area (excludes UI chrome).
PLOT_LEFT, PLOT_TOP, PLOT_RIGHT, PLOT_BOTTOM = 0.04, 0.13, 0.995, 0.95


def crop_to_content(path):
    """Crop the screenshot in place to the map content at CROP_ASPECT.

    Returns the new (width, height), or None if nothing was changed.
    """
    from PIL import Image  # installed via sentence-transformers

    im = Image.open(path).convert("RGB")
    W, H = im.size
    x0, y0 = int(W * PLOT_LEFT), int(H * PLOT_TOP)
    x1, y1 = int(W * PLOT_RIGHT), int(H * PLOT_BOTTOM)

    ink = im.crop((x0, y0, x1, y1)).convert("L").point(
        lambda v: 255 if v < INK_THRESHOLD else 0)
    bbox = ink.getbbox()
    if not bbox:
        return None

    l, t, r, b = bbox[0] + x0, bbox[1] + y0, bbox[2] + x0, bbox[3] + y0
    pad = int(max(r - l, b - t) * CROP_PAD_FRAC)
    l, t, r, b = l - pad, t - pad, r + pad, b + pad
    cw, ch = r - l, b - t
    if cw / ch > CROP_ASPECT:
        nh = int(cw / CROP_ASPECT)
        t -= (nh - ch) // 2
        b = t + nh
    else:
        nw = int(ch * CROP_ASPECT)
        l -= (nw - cw) // 2
        r = l + nw

    # White canvas; paste only the in-bounds part of the plot area so any
    # expansion past it (or into UI chrome) stays clean white.
    canvas = Image.new("RGB", (r - l, b - t), (255, 255, 255))
    sl, st, sr, sb = max(l, x0), max(t, y0), min(r, x1), min(b, y1)
    canvas.paste(im.crop((sl, st, sr, sb)), (sl - l, st - t))
    canvas.save(path, optimize=True)
    return canvas.size


def serve_docs():
    """Serve docs/ on 127.0.0.1:PORT in a daemon thread; return the server."""
    def handler(*args, **kwargs):
        return http.server.SimpleHTTPRequestHandler(*args, directory=DOCS_DIR, **kwargs)

    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("127.0.0.1", PORT), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def main():
    if not os.path.isdir(DOCS_DIR):
        print(f"SNAPSHOT SKIP: {DOCS_DIR}/ not found (Atlas build did not run?)")
        return 0

    from playwright.sync_api import sync_playwright

    httpd = serve_docs()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(args=[
                "--use-angle=swiftshader",       # software WebGL in headless CI
                "--enable-unsafe-swiftshader",   # newer Chromium gates SW WebGL behind this
            ])
            page = browser.new_page(viewport=VIEWPORT, device_scale_factor=SCALE)
            page.goto(f"http://127.0.0.1:{PORT}/", wait_until="domcontentloaded",
                      timeout=120_000)
            page.wait_for_selector("canvas", timeout=60_000)
            print(f"Canvas found; settling {SETTLE_SECONDS}s for data load + render...")
            time.sleep(SETTLE_SECONDS)
            page.screenshot(path=TEMP_OUTPUT)
            browser.close()
    finally:
        httpd.shutdown()

    size = os.path.getsize(TEMP_OUTPUT) if os.path.exists(TEMP_OUTPUT) else 0
    if size < MIN_BYTES:
        print(f"SNAPSHOT SUSPECT: render is only {size} bytes (< {MIN_BYTES}) — "
              f"likely blank. Keeping previous {OUTPUT}; discarding new render.")
        if os.path.exists(TEMP_OUTPUT):
            os.remove(TEMP_OUTPUT)
        return 0

    dims = f"{VIEWPORT['width'] * SCALE}x{VIEWPORT['height'] * SCALE}"
    if CROP:
        try:
            new_size = crop_to_content(TEMP_OUTPUT)
            if new_size:
                dims = f"cropped to {new_size[0]}x{new_size[1]}"
            else:
                print("SNAPSHOT CROP: no content found; keeping full screenshot.")
        except Exception as e:
            print(f"SNAPSHOT CROP FAILED (non-fatal, keeping full screenshot): "
                  f"{type(e).__name__}: {e}")

    os.replace(TEMP_OUTPUT, OUTPUT)
    print(f"SNAPSHOT OK: {OUTPUT} ({os.path.getsize(OUTPUT)} bytes, {dims})")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"SNAPSHOT FAILED (non-fatal): {type(e).__name__}: {e}")
        sys.exit(0)
