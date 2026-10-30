"""Render a static PNG snapshot of the built Atlas (docs/) for LinkedIn posting.

Serves the freshly built docs/ folder on localhost and screenshots it with
headless Chromium via Playwright. The embedding-atlas pin (0.20.0) still has
the WebGL2 fallback, which renders in headless CI through SwiftShader.

Failure philosophy: this step must never take down the daily pipeline, and a
bad render must never overwrite the previous good snapshot. The screenshot is
written to a temp name and only promoted to atlas_snapshot.png if it passes a
minimum-size sanity check (blank/uniform renders compress far smaller than a
real scatter plot). Any exception logs and exits 0.

Tunables via env:
  SNAPSHOT_SETTLE_SECONDS  extra wait after canvas appears (default 20)
  SNAPSHOT_MIN_BYTES       sanity threshold for a real render (default 80000)
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

    os.replace(TEMP_OUTPUT, OUTPUT)
    print(f"SNAPSHOT OK: {OUTPUT} ({size} bytes, "
          f"{VIEWPORT['width'] * SCALE}x{VIEWPORT['height'] * SCALE})")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"SNAPSHOT FAILED (non-fatal): {type(e).__name__}: {e}")
        sys.exit(0)
