"""Tiny write-only HTTP server for piping canvas captures from the browser
to disk during ad-hoc debugging.

This exists because the MCP `preview_screenshot` tool (which calls CDP
Page.captureScreenshot) times out when the Claude Code preview window is
minimized or hidden. During live debugging we want the vision-capable
agent to see what's on screen; this server gives `preview_eval` somewhere
to POST a base64 JPEG produced by `canvas.toDataURL(...)`, bypassing CDP.

NOT used by the automated test harness. The Playwright-based
`just test-visual` suite uses `captureScene()` in
`frontend/tests/e2e/helpers.ts`, which captures atomically in-process
and needs no external collector.

Usage:
  python tools/dev/shot_server.py
  # then from a Claude Code preview_eval:
  #   fetch('http://127.0.0.1:9999/<name>', { method: 'POST', body: canvas.toDataURL('image/jpeg') })
  # ... and Read C:\\Users\\<user>\\AppData\\Local\\Temp\\optics-shots\\<name>.jpg

Default output dir: %TEMP%/optics-shots (Windows) or $TMPDIR/optics-shots.
Override via OPTICS_SHOT_DIR env var.
"""
import base64
import os
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer

OUT = os.environ.get(
    "OPTICS_SHOT_DIR",
    os.path.join(tempfile.gettempdir(), "optics-shots"),
)
os.makedirs(OUT, exist_ok=True)


class H(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8", "replace")
        # Strip "data:image/jpeg;base64," prefix if present.
        if body.startswith("data:"):
            body = body.split(",", 1)[1]
        # Sanitize the path component into a safe filename.
        name = self.path.strip("/").replace("..", "_").replace("\\", "_").replace("/", "_")
        if not name:
            name = "shot"
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            name += ".jpg"
        path = os.path.join(OUT, name)
        try:
            data = base64.b64decode(body)
        except Exception as e:
            self.send_response(400)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(f"decode failed: {e}".encode())
            return
        with open(path, "wb") as f:
            f.write(data)
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        msg = f"wrote {len(data)} bytes to {path}"
        self.wfile.write(msg.encode())
        # ASCII-safe stdout (Windows default cp1252 can't encode unicode arrows).
        sys.stdout.write(msg + "\n")
        sys.stdout.flush()

    def log_message(self, *a):
        # Silence the default access-log chatter; we print our own line per POST.
        pass


if __name__ == "__main__":
    banner = f"shot_server listening on 127.0.0.1:9999 -> {OUT}"
    sys.stdout.write(banner.encode("ascii", "replace").decode("ascii") + "\n")
    sys.stdout.flush()
    HTTPServer(("127.0.0.1", 9999), H).serve_forever()
