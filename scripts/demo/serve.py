"""Local demo: a web page to paste / drop / upload an image, crop the formula and read it.

    python scripts/demo/serve.py --ckpt v4=runs/final_v4_s1/checkpoints/best.pt [--ckpt hw=...] [--port 7860]

Standard library only (``http.server``), in the project environment, so the page shows exactly what
``python -m mathocr.predict`` returns: the canonical path (one image, fp32, deterministic kernels).  The page
(``index.html``) loads Cropper.js and KaTeX from cdnjs.  Runs on the CPU by default (``--device cuda`` to use the GPU).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "diagnostics"))
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from mathocr import determinism  # noqa: E402
from mathocr.infer import load_checkpoint, predict_images  # noqa: E402
from unimer_protocol import normalize_text  # noqa: E402

PAGE = Path(__file__).with_name("index.html")
MAX_BYTES = 20 << 20


class Models:
    def __init__(self, specs: list[str], device: torch.device):
        self.device, self.lock, self.models = device, threading.Lock(), {}
        determinism.configure(device)
        for spec in specs:
            name, path = spec.split("=", 1)
            model, vocab, icfg, ckpt = load_checkpoint(Path(path), device)
            self.models[name] = (model, vocab, icfg)
            print(json.dumps({"loaded": name, "path": path, "epoch": ckpt.get("epoch")}), flush=True)

    def read(self, name: str, image: Image.Image) -> str:
        model, vocab, icfg = self.models[name]
        with self.lock, torch.inference_mode():
            return predict_images(model, [image], icfg, vocab, self.device)[0]


def make_handler(models: Models):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/models":
                self._send(200, json.dumps(list(models.models)).encode(), "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/predict":
                return self._send(404, b"not found", "text/plain")
            n = int(self.headers.get("Content-Length", "0"))
            if not 0 < n <= MAX_BYTES:
                return self._send(413, b"too large", "text/plain")
            try:
                req = json.loads(self.rfile.read(n))
                data = base64.b64decode(req["image"].split(",", 1)[-1])
                with Image.open(io.BytesIO(data)) as im:
                    im.load()
                    t0 = time.perf_counter()
                    latex = models.read(req.get("model") or next(iter(models.models)), im)
                    ms = (time.perf_counter() - t0) * 1000
                out = {"latex": latex, "compact": normalize_text(latex), "ms": round(ms), "size": im.size}
                self._send(200, json.dumps(out, ensure_ascii=False).encode(), "application/json")
            except Exception as e:  # a bad upload must not stop the demo
                self._send(400, json.dumps({"error": f"{type(e).__name__}: {e}"}).encode(), "application/json")

        def log_message(self, fmt: str, *args) -> None:
            sys.stderr.write("%s %s\n" % (self.log_date_time_string(), fmt % args))

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    models = Models(args.ckpt, torch.device(args.device))
    server = ThreadingHTTPServer((args.host, args.port), make_handler(models))
    print(f"demo on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
