"""Tiny static file server WITH HTTP Range support (PMTiles need byte serving; `python -m http.server` has none).
    python tests/rangeserver.py <folder> <port> [--prefix /spain_los/ [--inject-base]]

``--prefix``: serve the folder under that URL sub-path only (like the production host); requests outside it get 404.
``--inject-base``: add ``<base href="<prefix>">`` to map.html on the fly (what a host serving the map under a sub-path needs).
"""
import os
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


PREFIX = ""
INJECT_BASE = False


class H(SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def translate_path(self, path):
        if PREFIX:
            bare = path.split("?", 1)[0]
            if not (bare + "/").startswith(PREFIX):
                return os.path.join(os.getcwd(), "__outside_prefix__")
            path = "/" + bare[len(PREFIX):]
        return super().translate_path(path)

    def send_head(self):
        bare = self.path.split("?", 1)[0]
        page = self.translate_path(bare + "map.html" if bare.endswith("/") else bare)
        if INJECT_BASE and page.endswith("map.html") and os.path.isfile(page):
            import io

            body = open(page, encoding="utf-8").read().replace("<head>", f'<head><base href="{PREFIX}">', 1).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return io.BytesIO(body)
        rng = self.headers.get("Range")
        path = self.translate_path(self.path)
        if not rng or not os.path.isfile(path):
            return super().send_head()
        m = re.match(r"bytes=(\d*)-(\d*)", rng)
        size = os.path.getsize(path)
        start = int(m.group(1)) if m.group(1) else max(0, size - int(m.group(2)))
        end = min(int(m.group(2)), size - 1) if m.group(1) and m.group(2) else size - 1
        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self._remaining = end - start + 1
        return f

    def copyfile(self, src, dst):
        n = getattr(self, "_remaining", None)
        if n is None:
            return super().copyfile(src, dst)
        dst.write(src.read(n))
        self._remaining = None


if __name__ == "__main__":
    os.chdir(sys.argv[1])
    args = sys.argv[3:]
    if "--prefix" in args:
        PREFIX = "/" + args[args.index("--prefix") + 1].strip("/") + "/"
    INJECT_BASE = "--inject-base" in args
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[2])), H).serve_forever()
