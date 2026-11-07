"""Local test server (no Vercel needed):  python dev_server.py  ->  http://localhost:3000"""
import json, os, sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "api"))
import predict  # noqa: E402


class Dev(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=os.path.join(HERE, "public"), **k)

    def do_POST(self):
        if self.path != "/api/predict":
            self.send_error(404)
            return
        n = int(self.headers.get("content-length") or 0)
        data = json.loads(self.rfile.read(n) or b"{}")
        res = predict.predict_text(str(data.get("text", ""))[: predict.MAX_CHARS])
        body = json.dumps(res).encode()
        self.send_response(400 if "error" in res else 200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    print("Open http://localhost:3000")
    ThreadingHTTPServer(("", 3000), Dev).serve_forever()
