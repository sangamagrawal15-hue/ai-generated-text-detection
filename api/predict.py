"""Vercel serverless function: POST /api/predict  {"text": "..."}

Pure-Python re-implementation of the notebook pipeline
(clean_text -> TF-IDF (1-2 grams) -> calibrated linear SVM), using the
weights exported to model.json. No scikit-learn needed.
"""
import json
import math
import os
import re
from http.server import BaseHTTPRequestHandler

_HERE = os.path.dirname(os.path.abspath(__file__))
_TOKEN = re.compile(r"(?u)\b\w\w+\b")
MAX_CHARS = 20000

_M = None


def _load():
    global _M
    if _M is not None:
        return _M
    for p in (os.path.join(_HERE, "model.json"), os.path.join(_HERE, "..", "model.json")):
        if os.path.exists(p):
            with open(p) as f:
                raw = json.load(f)
            break
    else:
        raise FileNotFoundError("model.json not found")
    n = len(raw["folds"])
    coef_avg = [sum(f["coef"][i] for f in raw["folds"]) / n for i in range(len(raw["terms"]))]
    _M = {
        "index": {t: i for i, t in enumerate(raw["terms"])},
        "terms": raw["terms"],
        "idf": raw["idf"],
        "stop": set(raw["stop_words"]),
        "sublinear": raw["sublinear_tf"],
        "folds": raw["folds"],
        "coef_avg": coef_avg,
    }
    return _M


def clean_text(t):
    """Same preprocessing as in the training notebook."""
    t = str(t).lower()
    t = re.sub(r"http\S+|www\S+", " ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def vectorize(m, cleaned):
    """TF-IDF vector as {feature_index: weight}, L2-normalised."""
    toks = [w for w in _TOKEN.findall(cleaned) if w not in m["stop"]]
    grams = toks + [a + " " + b for a, b in zip(toks, toks[1:])]
    counts = {}
    for g in grams:
        j = m["index"].get(g)
        if j is not None:
            counts[j] = counts.get(j, 0) + 1
    vec = {}
    for j, c in counts.items():
        tf = 1.0 + math.log(c) if m["sublinear"] else float(c)
        vec[j] = tf * m["idf"][j]
    norm = math.sqrt(sum(v * v for v in vec.values()))
    if norm == 0:
        return {}
    return {j: v / norm for j, v in vec.items()}


def probability(m, vec):
    """Mean of the per-fold sigmoid-calibrated SVM scores (as CalibratedClassifierCV does)."""
    ps = []
    for f in m["folds"]:
        s = f["intercept"] + sum(f["coef"][j] * v for j, v in vec.items())
        z = -(f["a"] * s + f["b"])
        ps.append(1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z)))
    return sum(ps) / len(ps)


def verdict(p):
    if p >= 0.65:
        return "ai"
    if p <= 0.35:
        return "human"
    return "inconclusive"


def predict_text(text):
    m = _load()
    words = len(str(text).split())
    cleaned = clean_text(text)
    if not cleaned:
        return {"error": "empty", "message": "Paste some English text to analyze."}
    vec = vectorize(m, cleaned)
    if not vec:
        return {"error": "unknown_words",
                "message": "None of the words in this text are known to the model. Try a longer English passage."}
    p = probability(m, vec)
    contrib = {j: v * m["coef_avg"][j] for j, v in vec.items()}
    ranked = sorted(contrib.items(), key=lambda kv: kv[1])
    terms = m["terms"]
    ai = [terms[j] for j, c in reversed(ranked) if c > 0][:5]
    human = [terms[j] for j, c in ranked if c < 0][:5]
    # strongest single words (unigrams) for highlighting in the page
    uni = [(terms[j], c) for j, c in ranked if " " not in terms[j]]
    mark_ai = [w for w, c in reversed(uni) if c > 0][:15]
    mark_human = [w for w, c in uni if c < 0][:15]
    return {
        "probability": p,
        "verdict": verdict(p),
        "words": words,
        "ai_terms": ai,
        "human_terms": human,
        "mark_ai": mark_ai,
        "mark_human": mark_human,
    }


class handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        try:
            length = int(self.headers.get("content-length") or 0)
            if length > 200000:
                return self._send(413, {"error": "too_large", "message": "Text is too long."})
            data = json.loads(self.rfile.read(length) or b"{}")
            text = str(data.get("text", ""))[:MAX_CHARS]
            result = predict_text(text)
            self._send(400 if "error" in result else 200, result)
        except Exception:
            self._send(500, {"error": "server", "message": "Something went wrong. Please try again."})

    def do_GET(self):
        self._send(405, {"error": "method", "message": "Send a POST request with a text field."})
