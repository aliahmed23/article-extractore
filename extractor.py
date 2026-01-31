from flask import Flask, request, jsonify
from newspaper import Article
import nltk
import ssl

app = Flask(__name__)

def ensure_punkt():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

@app.get("/health")
def health():
    return jsonify({"ok": True})

@app.post("/extract")
def extract():
    data = request.get_json(silent=True) or {}
    url = data.get("url")
    if not url:
        return jsonify({"ok": False, "error": "Missing url"}), 400

    # optional: bypass SSL issues
    try:
        _create_unverified_https_context = ssl._create_unverified_context
        ssl._create_default_https_context = _create_unverified_https_context
    except Exception:
        pass

    ensure_punkt()

    article = Article(url)
    article.download()
    article.parse()

    if not (article.text or "").strip():
        return jsonify({"ok": False, "error": "Extracted text is empty"}), 422

    return jsonify({
        "ok": True,
        "url": url,
        "title": article.title,
        "text": article.text,
        "top_image": article.top_image,
        "authors": article.authors,
        "publish_date": article.publish_date.isoformat() if article.publish_date else None,
    })