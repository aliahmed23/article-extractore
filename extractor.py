from flask import Flask, request, jsonify
from newspaper import Article
import nltk
import ssl
import os

from openai import OpenAI

app = Flask(__name__)

def ensure_punkt():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

# --- OpenAI setup ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    # Don't crash the process at import-time on Render; return a clear error on request instead.
    # (Render health checks may still want /health to succeed.)
    pass

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

SYSTEM_PROMPT = """You are a deterministic text formatter for newspaper print layout.

Your ONLY job: reformat the input by adjusting whitespace and line breaks. You must not change the characters of any non-whitespace content.

Hard rules (must follow):
- Do NOT add, remove, reorder, or paraphrase any words or characters. Only whitespace may change (spaces, tabs, newlines).
- Treat any literal backslash-n sequences "\\n" appearing in the input as line breaks.
- Preserve all punctuation, emojis, capitalization, misspellings, and symbols exactly.
- Output plain text only: no markdown, bullets, numbering, code fences, quotes, or commentary.

Formatting rules:
1) Use a single blank line between major sections only. Otherwise, use normal sentence spacing.
2) Never output two blank lines in a row.
3) Heading merge rule:
   - A “heading” is a standalone line that is <= 8 words and does NOT end with punctuation.
   - Convert it into: "Heading: " followed by a space, then the next line’s text on the same line.
4) List merge rule:
   - If there are 2+ consecutive short standalone lines that look like list items, merge them into one line separated by ", ".
   - Keep each item’s characters exactly as-is.
5) Do not hard-wrap lines to a fixed width; keep natural paragraph flow.

Output requirement:
- Return ONLY the fully formatted text.

Safety fallback:
- If any rule conflicts or you are unsure, output the input text unchanged.
"""

@app.get("/health")
def health():
    return jsonify({"ok": True})

@app.post("/extract")
def extract():
    data = request.get_json(silent=True) or {}
    url = data.get("url")
    if not url:
        return jsonify({"ok": False, "error": "Missing url"}), 400

    # Optional: bypass SSL issues (some sites misconfigure cert chains)
    try:
        _create_unverified_https_context = ssl._create_unverified_context
        ssl._create_default_https_context = _create_unverified_https_context
    except Exception:
        pass

    ensure_punkt()

    # --- Extract article ---
    try:
        article = Article(url)
        article.download()
        article.parse()
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": "Failed to download/parse article",
            "details": str(e),
            "url": url
        }), 422

    raw_text = (article.text or "").strip()
    if not raw_text:
        return jsonify({"ok": False, "error": "Extracted text is empty", "url": url}), 422

    # --- OpenAI formatting step ---
    if not openai_client:
        return jsonify({
            "ok": False,
            "error": "Missing OPENAI_API_KEY on server"
        }), 500

    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": raw_text},
            ],
            temperature=0
        )

        formatted_text = (completion.choices[0].message.content or "").strip()
        if not formatted_text:
            return jsonify({
                "ok": False,
                "error": "OpenAI returned empty text"
            }), 502

    except Exception as e:
        return jsonify({
            "ok": False,
            "error": "OpenAI processing failed",
            "details": str(e)
        }), 502

    # NOTE: Per your request, we return the AI-formatted text as `text`.
    # Strongly consider also returning `raw_text` + `text_ai_formatted` later.
    return jsonify({
        "ok": True,
        "url": url,
        "title": article.title,
        "text": formatted_text,
        "top_image": article.top_image,
        "authors": article.authors,
        "publish_date": article.publish_date.isoformat() if article.publish_date else None,
    })