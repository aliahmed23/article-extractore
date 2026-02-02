from flask import Flask, request, jsonify
from newspaper import Article, Config
import ssl
import os
import httpx

from openai import OpenAI

app = Flask(__name__)

# --- OpenAI client with timeout ---
openai_client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    http_client=httpx.Client(timeout=30.0),
)

SYSTEM_PROMPT = """You are a deterministic text formatter for newspaper print layout.

Your ONLY job: reformat the input by adjusting whitespace and line breaks. You must not change the characters of any non-whitespace content.

Hard rules (must follow):
- Do NOT add, remove, reorder, or paraphrase any words or characters. Only whitespace may change.
- Treat any literal backslash-n sequences "\\n" appearing in the input as line breaks.
- Preserve all punctuation, emojis, capitalization, misspellings, and symbols exactly.
- Output plain text only: no markdown, bullets, numbering, code fences, quotes, or commentary.

Formatting rules:
1) Use a single blank line between major sections only.
2) Never output two blank lines in a row.
3) Heading merge rule:
   - A heading is a standalone line <= 8 words that does NOT end with punctuation.
   - Convert to: "Heading: " + next line on the same line.
4) List merge rule:
   - Merge consecutive short standalone list-like lines into one line separated by ", ".
5) Do not hard-wrap lines to a fixed width.

Output requirement:
- Return ONLY the fully formatted text.

Safety fallback:
- If unsure, return the input text unchanged.
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

    # Optional SSL bypass for misconfigured sites
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
    except Exception:
        pass

    # --- newspaper3k with timeout ---
    config = Config()
    config.request_timeout = 15

    try:
        article = Article(url, config=config)
        article.download()
        article.parse()
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": "Failed to download or parse article",
            "details": str(e),
            "url": url
        }), 422

    raw_text = (article.text or "").strip()
    if not raw_text:
        return jsonify({
            "ok": False,
            "error": "Extracted text is empty",
            "url": url
        }), 422

    # --- OpenAI formatting ---
    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": raw_text},
            ],
            temperature=0,
        )

        formatted_text = (completion.choices[0].message.content or "").strip()
        if not formatted_text:
            raise ValueError("OpenAI returned empty text")

    except Exception as e:
        return jsonify({
            "ok": False,
            "error": "OpenAI processing failed",
            "details": str(e)
        }), 502

    return jsonify({
        "ok": True,
        "url": url,
        "title": article.title,
        "text": formatted_text,
        "top_image": article.top_image,
        "authors": article.authors,
        "publish_date": article.publish_date.isoformat() if article.publish_date else None,
    })