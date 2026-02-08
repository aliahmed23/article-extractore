from flask import Flask, request, jsonify
from newspaper import Article, Config
import ssl
import os
import httpx

from openai import OpenAI

app = Flask(__name__)

# --- Configuration ---
# Maximum characters to send to OpenAI (prevents timeouts on very long articles)
MAX_AI_INPUT_CHARS = 23000

# Maximum words to include in output (truncate longer articles)
MAX_OUTPUT_WORDS = 2000

# Realistic browser User-Agent to avoid being blocked by websites
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

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


def truncate_to_words(text, max_words):
    """
    Truncate text to a maximum number of words (without adding any suffix).
    
    Args:
        text: The text to potentially truncate
        max_words: Maximum number of words to keep
    
    Returns:
        tuple: (text, was_truncated)
    """
    words = text.split()
    
    if len(words) <= max_words:
        return (text, False)
    
    # Take first max_words and rejoin
    truncated = " ".join(words[:max_words])
    return (truncated, True)


def call_openai_with_retry(text, max_retries=1):
    """
    Call OpenAI to format text, with retry logic.
    
    Returns:
        tuple: (formatted_text, error_message)
        - On success: (formatted_text, None)
        - On failure: (None, error_message)
    """
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            completion = openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                temperature=0,
            )
            
            formatted_text = (completion.choices[0].message.content or "").strip()
            
            # Check for empty response
            if not formatted_text:
                last_error = "OpenAI returned empty text"
                continue  # Retry
            
            return (formatted_text, None)  # Success
            
        except Exception as e:
            last_error = str(e)
            # Continue to retry if we have attempts left
            continue
    
    # All retries exhausted
    return (None, last_error)


@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.post("/extract")
def extract():
    """
    Extract article content from a URL.
    
    This endpoint is designed to be robust for automation (e.g., Zapier):
    - Never fails due to OpenAI issues
    - Falls back to raw text if AI formatting fails or is skipped
    - Only fails if the article itself cannot be extracted
    """
    data = request.get_json(silent=True) or {}
    url = data.get("url")

    # --- Validation: URL is required ---
    if not url:
        return jsonify({"ok": False, "error": "Missing url"}), 400

    # --- SSL bypass for misconfigured sites ---
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
    except Exception:
        pass

    # --- Configure newspaper3k with timeout and User-Agent ---
    config = Config()
    config.request_timeout = 15
    config.browser_user_agent = USER_AGENT
    # Disable memoization to avoid stale cache issues
    config.memoize_articles = False

    # --- Extract article ---
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
    
    # --- Validation: Extracted text must not be empty ---
    if not raw_text:
        return jsonify({
            "ok": False,
            "error": "Extracted text is empty",
            "url": url
        }), 422

    # --- Calculate word count from full article ---
    word_count = len(raw_text.split())

    # --- Truncate BEFORE sending to AI (saves tokens) ---
    text_for_processing, was_truncated = truncate_to_words(raw_text, MAX_OUTPUT_WORDS)

    # --- OpenAI formatting (best-effort, not required) ---
    ai_used = False
    ai_error = None
    formatted_text = text_for_processing  # Default: use (possibly truncated) raw text
    
    # Check if article is too long for OpenAI (by character count)
    if len(text_for_processing) > MAX_AI_INPUT_CHARS:
        ai_error = f"Skipped: article exceeds {MAX_AI_INPUT_CHARS} character limit ({len(text_for_processing)} chars)"
    else:
        # Attempt OpenAI formatting with retry
        result, error = call_openai_with_retry(text_for_processing, max_retries=1)
        
        if result:
            # Success: use AI-formatted text
            formatted_text = result
            ai_used = True
        else:
            # Failure: keep raw text, record the error
            ai_error = error

    # --- Add "Read more" suffix if article was truncated ---
    if was_truncated:
        formatted_text += f"\n\n[Read more at: {url}]"

    # --- Build successful response ---
    # Note: ok=True as long as extraction succeeded (AI is best-effort)
    return jsonify({
        "ok": True,
        "url": url,
        "title": article.title,
        "text_ai_formatted": formatted_text,
        "ai_used": ai_used,
        "ai_error": ai_error,
        "word_count": word_count,
        "truncated": was_truncated,
        "top_image": article.top_image,
        "authors": article.authors,
        "publish_date": article.publish_date.isoformat() if article.publish_date else None,
    })
