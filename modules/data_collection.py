"""
data_collection.py — Everything about gathering + labeling the dataset.

Ethics / ToS guardrails (important for the defense):
  - YouTube: uses the OFFICIAL YouTube Data API v3 (commentThreads.list).
    Requires the student's own API key. Respects quotas/rate limits.
  - Reddit: uses PRAW (official Reddit API wrapper, read-only).
  - Facebook / X / news sites: IMPORT-ONLY (CSV/Excel upload or manual
    paste box). We deliberately do NOT ship scrapers that bypass logins,
    evade rate limits, or violate platform Terms of Service — and the GUI
    explains why.

Dataset schema (every row has these columns):
  id, platform, source, date, text, political_bloc, manual_sentiment

  - political_bloc: Pro-Duterte | Pro-Accountability | Undecided/Neutral
  - manual_sentiment (gold label): Positive | Negative | Neutral
    See the labeling guide + stance-vs-tone caution in the app.
"""

import re
import uuid
from datetime import datetime

import pandas as pd

# Valid choices — shared with app.py so labels stay consistent everywhere.
PLATFORMS = ["YouTube", "Reddit", "Facebook (imported)", "X (imported)",
             "News site (imported)", "Other"]
BLOCS = ["Pro-Duterte", "Pro-Accountability", "Undecided/Neutral"]
SENTIMENTS = ["Positive", "Negative", "Neutral"]

REQUIRED_COLUMNS = ["id", "platform", "source", "date", "text",
                    "political_bloc", "manual_sentiment"]


# ---------------------------------------------------------------------------
# Demo data (CLEARLY marked, never mixed silently with real data)
# ---------------------------------------------------------------------------
def load_demo_dataframe(path: str = "data/demo_sample.csv") -> pd.DataFrame:
    """Load the built-in DEMO dataset for testing the app end-to-end."""
    df = pd.read_csv(path)
    # Ensure schema even if the CSV drifts.
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[REQUIRED_COLUMNS].copy()


# ---------------------------------------------------------------------------
# Cleaning / validation helpers
# ---------------------------------------------------------------------------
def anonymize_usernames(df: pd.DataFrame) -> pd.DataFrame:
    """Remove @handles and a possible 'username' column to protect privacy.

    Students: anonymization matters because online comments are personal
    data; dropping handles before analysis/export is a basic research-ethics
    step. We keep the comment TEXT but strip who wrote it.
    """
    df = df.copy()
    if "username" in df.columns:
        df = df.drop(columns=["username"])
    if "author" in df.columns:
        df = df.drop(columns=["author"])
    if "text" in df.columns:
        df["text"] = df["text"].astype(str).apply(
            lambda t: re.sub(r"@\w+", "@user", t)  # "@Maria123" -> "@user"
        )
    return df


def coerce_schema(df: pd.DataFrame, default_platform="Other",
                  default_source="upload") -> pd.DataFrame:
    """Force an arbitrary uploaded table into our required schema.

    Missing columns are filled with sensible defaults; extra columns are
    kept if they look like usernames-free metadata, else dropped down to
    the core schema for consistency.
    """
    df = df.copy()
    # Try to map common alternative column names to ours.
    rename_map = {}
    lower = {c.lower(): c for c in df.columns}
    candidates = {
        "text": ["text", "comment", "content", "message", "body"],
        "platform": ["platform", "site", "source_site"],
        "source": ["source", "video", "video_title", "thread", "url", "post"],
        "date": ["date", "created", "created_at", "published", "timestamp"],
        "political_bloc": ["political_bloc", "bloc", "camp", "stance"],
        "manual_sentiment": ["manual_sentiment", "sentiment", "label",
                             "gold_label"],
    }
    for target, options in candidates.items():
        for opt in options:
            if opt in lower and target not in df.columns:
                rename_map[lower[opt]] = target
                break
    df = df.rename(columns=rename_map)
    # Fill missing columns.
    if "id" not in df.columns:
        df["id"] = [str(uuid.uuid4())[:8] for _ in range(len(df))]
    if "platform" not in df.columns:
        df["platform"] = default_platform
    if "source" not in df.columns:
        df["source"] = default_source
    if "date" not in df.columns:
        df["date"] = datetime.now().strftime("%Y-%m-%d")
    if "text" not in df.columns:
        raise ValueError("Upload has no recognizable text/comment column.")
    if "political_bloc" not in df.columns:
        df["political_bloc"] = "Undecided/Neutral"
    if "manual_sentiment" not in df.columns:
        df["manual_sentiment"] = ""
    # Normalize labels (tolerate lowercase / whitespace variants).
    df["political_bloc"] = df["political_bloc"].astype(str).str.strip()
    df["manual_sentiment"] = df["manual_sentiment"].astype(str).str.strip()
    df.loc[~df["political_bloc"].isin(BLOCS), "political_bloc"] = \
        "Undecided/Neutral"
    valid_sent = set(SENTIMENTS + ["", "nan", "None"])
    df.loc[~df["manual_sentiment"].isin(valid_sent), "manual_sentiment"] = ""
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)
    return anonymize_usernames(df[REQUIRED_COLUMNS])


def parse_manual_paste(raw_text: str, platform: str,
                       source: str) -> pd.DataFrame:
    """Turn the manual paste box (one comment per line) into a DataFrame.

    Empty lines are ignored. Each non-empty line becomes one row with a
    fresh id, today's date, blank labels (to be filled in the labeling
    tool), and anonymized handles.
    """
    lines = [ln.strip() for ln in (raw_text or "").splitlines()
             if ln.strip()]
    if not lines:
        raise ValueError("Paste box is empty — paste at least one comment.")
    rows = [{
        "id": str(uuid.uuid4())[:8],
        "platform": platform,
        "source": source or "manual entry",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "text": anonymize_usernames(
            pd.DataFrame([{"text": ln}]))["text"].iloc[0],
        "political_bloc": "Undecided/Neutral",
        "manual_sentiment": "",
    } for ln in lines]
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


# ---------------------------------------------------------------------------
# Official API collectors (graceful errors, no ToS violations)
# ---------------------------------------------------------------------------
def extract_youtube_video_id(url_or_id: str) -> str:
    """Accept a full YouTube URL or a bare 11-char video id; return the id."""
    s = (url_or_id or "").strip()
    if not s:
        raise ValueError("Please enter a YouTube video URL or ID.")
    # Bare id?
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
        return s
    # Common URL forms: v=..., youtu.be/..., /embed/..., /shorts/...
    m = (re.search(r"[?&]v=([A-Za-z0-9_-]{11})", s)
         or re.search(r"youtu\.be/([A-Za-z0-9_-]{11})", s)
         or re.search(r"/(?:embed|shorts|live)/([A-Za-z0-9_-]{11})", s))
    if m:
        return m.group(1)
    raise ValueError("Could not find a video ID in that input. "
                     "Paste the full watch URL or the 11-character ID.")


def fetch_youtube_comments(video_id: str, api_key: str,
                           max_results: int = 100) -> pd.DataFrame:
    """Fetch top-level comments via the official YouTube Data API v3.

    Raises a helpful error if the key/quota/video is wrong — the app shows
    this message directly so the student knows what to fix.
    """
    if not api_key:
        raise ValueError("Missing YouTube API key. Create one in Google "
                         "Cloud Console and paste it in the app (see README).")
    try:
        from googleapiclient.discovery import build
    except ImportError as e:
        raise ImportError("google-api-python-client is not installed. Run: "
                          "py -m pip install -r requirements.txt") from e
    youtube = build("youtube", "v3", developerKey=api_key)
    rows, page_token = [], None
    try:
        while len(rows) < max_results:
            resp = youtube.commentThreads().list(
                part="snippet", videoId=video_id,
                maxResults=min(100, max_results - len(rows)),
                pageToken=page_token, textFormat="plainText").execute()
            for item in resp.get("items", []):
                sn = item["snippet"]["topLevelComment"]["snippet"]
                rows.append({
                    "id": str(uuid.uuid4())[:8],
                    "platform": "YouTube",
                    "source": video_id,
                    "date": sn.get("publishedAt", "")[:10],
                    "text": sn.get("textDisplay", ""),
                    "political_bloc": "Undecided/Neutral",
                    "manual_sentiment": "",
                })
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
    except Exception as e:  # surface API errors (quota, disabled comments…)
        raise RuntimeError(f"YouTube API error: {e}") from e
    if not rows:
        raise RuntimeError("No comments returned (video may have comments "
                           "disabled, or the ID/key is wrong).")
    return anonymize_usernames(
        pd.DataFrame(rows, columns=REQUIRED_COLUMNS))


def fetch_reddit_comments(subreddit: str, keyword: str, client_id: str,
                          client_secret: str, user_agent: str,
                          limit: int = 100) -> pd.DataFrame:
    """Fetch Reddit comments via PRAW (official API, read-only, rate-limited).

    Searches recent submissions in r/<subreddit> matching <keyword>, then
    collects up to <limit> comments. PRAW enforces Reddit's rate limits.
    """
    if not (client_id and client_secret):
        raise ValueError("Missing Reddit API credentials. Create a 'script' "
                         "app at reddit.com/prefs/apps to get them (README).")
    try:
        import praw
    except ImportError as e:
        raise ImportError("praw is not installed. Run: "
                          "py -m pip install -r requirements.txt") from e
    reddit = praw.Reddit(client_id=client_id, client_secret=client_secret,
                         user_agent=user_agent or "impeachment-research/1.0")
    rows = []
    try:
        for sub in reddit.subreddit(subreddit).search(keyword or "",
                                                      limit=max(1, limit // 5)):
            sub.comments.replace_more(limit=0)  # no deep pagination storms
            for c in sub.comments.list()[: max(1, limit // 5)]:
                rows.append({
                    "id": str(uuid.uuid4())[:8],
                    "platform": "Reddit",
                    "source": f"r/{subreddit}: {sub.title[:60]}",
                    "date": datetime.utcfromtimestamp(
                        c.created_utc).strftime("%Y-%m-%d"),
                    "text": c.body or "",
                    "political_bloc": "Undecided/Neutral",
                    "manual_sentiment": "",
                })
                if len(rows) >= limit:
                    break
            if len(rows) >= limit:
                break
    except Exception as e:
        raise RuntimeError(f"Reddit API error: {e}") from e
    if not rows:
        raise RuntimeError("No Reddit comments found — try another "
                           "subreddit/keyword.")
    return anonymize_usernames(
        pd.DataFrame(rows, columns=REQUIRED_COLUMNS))


# ---------------------------------------------------------------------------
# Inter-annotator agreement
# ---------------------------------------------------------------------------
def cohens_kappa(a1: list, a2: list) -> float:
    """Compute Cohen's kappa for two annotators' label lists (no sklearn).

    Formula: κ = (Po − Pe) / (1 − Pe), where Po = observed agreement and
    Pe = expected agreement by chance from each annotator's marginals.
    Only overlapping (both nonblank) positions are compared.
    """
    pairs = [(str(x).strip(), str(y).strip()) for x, y in zip(a1, a2)
             if str(x).strip() and str(y).strip()]
    if len(pairs) < 5:
        raise ValueError("Need at least 5 doubly-labeled comments to "
                         "compute kappa.")
    labels = sorted(set([p[0] for p in pairs] + [p[1] for p in pairs]))
    n = len(pairs)
    po = sum(1 for x, y in pairs if x == y) / n
    pe = 0.0
    for lab in labels:
        p1 = sum(1 for x, _ in pairs if x == lab) / n
        p2 = sum(1 for _, y in pairs if y == lab) / n
        pe += p1 * p2
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1 - pe)
