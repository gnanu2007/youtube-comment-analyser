"""Step 1 - Data collection from the YouTube Data API v3."""
import re

import pandas as pd


def extract_video_id(text: str):
    """Accept a full YouTube URL or a bare 11-character video ID."""
    text = text.strip()
    m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{11})", text)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", text):
        return text
    return None


def fetch_comments(api_key: str, video_id: str, max_comments: int = 1000) -> pd.DataFrame:
    """Download top-level comments for one video, following pagination."""
    from googleapiclient.discovery import build  # imported here so tests don't need it

    youtube = build("youtube", "v3", developerKey=api_key)
    rows, token = [], None
    while len(rows) < max_comments:
        resp = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=token,
            textFormat="plainText",
        ).execute()
        for item in resp.get("items", []):
            s = item["snippet"]["topLevelComment"]["snippet"]
            rows.append({
                "video_id": video_id,
                "author": s.get("authorDisplayName"),
                "text": s.get("textDisplay"),
                "likes": s.get("likeCount", 0),
                "replies": item["snippet"].get("totalReplyCount", 0),
                "published_at": s.get("publishedAt"),
            })
        token = resp.get("nextPageToken")
        if not token:
            break
    return pd.DataFrame(rows[:max_comments])
