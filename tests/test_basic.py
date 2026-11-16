import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.collector import extract_video_id
from src.preprocess import clean_text


def test_extract_video_id_from_urls():
    assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("not a link") is None


def test_clean_text_removes_noise():
    assert clean_text("Great VIDEO!!! 😀 https://x.com/a 123") == "great video"
