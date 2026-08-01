"""utils.py
Utility helpers used across the modules.
"""

import logging
from pathlib import Path
import pandas as pd


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        logging.getLogger("utils").warning("CSV not found: %s", path)
        return pd.DataFrame()
    except Exception:
        logging.getLogger("utils").exception("Failed reading CSV: %s", path)
        return pd.DataFrame()


def guess_message_type(text, media):
    """Heuristic to choose message type: text, image, voice, or unknown"""
    if media and isinstance(media, str):
        m = media.lower()
        if m.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
            return "image"
        if m.endswith(('.mp3', '.wav', '.ogg', '.m4a', '.flac')):
            return "voice"
        if m.endswith(('.mp4', '.mov', '.avi', '.mkv')):
            return "video"
        # treat other attachments as file
        return "file"
    # if very short text and no media, still text
    if text and str(text).strip():
        return "text"
    return "unknown"


def normalize_message_type(raw_type: str) -> str:
    """Map internal message type labels to the allowed HackerRank set.

    Allowed values (conservative set): text, image, audio, video, file, unknown
    """
    if not raw_type:
        return "unknown"
    t = raw_type.lower()
    if t == "voice":
        return "audio"
    if t in ("audio", "sound"):
        return "audio"
    if t in ("text", "txt"):
        return "text"
    if t == "image":
        return "image"
    if t == "video":
        return "video"
    if t == "file":
        return "file"
    return "unknown"
