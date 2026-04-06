"""
Parser dedicated to tiktok_database.csv (manual labels + links).

Outputs:
  - tiktok_database_parsed.csv
  - tiktok_database_combined.csv
"""

import argparse
import datetime
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd


AI_HASHTAGS = {
    "aigc", "ai", "aiart", "aigenerated", "aicreated", "aivideo",
    "generativeai", "syntheticmedia", "chatgpt", "openai",
    "midjourney", "dalle", "stablediffusion", "runway", "kling",
    "pika", "heygen", "luma", "deepfake",
}

CAPTION_AI_REGEX = re.compile(
    r"\b(ai[- ]?generated|ai[- ]?created|made\s+(with|by|using)\s+ai|aigc|synthetic\s+media|deepfake)\b",
    re.IGNORECASE,
)


def normalize_manual_label(x: Any) -> Optional[int]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip().upper()
    if s in ("AI", "1", "TRUE"):
        return 1
    if s in ("NOT AI", "NOT_AI", "NON AI", "NON-AI", "NO", "0", "FALSE"):
        return 0
    return None


def extract_video_id(url: str) -> Optional[str]:
    if not isinstance(url, str):
        return None
    m = re.search(r"/video/(\d+)", url)
    return m.group(1) if m else None


def extract_hashtags(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    return re.findall(r"#(\w+)", text, flags=re.UNICODE)


def creator_signal_from_desc(text: str) -> Tuple[bool, str]:
    if not isinstance(text, str):
        return False, ""
    tags = [t.lower() for t in extract_hashtags(text)]
    tag_hit = any(t in AI_HASHTAGS for t in tags)
    m = CAPTION_AI_REGEX.search(text)
    if tag_hit:
        return True, "hashtag"
    if m:
        return True, m.group(0)
    return False, ""


def load_metadata_index(metadata_dir: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not metadata_dir:
        return {}
    base = Path(metadata_dir)
    files = sorted(base.glob("*.json"))
    index: Dict[str, Dict[str, Any]] = {}
    for p in files:
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            item = {}
            if isinstance(payload, dict):
                # Shape 1: direct itemInfo.itemStruct
                item = payload.get("itemInfo", {}).get("itemStruct", {}) or {}
                # Shape 2: SIGI_STATE / webapp.video-detail
                if not item:
                    item = (
                        payload.get("__DEFAULT_SCOPE__", {})
                        .get("webapp.video-detail", {})
                        .get("itemInfo", {})
                        .get("itemStruct", {})
                    ) or {}
                # Shape 3: ItemModule keyed by video_id
                if not item:
                    item_module = payload.get("ItemModule", {})
                    if isinstance(item_module, dict) and item_module:
                        item = next(iter(item_module.values()))
                        if not isinstance(item, dict):
                            item = {}
        except Exception:
            continue
        if not item:
            continue
        vid = str(item.get("id") or p.stem)
        author = item.get("author", {}) if isinstance(item.get("author"), dict) else {}
        stats = item.get("stats", {}) if isinstance(item.get("stats"), dict) else {}
        index[vid] = {
            "video_id": vid,
            "description": item.get("desc", ""),
            "create_time": item.get("createTime"),
            "author": author.get("uniqueId"),
            "play_count": stats.get("playCount"),
            "like_count": stats.get("diggCount"),
            "comment_count": stats.get("commentCount"),
            "share_count": stats.get("shareCount"),
            "is_aigc": item.get("IsAigc"),
            "ai_gc_description": item.get("AIGCDescription"),
            "source_metadata_file": p.name,
        }
    return index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default="tiktok_database.csv")
    ap.add_argument("--metadata-dir", default="metadata")
    ap.add_argument("--fetch-report", default=None, help="Optional fetch report CSV to backfill resolved video_id/final_url")
    ap.add_argument("--out-parsed", default="tiktok_database_parsed.csv")
    ap.add_argument("--out-combined", default="tiktok_database_combined.csv")
    ap.add_argument("--cutoff", default="2024-02-01")
    args = ap.parse_args()

    cutoff = datetime.datetime.strptime(args.cutoff, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
    manual = pd.read_csv(args.sheet, low_memory=False)
    if "LINK" not in manual.columns:
        raise ValueError("Expected LINK column in tiktok_database.csv")

    metadata_index = load_metadata_index(args.metadata_dir)
    link_to_video: Dict[str, str] = {}
    if args.fetch_report:
        rep = pd.read_csv(args.fetch_report, low_memory=False)
        if {"input_url", "video_id"}.issubset(rep.columns):
            for _, r in rep.iterrows():
                iu = r.get("input_url")
                vv = r.get("video_id")
                if isinstance(iu, str) and isinstance(vv, str) and vv.strip():
                    link_to_video[iu] = vv.strip()
    parsed_rows = []
    for _, row in manual.iterrows():
        link = row.get("LINK")
        vid = extract_video_id(link) or link_to_video.get(link)
        md = metadata_index.get(vid, {})
        desc = md.get("description", "")
        creator_signal, creator_match = creator_signal_from_desc(desc)
        create_dt = pd.to_datetime(pd.to_numeric(pd.Series([md.get("create_time")]), errors="coerce"), unit="s", utc=True, errors="coerce").iloc[0]
        era = "post_ai" if pd.notna(create_dt) and create_dt >= cutoff else "pre_ai"
        parsed_rows.append(
            {
                "input_url_full": link,
                "video_id": vid,
                "status": "ok" if vid else "no_video_id_in_url",
                "manual_ai_raw": row.get("AI/NOT AI"),
                "manual_ai": normalize_manual_label(row.get("AI/NOT AI")),
                "description": desc,
                "author": md.get("author"),
                "create_time": md.get("create_time"),
                "create_dt": create_dt,
                "era": era,
                "play_count": pd.to_numeric(md.get("play_count"), errors="coerce"),
                "like_count": pd.to_numeric(md.get("like_count"), errors="coerce"),
                "comment_count": pd.to_numeric(md.get("comment_count"), errors="coerce"),
                "share_count": pd.to_numeric(md.get("share_count"), errors="coerce"),
                "is_aigc": md.get("is_aigc"),
                "signal_platform": bool(md.get("is_aigc") is True),
                "signal_creator_tag": bool(creator_signal),
                "creator_signal_match": creator_match,
                "source_metadata_file": md.get("source_metadata_file"),
            }
        )

    parsed = pd.DataFrame(parsed_rows)
    parsed["signal_manual"] = parsed["manual_ai"].astype("Int64")
    parsed["signals_platform_vs_manual"] = parsed.apply(
        lambda r: (bool(r["signal_platform"]) == bool(r["signal_manual"])) if pd.notna(r["signal_manual"]) else None,
        axis=1,
    )
    parsed["signals_creator_vs_manual"] = parsed.apply(
        lambda r: (bool(r["signal_creator_tag"]) == bool(r["signal_manual"])) if pd.notna(r["signal_manual"]) else None,
        axis=1,
    )
    parsed["signals_platform_vs_creator"] = parsed["signal_platform"] == parsed["signal_creator_tag"]

    parsed["engagement_total"] = parsed[["play_count", "like_count", "comment_count", "share_count"]].sum(axis=1, min_count=1)
    parsed["like_rate"] = (parsed["like_count"] / parsed["play_count"].replace(0, pd.NA)).round(6)
    parsed["comment_rate"] = (parsed["comment_count"] / parsed["play_count"].replace(0, pd.NA)).round(6)

    combined = manual.merge(parsed, left_on="LINK", right_on="input_url_full", how="left")
    parsed.to_csv(args.out_parsed, index=False)
    combined.to_csv(args.out_combined, index=False)

    print(f"Wrote: {args.out_parsed}")
    print(f"Wrote: {args.out_combined}")
    print(f"Rows: {len(parsed)}")
    print("manual_ai counts:")
    print(parsed["manual_ai"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
