"""
Metadata parser for random-sample TikTok dataset.
"""

import argparse
import datetime
import json
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple

import pandas as pd

AI_HASHTAGS: set = {
    "aigc", "aiart", "aigenerated", "aicreated", "aiartwork",
    "aianimation", "aivideo", "aifilm", "aiclip", "aicontentcreator",
    "generativeai", "syntheticmedia", "artificialintelligence",
    "ai_generated", "ai_art",
    "soraai", "sora", "runwayml", "runway", "kling", "klingai",
    "pika", "pikaart", "pikalabs", "heygen", "heygenai",
    "midjourney", "stablediffusion", "dalle", "dalle3",
    "luma", "lumaai", "dreamina", "haiper",
    "invideo", "morphstudio", "genmo",
    # broader creator-side AI mentions
    "chatgpt", "openai", "aiavatar", "aiedit", "aiediting", "aitools",
    "aitool", "aifilter", "aifilters", "deepfake", "deepfakes",
}

CAPTION_AI_REGEX = re.compile(
    r"""
    \b(
        ai[- ]?generated|ai[- ]?created|ai[- ]?made|ai[- ]?art\b|ai[- ]?video|ai[- ]?animation
      | made\s+(with|by|using)\s+ai\b|created\s+(with|by|using)\s+ai\b|generated\s+by\s+ai\b
      | this\s+is\s+ai\b|not\s+real[,\.\s]|digitally\s+created|synthetic\s+media|aigc\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

POTENTIAL_AI_TEXT_REGEX = re.compile(
    r"""
    \b(
        ai\b|aigc|ai[- ]?generated|ai[- ]?made|artificial\s+intelligence|synthetic
      | chatgpt|openai|midjourney|dall[- ]?e|stable\s*diffusion|runway|kling|pika|heygen|luma
      | deepfake|face\s*swap|text[- ]?to[- ]?video|image[- ]?to[- ]?video
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def normalize_manual_label(x: Any) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip().upper()
    if s == "AI":
        return 1
    if s in ("NOT AI", "NOT_AI", "NON AI", "NON-AI", "NONAI", "NO", "0", "FALSE"):
        return 0
    return None


def pick_label_column(df: pd.DataFrame) -> Optional[str]:
    preferred = ("AI/NOT AI", "AI_NOT_AI", "LABEL", "AI")
    for c in df.columns:
        if str(c).strip().upper() in preferred:
            return c
    return None


def coerce_bool(x: Any) -> Optional[bool]:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)) and x in (0, 1):
        return bool(int(x))
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("true", "1", "yes", "y"):
            return True
        if s in ("false", "0", "no", "n", ""):
            return False
    return None


def normalize_empty_text(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).strip()
    if s.lower() in {"", "nan", "none", "null"}:
        return ""
    if s in {'""', "''", '""""""'}:
        return ""
    return s


def extract_hashtags(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    return re.findall(r"#(\w+)", text, re.UNICODE)


def hashtag_ai_analysis(text: str) -> Tuple[bool, List[str]]:
    tags = extract_hashtags(text)
    matches = [t for t in tags if t.lower() in AI_HASHTAGS]
    return bool(matches), matches


def caption_ai_analysis(text: str) -> Tuple[bool, str]:
    if not isinstance(text, str):
        return False, ""
    m = CAPTION_AI_REGEX.search(text)
    return (True, m.group(0).strip()) if m else (False, "")


def potential_ai_text_analysis(text: str) -> Tuple[bool, str]:
    if not isinstance(text, str):
        return False, ""
    m = POTENTIAL_AI_TEXT_REGEX.search(text)
    return (True, m.group(0).strip()) if m else (False, "")


def col_or_default(meta: pd.DataFrame, col: str, default: Any) -> pd.Series:
    if col in meta.columns:
        return meta[col]
    return pd.Series([default] * len(meta), index=meta.index)


def build_parsed_df(meta: pd.DataFrame) -> pd.DataFrame:
    desc = col_or_default(meta, "description", "").fillna("").astype(str)
    ht_res = desc.apply(hashtag_ai_analysis)
    cap_res = desc.apply(caption_ai_analysis)
    broad_res = desc.apply(potential_ai_text_analysis)

    return pd.DataFrame(
        {
            "video_id": col_or_default(meta, "video_id", pd.NA),
            "author": col_or_default(meta, "author_unique_id", pd.NA),
            "create_time": col_or_default(meta, "create_timestamp", pd.NA),
            "description": desc,
            "play_count": pd.to_numeric(col_or_default(meta, "stats_view_count", pd.NA), errors="coerce"),
            "like_count": pd.to_numeric(col_or_default(meta, "stats_like_count", pd.NA), errors="coerce"),
            "comment_count": pd.to_numeric(col_or_default(meta, "stats_comment_count", pd.NA), errors="coerce"),
            "share_count": pd.to_numeric(col_or_default(meta, "stats_share_count", pd.NA), errors="coerce"),
            "is_aigc": col_or_default(meta, "video_is_ai_gc", pd.NA).apply(coerce_bool),
            "aigc_badge_type": col_or_default(meta, "ai_gc_label_type", pd.NA),
            "ai_gc_description_parsed": col_or_default(meta, "ai_gc_description", pd.NA),
            "hashtags": desc.apply(lambda t: ";".join(extract_hashtags(t))),
            "hashtag_ai_signal": ht_res.apply(lambda r: r[0]),
            "hashtag_ai_matches": ht_res.apply(lambda r: ";".join(r[1])),
            "caption_ai_signal": cap_res.apply(lambda r: r[0]),
            "caption_ai_match": cap_res.apply(lambda r: r[1]),
            "potential_ai_text_signal": broad_res.apply(lambda r: r[0]),
            "potential_ai_text_match": broad_res.apply(lambda r: r[1]),
            "status": "ok",
            "raw_source_used": "metadata_csv",
        }
    )


def build_signal_columns(df: pd.DataFrame, cutoff: datetime.datetime) -> pd.DataFrame:
    out = df.copy()

    def _platform_signal(row: pd.Series) -> bool:
        aigc = coerce_bool(row.get("is_aigc"))
        badge = normalize_empty_text(row.get("aigc_badge_type")).lower()
        desc = normalize_empty_text(row.get("ai_gc_description"))
        if aigc is True:
            return True
        if badge not in ("", "none", "nan", "0", "0.0"):
            return True
        if desc:
            return True
        return False

    out["signal_platform"] = out.apply(_platform_signal, axis=1)
    out["signal_creator_tag"] = out["hashtag_ai_signal"].fillna(False).astype(bool) | out["caption_ai_signal"].fillna(False).astype(bool)
    out["signal_potential_text"] = out.get("potential_ai_text_signal", False)
    out["signal_visual_model"] = out.get("signal_visual_model", pd.NA)

    if "manual_ai" in out.columns:
        out["signal_manual"] = out["manual_ai"].apply(normalize_manual_label).astype("Int64")
    else:
        out["signal_manual"] = pd.NA

    def _agree(row: pd.Series, a: str, b: str):
        va, vb = row.get(a), row.get(b)
        try:
            if pd.isna(va) or pd.isna(vb):
                return None
        except (TypeError, ValueError):
            pass
        return bool(va) == bool(vb)

    out["signals_platform_vs_creator"] = out.apply(lambda r: _agree(r, "signal_platform", "signal_creator_tag"), axis=1)
    out["signals_platform_vs_manual"] = out.apply(lambda r: _agree(r, "signal_platform", "signal_manual"), axis=1)
    out["signals_creator_vs_manual"] = out.apply(lambda r: _agree(r, "signal_creator_tag", "signal_manual"), axis=1)
    out["signals_agree_all"] = out.apply(lambda r: _agree(r, "signal_platform", "signal_creator_tag") if pd.notna(r.get("signal_manual")) else None, axis=1)
    out["disclosure_gap"] = out["signal_creator_tag"].astype(bool) & ~out["signal_platform"].astype(bool)
    out["signal_potential_ai_any"] = (
        out["signal_platform"].astype(bool)
        | out["signal_creator_tag"].astype(bool)
        | out["signal_potential_text"].fillna(False).astype(bool)
    )

    create_time_numeric = pd.to_numeric(out["create_time"], errors="coerce")
    out["create_dt"] = pd.to_datetime(create_time_numeric, unit="s", utc=True, errors="coerce")
    cutoff_aware = cutoff if cutoff.tzinfo else cutoff.replace(tzinfo=datetime.timezone.utc)
    out["era"] = out["create_dt"].apply(lambda d: "post_ai" if pd.notna(d) and d >= cutoff_aware else "pre_ai")

    for col in ("play_count", "like_count", "comment_count", "share_count"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["engagement_total"] = out[["play_count", "like_count", "comment_count", "share_count"]].sum(axis=1, min_count=1)
    out["like_rate"] = (out["like_count"] / out["play_count"].replace(0, pd.NA)).round(6)
    out["comment_rate"] = (out["comment_count"] / out["play_count"].replace(0, pd.NA)).round(6)
    out["tiktok_labeled_aigc"] = out["signal_platform"].astype(bool)
    return out


def load_metadata(args: argparse.Namespace) -> pd.DataFrame:
    # Backward compatible: --sheet can still point to a single file.
    # New behavior: use --metadata-dir to load/concat many CSVs.
    if args.metadata_dir:
        base = Path(args.metadata_dir)
        csv_files = sorted(base.glob(args.pattern))
        if not csv_files:
            raise FileNotFoundError(f"No metadata CSV files found in '{args.metadata_dir}' with pattern '{args.pattern}'")
        frames = []
        for p in csv_files:
            part = pd.read_csv(p, low_memory=False)
            part["source_metadata_file"] = p.name
            frames.append(part)
        merged = pd.concat(frames, ignore_index=True)
        print(f"Loaded {len(csv_files)} files from {args.metadata_dir} ({len(merged)} rows).")
        return merged

    sheet_path = Path(args.sheet)
    if sheet_path.is_dir():
        csv_files = sorted(sheet_path.glob(args.pattern))
        if not csv_files:
            raise FileNotFoundError(f"No metadata CSV files found in '{args.sheet}' with pattern '{args.pattern}'")
        frames = []
        for p in csv_files:
            part = pd.read_csv(p, low_memory=False)
            part["source_metadata_file"] = p.name
            frames.append(part)
        merged = pd.concat(frames, ignore_index=True)
        print(f"Loaded {len(csv_files)} files from {args.sheet} ({len(merged)} rows).")
        return merged

    return pd.read_csv(args.sheet, low_memory=False)


def load_metadata_json_dir(metadata_dir: str, pattern: str) -> pd.DataFrame:
    base = Path(metadata_dir)
    json_files = sorted(base.glob(pattern))
    if not json_files:
        raise FileNotFoundError(f"No metadata JSON files found in '{metadata_dir}' with pattern '{pattern}'")

    rows = []
    for p in json_files:
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        item = (
            payload.get("itemInfo", {})
            .get("itemStruct", {})
        )
        author = item.get("author", {}) if isinstance(item.get("author"), dict) else {}
        stats = item.get("stats", {}) if isinstance(item.get("stats"), dict) else {}
        stats_v2 = item.get("statsV2", {}) if isinstance(item.get("statsV2"), dict) else {}
        creator_ai_comment = item.get("creatorAIComment", {}) if isinstance(item.get("creatorAIComment"), dict) else {}

        rows.append(
            {
                "video_id": item.get("id"),
                "description": item.get("desc", ""),
                "create_timestamp": item.get("createTime"),
                "author_unique_id": author.get("uniqueId"),
                "stats_view_count": stats.get("playCount", stats_v2.get("playCount")),
                "stats_like_count": stats.get("diggCount", stats_v2.get("diggCount")),
                "stats_comment_count": stats.get("commentCount", stats_v2.get("commentCount")),
                "stats_share_count": stats.get("shareCount", stats_v2.get("shareCount")),
                "video_is_ai_gc": item.get("IsAigc"),
                "ai_gc_label_type": item.get("AIGCLabelType"),
                "ai_gc_description": item.get("AIGCDescription"),
                "creator_ai_topic": creator_ai_comment.get("hasAITopic"),
                "source_metadata_file": p.name,
            }
        )

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} rows from {len(json_files)} JSON files in {metadata_dir}.")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default="metadata.csv")
    ap.add_argument("--metadata-dir", default=None, help="Directory containing metadata CSV files")
    ap.add_argument("--pattern", default="*.csv", help="Glob pattern for metadata files when using a directory")
    ap.add_argument("--json-metadata-dir", default=None, help="Directory containing metadata JSON files")
    ap.add_argument("--json-pattern", default="*.json", help="Glob pattern for JSON metadata files")
    ap.add_argument("--out-parsed", default="metadata_parsed.csv")
    ap.add_argument("--out-combined", default="metadata_combined.csv")
    ap.add_argument("--cutoff", default="2024-02-01")
    args = ap.parse_args()

    cutoff = datetime.datetime.strptime(args.cutoff, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
    if args.json_metadata_dir:
        meta = load_metadata_json_dir(args.json_metadata_dir, args.json_pattern)
    else:
        meta = load_metadata(args)
    parsed = build_parsed_df(meta)

    label_col = pick_label_column(meta)
    parsed["manual_ai"] = meta[label_col].apply(normalize_manual_label) if label_col else None

    combined = pd.concat([meta, parsed], axis=1)
    combined = build_signal_columns(combined, cutoff)

    parsed.to_csv(args.out_parsed, index=False)
    combined.to_csv(args.out_combined, index=False)
    print(f"Wrote: {args.out_parsed}")
    print(f"Wrote: {args.out_combined}")


if __name__ == "__main__":
    main()
