#!/usr/bin/env python3
"""
tiktok_parser_v2_3.py

What it does
- Reads a CSV exported from Google Sheets with a required column: LINK
- Optional manual label column: AI/NOT AI, AI_NOT_AI, LABEL, AI (values like "AI", "NOT AI")
- Uses Selenium to load each TikTok URL and pulls:
  - embedded hydration JSON (SIGI_STATE / __UNIVERSAL_DATA_FOR_REHYDRATION__ / __NEXT_DATA__)
  - post metadata + stats
  - TikTok JSON flag: IsAigc / isAigc (coerced from bool/0/1/"true"/"false")
  - AI key-path matches (your existing approach)
  - NEW: AI text detection in DOM-visible text + HTML snippet around keywords
  - NEW: optionally dumps full HTML per video to ./html_dumps/<video_id>.html

Outputs
1) tiktok_parsed.csv   (scrape output only)
2) tiktok_combined.csv (manual sheet + scrape output + eval fields)

Run (recommended debug)
  python3 tiktok_parser.py --sheet tiktok_database.csv --headless 0 --sleep 8 --limit 10 --dump-html 1

Notes
- The DOM badge/snippet is for finding on-page "AI-generated" (if it exists in web UI).
- Your personal ",AI" tagging in the URL query is preserved in `input_url_full` and `url_query_web_id`.
"""

import argparse
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit, parse_qs

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options


print("RUNNING tiktok_parser_v2_3.py (IsAigc coercion + DOM AI snippet + optional HTML dumps)")


# -----------------------------
# AI-flag detection config
# -----------------------------
AI_KEY_PATTERNS_STRICT = [
    r"\baigc\b",
    r"\bis_aigc\b",
    r"\bisaigc\b",
    r"\bai[_-]?generated\b",
    r"\bcontentcredentials\b",
    r"\bcontent[_-]?credentials\b",
    r"\bprovenance\b",
]

AI_KEY_PATTERNS_BROAD = AI_KEY_PATTERNS_STRICT + [
    r"\bsynthetic\b",
    r"\bwatermark\b",
    r"\bcredential\b",
    r"\bc2pa\b",
    r"\bai[_-]?label\b",
    r"\bcontent[_-]?label(s)?\b",
    r"\bbadges?\b",
    r"\blabels?\b",
    r"\bsynthetic[_-]?media\b",
]

AI_KEY_REGEX_STRICT = re.compile("|".join(AI_KEY_PATTERNS_STRICT), re.IGNORECASE)
AI_KEY_REGEX_BROAD = re.compile("|".join(AI_KEY_PATTERNS_BROAD), re.IGNORECASE)

# NEW: search values (strings) too
AI_VALUE_REGEX = re.compile(
    r"(aigc|ai[-\s]?generated|synthetic|c2pa|content credentials|provenance|watermark)",
    re.IGNORECASE,
)

# NEW: DOM badge keywords (what you want snippet around)
DOM_NEEDLES = [
    "ai-generated",
    "ai generated",
    "aigc",
    "content credentials",
    "c2pa",
    "provenance",
    "synthetic media",
]

SCRIPT_ID_CANDIDATES = [
    "SIGI_STATE",
    "__UNIVERSAL_DATA_FOR_REHYDRATION__",
    "__NEXT_DATA__",
]


# -----------------------------
# Helpers
# -----------------------------
def canonical_url(url: str) -> str:
    """Return canonical TikTok URL (drops querystring/fragment)."""
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def parse_query(url: str) -> Dict[str, Any]:
    parts = urlsplit(url)
    q = parse_qs(parts.query)
    # parse_qs returns lists for values; flatten singletons
    out = {}
    for k, v in q.items():
        if len(v) == 1:
            out[k] = v[0]
        else:
            out[k] = v
    return out


def make_json_safe(x: Any) -> Any:
    try:
        json.dumps(x)
        return x
    except TypeError:
        return str(x)


def normalize_manual_label(x: Any) -> Optional[int]:
    """Map your sheet labels to 1/0. Returns None if unknown."""
    if x is None:
        return None
    s = str(x).strip().upper()
    if s == "AI":
        return 1
    if s in ("NOT AI", "NOT_AI", "NON AI", "NON-AI", "NONAI", "NO", "0"):
        return 0
    return None


def pick_label_column(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns)
    for c in cols:
        uc = str(c).strip().upper()
        if uc in ("AI/NOT AI", "AI_NOT_AI", "LABEL", "AI"):
            return c
    for c in cols:
        uc = str(c).strip().upper()
        if "AI" in uc and ("NOT" in uc or "/" in uc or "_" in uc):
            return c
    return None


def safe_get(d: Any, *keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def pick_first_non_null(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


def to_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
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
        if s in ("false", "0", "no", "n"):
            return False
    return None


def snippet_around(html: str, needle: str, radius: int = 800) -> str:
    low = html.lower()
    nlow = needle.lower()
    i = low.find(nlow)
    if i == -1:
        return ""
    start = max(0, i - radius)
    end = min(len(html), i + len(needle) + radius)
    return html[start:end]


def best_dom_snippet(html: str, needles: List[str], radius: int = 800) -> Tuple[bool, str, str]:
    """
    Returns:
      (has_any, matched_needle, snippet)
    """
    for needle in needles:
        snip = snippet_around(html, needle, radius=radius)
        if snip:
            return True, needle, snip
    return False, "", ""


def save_html_dump(video_id: str, html: str, out_dir: str = "html_dumps") -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{video_id}.html" if video_id else f"unknown_{int(time.time())}.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


# -----------------------------
# Selenium setup
# -----------------------------
def make_driver(headless: bool = True) -> webdriver.Chrome:
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1400,900")
    opts.add_argument("--lang=en-US")

    # basic automation hygiene (not a bypass; just reduces noise)
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)

    driver = webdriver.Chrome(options=opts)
    driver.set_page_load_timeout(45)
    return driver


# -----------------------------
# JSON extraction from HTML
# -----------------------------
def extract_embedded_json_from_html(html: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    for sid in SCRIPT_ID_CANDIDATES:
        m = re.search(
            rf'<script[^>]+id="{re.escape(sid)}"[^>]*>(.*?)</script>',
            html,
            re.DOTALL | re.IGNORECASE,
        )
        if m:
            blob = m.group(1).strip()
            blob = blob.replace("&quot;", '"').replace("&amp;", "&")
            try:
                return json.loads(blob), sid
            except json.JSONDecodeError:
                pass

    assign_patterns = [
        r"SIGI_STATE\s*=\s*({.*?})\s*;\s*</script>",
        r"__UNIVERSAL_DATA_FOR_REHYDRATION__\s*=\s*({.*?})\s*;\s*</script>",
    ]
    for pat in assign_patterns:
        m = re.search(pat, html, re.DOTALL | re.IGNORECASE)
        if m:
            blob = m.group(1).strip()
            try:
                return json.loads(blob), "assignment_fallback"
            except json.JSONDecodeError:
                continue

    return None, None


# -----------------------------
# Recursive search for AI-ish keys + NEW value search
# -----------------------------
def find_ai_kv_pairs(obj: Any, regex: re.Pattern, path: str = "") -> List[Tuple[str, Any]]:
    hits: List[Tuple[str, Any]] = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            k_str = str(k)
            new_path = f"{path}.{k_str}" if path else k_str

            if regex.search(k_str):
                hits.append((new_path, v))

            hits.extend(find_ai_kv_pairs(v, regex, new_path))

    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits.extend(find_ai_kv_pairs(v, regex, f"{path}[{i}]"))

    return hits


def find_ai_string_values(obj: Any, regex: re.Pattern, path: str = "") -> List[Tuple[str, str]]:
    hits: List[Tuple[str, str]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            hits.extend(find_ai_string_values(v, regex, p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits.extend(find_ai_string_values(v, regex, f"{path}[{i}]"))
    elif isinstance(obj, str):
        if regex.search(obj):
            hits.append((path, obj))
    return hits


# -----------------------------
# Extract IsAigc boolean value (coerced)
# -----------------------------
def extract_is_aigc_value(data: Dict[str, Any]) -> Optional[bool]:
    candidates = [
        safe_get(data, "__DEFAULT_SCOPE__", "webapp.video-detail", "itemInfo", "itemStruct", "IsAigc", default=None),
        safe_get(data, "__DEFAULT_SCOPE__", "webapp.video-detail", "itemInfo", "itemStruct", "isAigc", default=None),
        safe_get(data, "__DEFAULT_SCOPE__", "webapp.video-detail", "itemInfo", "itemStruct", "IsAIGC", default=None),
        safe_get(data, "__DEFAULT_SCOPE__", "webapp.video-detail", "itemInfo", "itemStruct", "isAIGC", default=None),
    ]
    for v in candidates:
        b = coerce_bool(v)
        if b is not None:
            return b

    target_keys = {"isaigc", "is_aigc", "is-aigc", "isaigcflag", "isaigccontent", "aigc", "aigclabel"}

    def walk(obj: Any) -> Optional[bool]:
        if isinstance(obj, dict):
            for k, vv in obj.items():
                if str(k).strip().lower() in target_keys:
                    b = coerce_bool(vv)
                    if b is not None:
                        return b
                got = walk(vv)
                if got is not None:
                    return got
        elif isinstance(obj, list):
            for vv in obj:
                got = walk(vv)
                if got is not None:
                    return got
        return None

    return walk(data)


# -----------------------------
# Pull post + stats from common TikTok state shapes
# -----------------------------
def extract_post_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    item_module = safe_get(data, "ItemModule", default=None)
    video_obj = None
    video_id = None

    if isinstance(item_module, dict) and item_module:
        video_id = next(iter(item_module.keys()))
        video_obj = item_module.get(video_id)

    default_scope = safe_get(data, "__DEFAULT_SCOPE__", default=None)
    video_detail = None
    if isinstance(default_scope, dict):
        video_detail = safe_get(default_scope, "webapp.video-detail", default=None)

    author = None
    desc = None
    create_time = None
    stats: Dict[str, Any] = {}

    if isinstance(video_obj, dict):
        author = pick_first_non_null(video_obj.get("author"), video_obj.get("authorUniqueId"))
        desc = pick_first_non_null(video_obj.get("desc"), video_obj.get("description"))
        create_time = video_obj.get("createTime")
        stats = video_obj.get("stats") or {}

    if isinstance(video_detail, dict):
        item_struct = safe_get(video_detail, "itemInfo", "itemStruct", default=None)

        if isinstance(item_struct, dict):
            video_id = pick_first_non_null(video_id, item_struct.get("id"))
            author = pick_first_non_null(author, safe_get(item_struct, "author", "uniqueId", default=None))
            desc = pick_first_non_null(desc, item_struct.get("desc"))
            create_time = pick_first_non_null(create_time, item_struct.get("createTime"))

            stats2 = item_struct.get("stats")
            if isinstance(stats2, dict) and stats2:
                stats = stats2

        item_info_struct = safe_get(video_detail, "itemInfo", "itemStruct", "itemInfo", "itemStruct", default=None)
        if isinstance(item_info_struct, dict):
            stats3 = item_info_struct.get("stats")
            if isinstance(stats3, dict) and stats3:
                stats = stats3

    play_count = pick_first_non_null(stats.get("playCount"), stats.get("viewCount"), stats.get("views"))
    like_count = pick_first_non_null(stats.get("diggCount"), stats.get("likeCount"), stats.get("likes"))
    comment_count = pick_first_non_null(stats.get("commentCount"), stats.get("comments"))
    share_count = pick_first_non_null(stats.get("shareCount"), stats.get("shares"))

    return {
        "video_id": video_id,
        "author": author,
        "description": desc,
        "create_time": str(create_time) if create_time is not None else None,
        "play_count": to_int(play_count),
        "like_count": to_int(like_count),
        "comment_count": to_int(comment_count),
        "share_count": to_int(share_count),
    }


# -----------------------------
# Scrape one URL
# -----------------------------
@dataclass
class ExtractResult:
    # input + url fields
    input_url_full: str
    url: str  # canonical
    url_query_is_from_webapp: Optional[str]
    url_query_web_id: Optional[str]

    status: str
    raw_source_used: Optional[str]

    video_id: Optional[str]
    author: Optional[str]
    create_time: Optional[str]
    description: Optional[str]

    play_count: Optional[int]
    like_count: Optional[int]
    comment_count: Optional[int]
    share_count: Optional[int]

    is_aigc: Optional[bool]  # TikTok JSON flag

    ai_keys_strict: str
    ai_keys_broad: str
    ai_pairs_strict_json: str
    ai_pairs_broad_json: str

    # NEW: value hits + DOM badge/snippet
    ai_value_hits_json: str
    dom_has_ai_text: bool
    dom_matched_needle: str
    dom_ai_snippet: str

    # NEW: optional HTML dump path
    html_dump_path: str


def scrape_one(
    driver: webdriver.Chrome,
    input_url_full: str,
    sleep_s: float,
    dump_html: bool,
) -> ExtractResult:
    q = parse_query(input_url_full)
    canon = canonical_url(input_url_full)

    try:
        driver.get(input_url_full)
        time.sleep(sleep_s)

        html = driver.page_source

        # NEW: DOM detection + snippet
        dom_has, dom_needle, dom_snip = best_dom_snippet(html, DOM_NEEDLES, radius=900)
        # cap snippet to keep CSV sane
        if len(dom_snip) > 4000:
            dom_snip = dom_snip[:4000]

        data, source = extract_embedded_json_from_html(html)
        if not data:
            dump_path = ""
            if dump_html:
                dump_path = save_html_dump(video_id="", html=html)
            return ExtractResult(
                input_url_full=input_url_full,
                url=canon,
                url_query_is_from_webapp=q.get("is_from_webapp"),
                url_query_web_id=q.get("web_id"),
                status="no_embedded_json_found",
                raw_source_used=None,
                video_id=None,
                author=None,
                create_time=None,
                description=None,
                play_count=None,
                like_count=None,
                comment_count=None,
                share_count=None,
                is_aigc=None,
                ai_keys_strict="",
                ai_keys_broad="",
                ai_pairs_strict_json="[]",
                ai_pairs_broad_json="[]",
                ai_value_hits_json="[]",
                dom_has_ai_text=bool(dom_has),
                dom_matched_needle=dom_needle,
                dom_ai_snippet=dom_snip,
                html_dump_path=dump_path,
            )

        post = extract_post_fields(data)
        is_aigc_val = extract_is_aigc_value(data)

        strict_hits = find_ai_kv_pairs(data, AI_KEY_REGEX_STRICT)
        broad_hits = find_ai_kv_pairs(data, AI_KEY_REGEX_BROAD)

        strict_keys = sorted({p for (p, _) in strict_hits})
        broad_keys = sorted({p for (p, _) in broad_hits})

        strict_pairs = [{"path": p, "value": make_json_safe(v)} for (p, v) in strict_hits][:200]
        broad_pairs = [{"path": p, "value": make_json_safe(v)} for (p, v) in broad_hits][:200]

        # NEW: value-based string hits anywhere in JSON
        value_hits = find_ai_string_values(data, AI_VALUE_REGEX)
        value_pairs = [{"path": p, "value": v} for (p, v) in value_hits][:200]

        dump_path = ""
        if dump_html:
            dump_path = save_html_dump(video_id=post.get("video_id") or "", html=html)

        return ExtractResult(
            input_url_full=input_url_full,
            url=canon,
            url_query_is_from_webapp=q.get("is_from_webapp"),
            url_query_web_id=q.get("web_id"),
            status="ok",
            raw_source_used=source,
            video_id=post["video_id"],
            author=post["author"],
            create_time=post["create_time"],
            description=post["description"],
            play_count=post["play_count"],
            like_count=post["like_count"],
            comment_count=post["comment_count"],
            share_count=post["share_count"],
            is_aigc=is_aigc_val,
            ai_keys_strict=";".join(strict_keys),
            ai_keys_broad=";".join(broad_keys),
            ai_pairs_strict_json=json.dumps(strict_pairs, ensure_ascii=False),
            ai_pairs_broad_json=json.dumps(broad_pairs, ensure_ascii=False),
            ai_value_hits_json=json.dumps(value_pairs, ensure_ascii=False),
            dom_has_ai_text=bool(dom_has),
            dom_matched_needle=dom_needle,
            dom_ai_snippet=dom_snip,
            html_dump_path=dump_path,
        )

    except Exception as e:
        return ExtractResult(
            input_url_full=input_url_full,
            url=canon,
            url_query_is_from_webapp=q.get("is_from_webapp"),
            url_query_web_id=q.get("web_id"),
            status=f"error:{type(e).__name__}:{str(e)[:120]}",
            raw_source_used=None,
            video_id=None,
            author=None,
            create_time=None,
            description=None,
            play_count=None,
            like_count=None,
            comment_count=None,
            share_count=None,
            is_aigc=None,
            ai_keys_strict="",
            ai_keys_broad="",
            ai_pairs_strict_json="[]",
            ai_pairs_broad_json="[]",
            ai_value_hits_json="[]",
            dom_has_ai_text=False,
            dom_matched_needle="",
            dom_ai_snippet="",
            html_dump_path="",
        )


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", required=True, help="Path to exported Google Sheet CSV (must include LINK column)")
    ap.add_argument("--out-parsed", default="tiktok_parsed.csv")
    ap.add_argument("--out-combined", default="tiktok_combined.csv")
    ap.add_argument("--headless", type=int, default=0, help="1=headless, 0=visible browser")
    ap.add_argument("--sleep", type=float, default=8.0, help="Seconds to wait after page load")
    ap.add_argument("--limit", type=int, default=20, help="Max URLs to scrape (set -1 for all)")
    ap.add_argument("--dump-html", type=int, default=0, help="1=dump full HTML files into ./html_dumps")
    args = ap.parse_args()

    manual = pd.read_csv(args.sheet)

    if "LINK" not in manual.columns:
        raise ValueError(f"Could not find column 'LINK'. Columns found: {list(manual.columns)}")

    label_col = pick_label_column(manual)
    if label_col is None:
        print("[WARN] Could not find a label column like 'AI/NOT AI'. Running without manual labels/eval.")

    urls = manual["LINK"].dropna().astype(str).tolist()
    if args.limit is not None and args.limit >= 0:
        urls = urls[: args.limit]

    driver = make_driver(headless=bool(args.headless))
    results: List[ExtractResult] = []
    try:
        for i, u in enumerate(urls, start=1):
            print(f"[{i}/{len(urls)}] scraping: {u}")
            results.append(scrape_one(driver, u, sleep_s=args.sleep, dump_html=bool(args.dump_html)))
    finally:
        driver.quit()

    # Write parsed output
    with open(args.out_parsed, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "input_url_full",
                "url",  # canonical
                "url_query_is_from_webapp",
                "url_query_web_id",
                "status",
                "raw_source_used",
                "video_id",
                "author",
                "create_time",
                "description",
                "play_count",
                "like_count",
                "comment_count",
                "share_count",
                "is_aigc",
                "ai_keys_strict",
                "ai_keys_broad",
                "ai_pairs_strict_json",
                "ai_pairs_broad_json",
                "ai_value_hits_json",
                "dom_has_ai_text",
                "dom_matched_needle",
                "dom_ai_snippet",
                "html_dump_path",
            ],
        )
        w.writeheader()
        for r in results:
            w.writerow(
                {
                    "input_url_full": r.input_url_full,
                    "url": r.url,
                    "url_query_is_from_webapp": r.url_query_is_from_webapp,
                    "url_query_web_id": r.url_query_web_id,
                    "status": r.status,
                    "raw_source_used": r.raw_source_used,
                    "video_id": r.video_id,
                    "author": r.author,
                    "create_time": r.create_time,
                    "description": r.description,
                    "play_count": r.play_count,
                    "like_count": r.like_count,
                    "comment_count": r.comment_count,
                    "share_count": r.share_count,
                    "is_aigc": r.is_aigc,
                    "ai_keys_strict": r.ai_keys_strict,
                    "ai_keys_broad": r.ai_keys_broad,
                    "ai_pairs_strict_json": r.ai_pairs_strict_json,
                    "ai_pairs_broad_json": r.ai_pairs_broad_json,
                    "ai_value_hits_json": r.ai_value_hits_json,
                    "dom_has_ai_text": r.dom_has_ai_text,
                    "dom_matched_needle": r.dom_matched_needle,
                    "dom_ai_snippet": r.dom_ai_snippet,
                    "html_dump_path": r.html_dump_path,
                }
            )

    parsed_df = pd.read_csv(args.out_parsed)
    combined = manual.merge(parsed_df, left_on="LINK", right_on="input_url_full", how="left")

    if label_col is not None:
        combined["manual_ai"] = combined[label_col].apply(normalize_manual_label)
    else:
        combined["manual_ai"] = None

    # is_aigc column might be strings depending on pandas; normalize to True/False/None
    if "is_aigc" in combined.columns:
        combined["is_aigc"] = combined["is_aigc"].apply(coerce_bool)
    else:
        combined["is_aigc"] = None

    # Badge subtype: distinguish "Creator labeled as AI-generated" vs "Contains AI-generated media"
    if "dom_ai_snippet" in combined.columns:
        def _classify_aigc_badge(snippet: Any) -> str:
            if pd.isna(snippet):
                return "none"
            s = str(snippet).lower()
            if "creator labeled as ai-generated" in s:
                return "creator_labeled"
            # be a bit forgiving about truncation/variants
            if (
                "contains ai-generated media" in s
                or "contains ai-gen" in s
                or "contains ai generated" in s
            ):
                return "contains_ai"
            return "none"

        combined["aigc_badge_type"] = combined["dom_ai_snippet"].apply(_classify_aigc_badge)
        combined["aigc_creator_labeled"] = combined["aigc_badge_type"] == "creator_labeled"
        combined["aigc_contains_ai_media"] = combined["aigc_badge_type"] == "contains_ai"
    else:
        combined["aigc_badge_type"] = "none"
        combined["aigc_creator_labeled"] = False
        combined["aigc_contains_ai_media"] = False

    # Platform-level AI label: either explicit JSON IsAigc True OR an AIGC badge is present
    combined["tiktok_labeled_aigc"] = combined.apply(
        lambda r: True
        if (r.get("is_aigc") is True)
        else bool(r.get("aigc_badge_type") in ("creator_labeled", "contains_ai")),
        axis=1,
    )

    combined.to_csv(args.out_combined, index=False)

    print("\n=== Scrape status counts (only scraped rows) ===")
    if "status" in parsed_df.columns:
        print(parsed_df["status"].value_counts(dropna=False).to_string())
    else:
        print("No status column found in parsed output (unexpected).")

    print(f"\nWrote: {args.out_parsed}")
    print(f"Wrote: {args.out_combined}")
    if bool(args.dump_html):
        print("HTML dumps (if any): ./html_dumps/")


if __name__ == "__main__":
    main()