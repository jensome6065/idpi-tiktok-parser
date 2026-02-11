#!/usr/bin/env python3
import json
import re
import pandas as pd
from typing import Any, Dict, List, Tuple

METADATA_CSV = "metadata.csv"   # adjust path if needed
N_SAMPLE_ROWS = 50              # how many rows to test JSON shape on


def detect_json_columns(df: pd.DataFrame, sample_rows: int = 50) -> List[str]:
    """
    Heuristically find columns that look like JSON strings:
    - first non-null value starts with '{' or '['
    """
    json_cols: List[str] = []
    n = min(len(df), sample_rows)

    for col in df.columns:
        series = df[col].dropna().astype(str)
        if series.empty:
            continue
        # look at first few non-empty values
        for v in series.head(10):
            v_str = v.strip()
            if not v_str:
                continue
            if v_str[0] in "{[":
                json_cols.append(col)
            break  # only need first non-empty value
    return sorted(set(json_cols))


def try_parse_json(v: Any) -> Any:
    """
    Safely parse a value as JSON. Returns the parsed object or the original
    value if it doesn't parse cleanly.
    """
    if pd.isna(v):
        return None
    s = str(v).strip()
    if not s:
        return None
    # Only attempt JSON if it looks like an object/array
    if not (s.startswith("{") or s.startswith("[")):
        return v
    try:
        return json.loads(s)
    except Exception:
        return v  # leave as-is if broken


def coerce_bool(x: Any) -> Any:
    """
    Normalize various boolean-ish representations to True/False/None.
    Keeps non-bool / non-null values as-is so we don't lose information.
    """
    if pd.isna(x):
        return None
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
    return x


# -----------------------------
# AI-related pattern config
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

AI_VALUE_REGEX = re.compile(
    r"(aigc|ai[-\s]?generated|synthetic|c2pa|content credentials|provenance|watermark)",
    re.IGNORECASE,
)


def find_ai_kv_pairs(obj: Any, regex: re.Pattern, path: str = "") -> List[Tuple[str, Any]]:
    """
    Recursively find key paths in nested JSON whose *keys* match the regex.
    Returns list of (json_path, value).
    """
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
    """
    Recursively find key paths in nested JSON whose *values* (strings) match the regex.
    Returns list of (json_path, matched_string).
    """
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


def main():
    print(f"Loading {METADATA_CSV} ...")
    df = pd.read_csv(METADATA_CSV)

    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    json_cols = detect_json_columns(df)
    print("\nJSON-like columns detected:")
    for c in json_cols:
        print(f"  - {c}")

    # Test-parse a subset of rows for those columns
    sample = df.head(N_SAMPLE_ROWS).copy()

    parse_errors: Dict[str, int] = {c: 0 for c in json_cols}

    for c in json_cols:
        parsed_values = []
        for v in sample[c]:
            try:
                parsed_values.append(try_parse_json(v))
            except Exception:
                parse_errors[c] += 1
                parsed_values.append(v)
        sample[c] = parsed_values

    print("\nParse error counts (on first", N_SAMPLE_ROWS, "rows):")
    for c in json_cols:
        print(f"  {c}: {parse_errors[c]} errors")

    # Show a couple of examples for a few key JSON fields
    for c in ["video_volume_info", "challenges", "contents", "music"]:
        if c in sample.columns:
            print(f"\n=== Examples for column: {c} ===")
            for i, val in enumerate(sample[c].head(3)):
                print(f"Row {i}: type={type(val).__name__}")
                print(val)
                print("---")

    # -----------------------------
    # Discover AI-related fields
    # -----------------------------
    print("\n=== Discovering AI-related fields (names + JSON keys/values) ===")

    # 1) Column names that look AI-related
    ai_name_cols: List[str] = []
    for c in df.columns:
        name = str(c)
        name_lower = name.lower()
        if re.search(r"(aigc|ai[_-]?gc|ai[_-]?generated|c2pa|content[_-]?credential|synthetic)", name_lower):
            ai_name_cols.append(name)

    print("\nColumns whose NAMES look AI-related:")
    if ai_name_cols:
        for c in ai_name_cols:
            print(f"  - {c}")
    else:
        print("  (none found by name patterns)")

    # 2) JSON content scan for AI-ish keys and values
    N_SCAN_ROWS = min(len(df), 1000)
    scan_df = df.head(N_SCAN_ROWS)

    ai_key_counts_strict: Dict[str, int] = {}
    ai_key_counts_broad: Dict[str, int] = {}
    ai_value_counts: Dict[Tuple[str, str], int] = {}

    for _, row in scan_df.iterrows():
        for col in json_cols:
            obj = try_parse_json(row[col])
            if not isinstance(obj, (dict, list)):
                continue

            strict_hits = find_ai_kv_pairs(obj, AI_KEY_REGEX_STRICT)
            broad_hits = find_ai_kv_pairs(obj, AI_KEY_REGEX_BROAD)
            value_hits = find_ai_string_values(obj, AI_VALUE_REGEX)

            for path, _ in strict_hits:
                ai_key_counts_strict[path] = ai_key_counts_strict.get(path, 0) + 1
            for path, _ in broad_hits:
                ai_key_counts_broad[path] = ai_key_counts_broad.get(path, 0) + 1
            for path, val in value_hits:
                key = (path, str(val))
                ai_value_counts[key] = ai_value_counts.get(key, 0) + 1

    def _print_top_dict(title: str, d: Dict[Any, int], fmt) -> None:
        print(f"\n{title}")
        if not d:
            print("  (no matches found)")
            return
        for k, v in sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:30]:
            print(" ", fmt(k, v))

    _print_top_dict(
        "AI-related JSON KEYS (strict patterns, top 30):",
        ai_key_counts_strict,
        lambda k, v: f"{k}  -> {v} rows",
    )

    _print_top_dict(
        "AI-related JSON KEYS (broad patterns, top 30):",
        ai_key_counts_broad,
        lambda k, v: f"{k}  -> {v} rows",
    )

    _print_top_dict(
        "AI-related JSON STRING VALUES (top 30):",
        ai_value_counts,
        lambda k, v: f"path={k[0]!r}, value={k[1]!r}  -> {v} rows",
    )

    # -----------------------------
    # AI-specific metadata summary
    # -----------------------------
    print("\n=== AI-related metadata summary ===")

    if "video_is_ai_gc" in df.columns:
        df["video_is_ai_gc_norm"] = df["video_is_ai_gc"].apply(coerce_bool)
        ai_series = df["video_is_ai_gc_norm"]

        total = len(df)
        n_true = (ai_series == True).sum()
        n_false = (ai_series == False).sum()
        n_other = total - n_true - n_false

        pct = lambda n: (100.0 * n / total) if total else 0.0

        print(f"\nvideo_is_ai_gc present (n={total} rows)")
        print(f"  True : {n_true} ({pct(n_true):.2f}%)")
        print(f"  False: {n_false} ({pct(n_false):.2f}%)")
        print(f"  Other/NA: {n_other} ({pct(n_other):.2f}%)")

        # For rows TikTok marks as AI, inspect label fields
        ai_rows = df[ai_series == True].copy()
        if not ai_rows.empty:
            if "ai_gc_description" in ai_rows.columns:
                print("\nTop ai_gc_description values for AI-flagged videos:")
                print(
                    ai_rows["ai_gc_description"]
                    .fillna("NA")
                    .astype(str)
                    .value_counts()
                    .head(10)
                    .to_string()
                )

            if "ai_gc_label_type" in ai_rows.columns:
                print("\nTop ai_gc_label_type values for AI-flagged videos:")
                print(
                    ai_rows["ai_gc_label_type"]
                    .fillna("NA")
                    .astype(str)
                    .value_counts()
                    .head(10)
                    .to_string()
                )

            # Optional: basic engagement comparison
            stats_cols = [
                "stats_view_count",
                "stats_like_count",
                "stats_comment_count",
                "stats_share_count",
            ]
            present_stats = [c for c in stats_cols if c in df.columns]
            if present_stats:
                print("\nMean engagement stats by AI flag (only numeric rows):")
                # Coerce to numeric for summary
                tmp = df.copy()
                for c in present_stats:
                    tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
                grouped = tmp.groupby("video_is_ai_gc_norm")[present_stats].mean(numeric_only=True)
                print(grouped.to_string())
    else:
        print("Column 'video_is_ai_gc' not found; skipping AI flag analysis.")


if __name__ == "__main__":
    main()