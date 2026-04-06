"""
Run this on your existing tiktok_combined.csv (or any CSV that has at least
a `description` column and some of the scraper output columns).

What it adds
────────────
TEXT / HASHTAG SIGNALS
  hashtags                 semicolon-joined list of all hashtags in caption
  hashtag_ai_signal        bool — any hashtag matches AI keyword list
  hashtag_ai_matches       which AI hashtags were found (semicolon-joined)
  caption_ai_signal        bool — caption prose contains AI disclosure phrase
  caption_ai_match         the matched phrase

CONSOLIDATED SIGNAL COLUMNS  (for agreement matrix)
  signal_platform          True if TikTok JSON IsAigc=True OR badge present
  signal_creator_tag       True if hashtag_ai_signal OR caption_ai_signal
  signal_manual            normalised manual label (1/0/None)
  signal_visual_model      placeholder column (fill from your partners' output)

AGREEMENT / DIVERGENCE
  signals_agree_all        True if all three non-null signals agree
  signals_platform_vs_creator   True/False/NaN
  signals_platform_vs_manual    True/False/NaN
  signals_creator_vs_manual     True/False/NaN
  disclosure_gap           True = creator tagged AI but platform did NOT flag

TEMPORAL
  create_dt                parsed datetime from create_time (Unix seconds)
  era                      'pre_ai' / 'post_ai' based on configurable CUTOFF

ENGAGEMENT NORMALISATION
  engagement_total         sum of plays + likes + comments + shares
  like_rate                likes / plays  (proxy for positive reaction)
  comment_rate             comments / plays

Usage
─────
  python3 enrich_signals.py --input tiktok_combined.csv --output tiktok_enriched.csv

  # custom cutoff date (default: 2024-02-01, approximate Sora announcement)
  python3 enrich_signals.py --input tiktok_combined.csv --cutoff 2024-02-15

  # print a summary report after enriching
  python3 enrich_signals.py --input tiktok_combined.csv --report
"""

import argparse
import datetime
import re
import sys
from typing import Optional

import pandas as pd

# ─────────────────────────────────────────────
# CONFIGURATION — edit these lists as needed
# ─────────────────────────────────────────────

# Hashtags (without #) that signal AI content.  Case-insensitive.
AI_HASHTAGS: set[str] = {
    # generic AI labels
    "aigc", "aiart", "aigenerated", "aicreated", "aiartwork",
    "aianimation", "aivideo", "aifilm", "aiclip", "aicontentcreator",
    "generativeai", "syntheticmedia", "artificialintelligence",
    "aigenerated", "ai_generated", "ai_art",
    # tools / models — add or remove to match your research scope
    "soraai", "sora", "runwayml", "runway", "kling", "klingai",
    "pika", "pikaart", "pikalabs", "heygen", "heygenai",
    "midjourney", "stablediffusion", "dalle", "dalle3",
    "luma", "lumaai", "dreamina", "haiper",
    "invideo", "morphstudio", "genmo",
}

# Regex for prose AI disclosure in the caption / description
CAPTION_AI_REGEX = re.compile(
    r"""
    \b(
        ai[- ]?generated             # "AI generated", "AI-generated"
      | ai[- ]?created
      | ai[- ]?made
      | ai[- ]?art\b
      | ai[- ]?video
      | ai[- ]?animation
      | made\s+(with|by|using)\s+ai\b
      | created\s+(with|by|using)\s+ai\b
      | generated\s+by\s+ai\b
      | this\s+is\s+ai\b
      | not\s+real[,\.\s]
      | digitally\s+created
      | synthetic\s+media
      | aigc\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Default temporal cutoff: Sora public announcement / Feb 2024
DEFAULT_CUTOFF = datetime.datetime(2024, 2, 1, tzinfo=datetime.timezone.utc)


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def extract_hashtags(text: str) -> list[str]:
    """Return all hashtags (without #) found in text."""
    if not isinstance(text, str) or not text.strip():
        return []
    return re.findall(r"#(\w+)", text, re.UNICODE)


def hashtag_ai_analysis(text: str) -> tuple[bool, list[str]]:
    """Return (has_ai_hashtag, [matched hashtags])."""
    tags = extract_hashtags(text)
    matches = [t for t in tags if t.lower() in AI_HASHTAGS]
    return bool(matches), matches


def caption_ai_analysis(text: str) -> tuple[bool, str]:
    """Return (has_ai_prose_signal, first_matched_phrase)."""
    if not isinstance(text, str):
        return False, ""
    m = CAPTION_AI_REGEX.search(text)
    return (True, m.group(0).strip()) if m else (False, "")


def coerce_bool(x) -> Optional[bool]:
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
    return None


def normalize_manual_label(x) -> Optional[int]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip().upper()
    if s == "AI" or s == "1":
        return 1
    if s in ("NOT AI", "NOT_AI", "NON AI", "NON-AI", "NONAI", "NO", "0"):
        return 0
    return None


def signals_agree(a, b) -> Optional[bool]:
    """Return True/False if both non-null, else None."""
    if a is None or b is None or (isinstance(a, float) and pd.isna(a)) or (isinstance(b, float) and pd.isna(b)):
        return None
    return bool(a) == bool(b)


# ─────────────────────────────────────────────
# CORE ENRICHMENT
# ─────────────────────────────────────────────

def enrich(df: pd.DataFrame, cutoff: datetime.datetime) -> pd.DataFrame:
    df = df.copy()

    desc_col = "description"
    if desc_col not in df.columns:
        print(f"[WARN] No '{desc_col}' column found — hashtag/caption signals will be empty.")
        df[desc_col] = ""

    # ── TEXT / HASHTAG SIGNALS ──────────────────
    print("  → Extracting hashtags and caption signals...")

    ht_results = df[desc_col].apply(
        lambda t: hashtag_ai_analysis(t) if isinstance(t, str) else (False, [])
    )
    df["hashtags"] = df[desc_col].apply(
        lambda t: ";".join(extract_hashtags(t)) if isinstance(t, str) else ""
    )
    df["hashtag_ai_signal"] = ht_results.apply(lambda r: r[0])
    df["hashtag_ai_matches"] = ht_results.apply(lambda r: ";".join(r[1]))

    cap_results = df[desc_col].apply(
        lambda t: caption_ai_analysis(t) if isinstance(t, str) else (False, "")
    )
    df["caption_ai_signal"] = cap_results.apply(lambda r: r[0])
    df["caption_ai_match"] = cap_results.apply(lambda r: r[1])

    # ── CONSOLIDATED SIGNAL COLUMNS ─────────────
    print("  → Building consolidated signal columns...")

    # Platform signal: is_aigc JSON flag OR any AIGC badge
    def _platform_signal(row):
        aigc = coerce_bool(row.get("is_aigc"))
        badge = str(row.get("aigc_badge_type", "none")).strip().lower()
        tiktok_labeled = coerce_bool(row.get("tiktok_labeled_aigc"))
        if tiktok_labeled is True or aigc is True:
            return True
        if badge not in ("none", "", "nan"):
            return True
        return False

    df["signal_platform"] = df.apply(_platform_signal, axis=1)

    # Creator tag signal: hashtag OR caption prose
    df["signal_creator_tag"] = df["hashtag_ai_signal"] | df["caption_ai_signal"]

    # Manual label signal (normalised)
    manual_col = None
    for c in df.columns:
        uc = str(c).strip().upper()
        if uc in ("AI/NOT AI", "AI_NOT_AI", "LABEL", "AI", "MANUAL_AI"):
            manual_col = c
            break
    if manual_col:
        df["signal_manual"] = df[manual_col].apply(normalize_manual_label).astype("Int64")
    elif "manual_ai" in df.columns:
        df["signal_manual"] = df["manual_ai"].apply(normalize_manual_label).astype("Int64")
    else:
        print("  [WARN] No manual label column found — signal_manual will be null.")
        df["signal_manual"] = pd.NA

    # Visual model placeholder — your partners will fill this column
    if "signal_visual_model" not in df.columns:
        df["signal_visual_model"] = pd.NA
        print("  [NOTE] signal_visual_model column added as placeholder (fill with your partners' output).")

    # ── AGREEMENT / DIVERGENCE ───────────────────
    print("  → Computing signal agreement columns...")

    def _agree_row(row, col_a, col_b):
        a = row.get(col_a)
        b = row.get(col_b)
        # handle pandas NA / numpy nan
        try:
            if pd.isna(a) or pd.isna(b):
                return None
        except (TypeError, ValueError):
            pass
        return bool(a) == bool(b)

    df["signals_platform_vs_creator"] = df.apply(
        lambda r: _agree_row(r, "signal_platform", "signal_creator_tag"), axis=1
    )
    df["signals_platform_vs_manual"] = df.apply(
        lambda r: _agree_row(r, "signal_platform", "signal_manual"), axis=1
    )
    df["signals_creator_vs_manual"] = df.apply(
        lambda r: _agree_row(r, "signal_creator_tag", "signal_manual"), axis=1
    )

    def _all_agree(row):
        vals = []
        for col in ("signal_platform", "signal_creator_tag", "signal_manual"):
            v = row.get(col)
            try:
                if pd.isna(v):
                    return None
            except (TypeError, ValueError):
                pass
            vals.append(bool(v))
        if len(vals) < 3:
            return None
        return len(set(vals)) == 1

    df["signals_agree_all"] = df.apply(_all_agree, axis=1)

    # Disclosure gap: creator flagged AI but platform did NOT
    df["disclosure_gap"] = (
        df["signal_creator_tag"].astype(bool) & ~df["signal_platform"].astype(bool)
    )

    # ── TEMPORAL ────────────────────────────────
    print("  → Parsing timestamps and assigning era...")

    if "create_time" in df.columns:
        # try numeric (Unix seconds) first, then string parsing
        def _parse_dt(x):
            if pd.isna(x):
                return pd.NaT
            try:
                ts = float(x)
                return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
            except (ValueError, TypeError, OSError):
                pass
            try:
                return pd.to_datetime(x, utc=True)
            except Exception:
                return pd.NaT

        df["create_dt"] = df["create_time"].apply(_parse_dt)
        cutoff_aware = cutoff if cutoff.tzinfo else cutoff.replace(tzinfo=datetime.timezone.utc)
        df["era"] = df["create_dt"].apply(
            lambda d: "post_ai" if pd.notna(d) and d >= cutoff_aware else "pre_ai"
        )
    else:
        print("  [WARN] No 'create_time' column found — create_dt and era will be null.")
        df["create_dt"] = pd.NaT
        df["era"] = "unknown"

    # ── ENGAGEMENT NORMALISATION ─────────────────
    print("  → Computing engagement normalisation...")

    for col in ("play_count", "like_count", "comment_count", "share_count"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    plays_col = "play_count" if "play_count" in df.columns else None

    engagement_cols = [c for c in ("play_count", "like_count", "comment_count", "share_count") if c in df.columns]
    if engagement_cols:
        df["engagement_total"] = df[engagement_cols].sum(axis=1, min_count=1)
    else:
        df["engagement_total"] = pd.NA

    if plays_col:
        df["like_rate"] = (df["like_count"] / df["play_count"].replace(0, pd.NA)).round(6) if "like_count" in df.columns else pd.NA
        df["comment_rate"] = (df["comment_count"] / df["play_count"].replace(0, pd.NA)).round(6) if "comment_count" in df.columns else pd.NA
    else:
        df["like_rate"] = pd.NA
        df["comment_rate"] = pd.NA

    return df


# ─────────────────────────────────────────────
# SUMMARY REPORT
# ─────────────────────────────────────────────

def print_report(df: pd.DataFrame):
    total = len(df)
    print("\n" + "═" * 60)
    print("ENRICHMENT SUMMARY REPORT")
    print("═" * 60)
    print(f"Total videos: {total}")

    print("\n── Signal prevalence ──────────────────────────────────────")
    for col in ("signal_platform", "signal_creator_tag", "signal_manual"):
        if col in df.columns:
            n = df[col].sum(skipna=True)
            try:
                pct = 100.0 * int(n) / total
                print(f"  {col:<30} {int(n):>5} / {total}  ({pct:.1f}%)")
            except Exception:
                print(f"  {col:<30} (could not compute)")

    print("\n── Signal agreement rates (pairwise) ──────────────────────")
    for col in ("signals_platform_vs_creator", "signals_platform_vs_manual", "signals_creator_vs_manual"):
        if col in df.columns:
            series = df[col].dropna()
            if len(series) == 0:
                print(f"  {col:<40} N/A (all null)")
                continue
            rate = series.mean()
            print(f"  {col:<40} {rate:.1%}  (n={len(series)})")

    print("\n── Disclosure gap (creator tagged AI, platform did NOT) ────")
    if "disclosure_gap" in df.columns:
        n_gap = df["disclosure_gap"].sum()
        print(f"  {n_gap} videos  ({100.0 * n_gap / total:.1f}%)")

    print("\n── Top AI hashtags used ────────────────────────────────────")
    if "hashtag_ai_matches" in df.columns:
        from collections import Counter
        all_matches = []
        for cell in df["hashtag_ai_matches"].dropna():
            all_matches.extend([h for h in str(cell).split(";") if h])
        top = Counter(all_matches).most_common(15)
        if top:
            for tag, count in top:
                print(f"  #{tag:<30} {count}")
        else:
            print("  (none found)")

    print("\n── Era breakdown ───────────────────────────────────────────")
    if "era" in df.columns:
        print(df["era"].value_counts(dropna=False).to_string())

    print("\n── Median engagement by platform signal ────────────────────")
    eng_cols = [c for c in ("play_count", "like_count", "comment_count", "share_count") if c in df.columns]
    if eng_cols and "signal_platform" in df.columns:
        grouped = df.groupby("signal_platform", dropna=False)[eng_cols].median(numeric_only=True)
        print(grouped.to_string())

    print("\n── Median engagement by era ────────────────────────────────")
    if eng_cols and "era" in df.columns:
        grouped = df.groupby("era", dropna=False)[eng_cols].median(numeric_only=True)
        print(grouped.to_string())

    print("\n" + "═" * 60)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Enrich tiktok_combined.csv with hashtag/caption signals, "
                    "multi-signal agreement columns, and temporal era fields."
    )
    ap.add_argument("--input", default="tiktok_combined.csv", help="Input CSV (default: tiktok_combined.csv)")
    ap.add_argument("--output", default="tiktok_enriched.csv", help="Output CSV (default: tiktok_enriched.csv)")
    ap.add_argument(
        "--cutoff",
        default="2024-02-01",
        help="Pre/post AI era cutoff date YYYY-MM-DD (default: 2024-02-01)",
    )
    ap.add_argument("--report", action="store_true", help="Print a summary report after enriching")
    args = ap.parse_args()

    try:
        cutoff_dt = datetime.datetime.strptime(args.cutoff, "%Y-%m-%d").replace(
            tzinfo=datetime.timezone.utc
        )
    except ValueError:
        print(f"[ERROR] Could not parse cutoff date '{args.cutoff}'. Use YYYY-MM-DD format.")
        sys.exit(1)

    print(f"Loading {args.input} ...")
    try:
        df = pd.read_csv(args.input, low_memory=False)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {args.input}")
        sys.exit(1)

    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")
    print(f"Era cutoff: {args.cutoff}\n")

    print("Enriching...")
    enriched = enrich(df, cutoff=cutoff_dt)

    enriched.to_csv(args.output, index=False)
    print(f"\n✓ Wrote {len(enriched)} rows → {args.output}")

    new_cols = [c for c in enriched.columns if c not in df.columns]
    print(f"  New columns added ({len(new_cols)}): {', '.join(new_cols)}")

    if args.report:
        print_report(enriched)


if __name__ == "__main__":
    main()