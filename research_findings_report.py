import argparse
import os
from math import sqrt
from typing import Dict, List

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _as_bool(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    margin = (z / denom) * sqrt((p * (1 - p) / n) + ((z**2) / (4 * (n**2))))
    return (max(0.0, center - margin), min(1.0, center + margin))


def build_core_tables(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    total = len(df)
    raw_platform = _as_bool(df["is_aigc"]) if "is_aigc" in df.columns else pd.Series([False] * total)

    prevalence = pd.DataFrame(
        [
            {
                "signal": "platform_raw_is_aigc",
                "count_true": int(raw_platform.sum()),
                "total": total,
            },
            {
                "signal": "signal_platform",
                "count_true": int(_as_bool(df["signal_platform"]).sum()),
                "total": total,
            },
            {
                "signal": "signal_creator_tag",
                "count_true": int(_as_bool(df["signal_creator_tag"]).sum()),
                "total": total,
            },
            {
                "signal": "disclosure_gap",
                "count_true": int(_as_bool(df["disclosure_gap"]).sum()),
                "total": total,
            },
            {
                "signal": "signal_potential_text",
                "count_true": int(_as_bool(df.get("signal_potential_text", pd.Series([False] * total))).sum()),
                "total": total,
            },
            {
                "signal": "signal_potential_ai_any",
                "count_true": int(_as_bool(df.get("signal_potential_ai_any", pd.Series([False] * total))).sum()),
                "total": total,
            },
        ]
    )
    prevalence["pct_true"] = (100.0 * prevalence["count_true"] / prevalence["total"]).round(2)
    ci_bounds = prevalence.apply(
        lambda r: wilson_ci(int(r["count_true"]), int(r["total"])),
        axis=1,
    )
    prevalence["ci95_low_pct"] = ci_bounds.apply(lambda b: round(100.0 * b[0], 2))
    prevalence["ci95_high_pct"] = ci_bounds.apply(lambda b: round(100.0 * b[1], 2))

    agreement_rows: List[Dict[str, object]] = []
    for col in [
        "signals_platform_vs_creator",
        "signals_platform_vs_manual",
        "signals_creator_vs_manual",
        "signals_agree_all",
    ]:
        valid = df[col].dropna()
        rate = float(valid.mean()) if len(valid) else None
        agreement_rows.append({"comparison": col, "agreement_rate": rate, "n_non_null": int(len(valid))})
    agreement = pd.DataFrame(agreement_rows)
    agreement["agreement_rate_pct"] = agreement["agreement_rate"].apply(
        lambda x: round(x * 100, 2) if x is not None else None
    )
    # Positive agreement avoids inflated agreement under heavy class imbalance.
    platform = _as_bool(df["signal_platform"])
    creator = _as_bool(df["signal_creator_tag"])
    both_pos = int((platform & creator).sum())
    either_pos = int((platform | creator).sum())
    pos_agreement = (both_pos / either_pos) if either_pos else None
    agreement.loc[len(agreement)] = {
        "comparison": "positive_agreement_platform_creator",
        "agreement_rate": pos_agreement,
        "n_non_null": either_pos,
        "agreement_rate_pct": round(pos_agreement * 100, 2) if pos_agreement is not None else None,
    }

    engagement_cols = ["play_count", "like_count", "comment_count", "share_count", "engagement_total", "like_rate", "comment_rate"]
    engagement_by_platform = (
        df.groupby("signal_platform", dropna=False)[engagement_cols]
        .median(numeric_only=True)
        .reset_index()
    )
    platform_counts = (
        df.groupby("signal_platform", dropna=False)
        .size()
        .reset_index(name="n_videos")
    )
    engagement_by_creator = (
        df.groupby("signal_creator_tag", dropna=False)[engagement_cols]
        .median(numeric_only=True)
        .reset_index()
    )
    engagement_by_era = (
        df.groupby("era", dropna=False)[engagement_cols]
        .median(numeric_only=True)
        .reset_index()
    )

    era_by_platform = (
        df.groupby(["era", "signal_platform"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    era_by_platform["pct_within_era"] = (
        era_by_platform["count"]
        / era_by_platform.groupby("era")["count"].transform("sum")
        * 100.0
    ).round(2)

    desc = df["description"].fillna("").astype(str) if "description" in df.columns else pd.Series([], dtype=str)
    diag_rows = []
    diagnostic_patterns = {
        "contains_ai_token": r"(?:\bai\b|aigc|ai-generated|ai generated|artificial intelligence|synthetic)",
        "contains_model_names": r"(?:sora|midjourney|dalle|stable diffusion|runway|kling|pika|heygen|luma)",
        "contains_hashtag_ai": r"(?:#(?:ai|aigc|aiart|generativeai|aivideo|aigenerated)\b)",
    }
    for name, pat in diagnostic_patterns.items():
        n = int(desc.str.contains(pat, case=False, regex=True).sum()) if len(desc) else 0
        diag_rows.append({"diagnostic": name, "count": n, "total": total, "pct": round(100.0 * n / total, 4) if total else 0.0})
    diagnostics = pd.DataFrame(diag_rows)

    return {
        "prevalence": prevalence,
        "agreement": agreement,
        "engagement_by_platform": engagement_by_platform,
        "platform_counts": platform_counts,
        "engagement_by_creator": engagement_by_creator,
        "engagement_by_era": engagement_by_era,
        "era_by_platform": era_by_platform,
        "diagnostics": diagnostics,
    }


def save_figures(tables: Dict[str, pd.DataFrame], out_dir: str) -> None:
    prev = tables["prevalence"]
    plt.figure(figsize=(7, 4))
    plt.bar(prev["signal"], prev["pct_true"])
    plt.ylabel("% of videos")
    plt.title("AI Signal Prevalence")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_prevalence.png"), dpi=180)
    plt.close()

    agree = tables["agreement"]
    plt.figure(figsize=(8, 4))
    plt.bar(agree["comparison"], agree["agreement_rate_pct"])
    plt.ylabel("Agreement rate (%)")
    plt.title("Signal Agreement Rates")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_agreement.png"), dpi=180)
    plt.close()

    era = tables["era_by_platform"]
    pivot = era.pivot(index="era", columns="signal_platform", values="pct_within_era").fillna(0.0)
    pivot.plot(kind="bar", figsize=(8, 4))
    plt.ylabel("% within era")
    plt.title("Platform AI Labeling by Era")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_platform_by_era.png"), dpi=180)
    plt.close()


def write_findings_markdown(df: pd.DataFrame, tables: Dict[str, pd.DataFrame], out_path: str) -> None:
    total = len(df)
    prev = tables["prevalence"].set_index("signal")
    agree = tables["agreement"].set_index("comparison")
    era_counts = df["era"].value_counts(dropna=False).to_dict()
    platform_counts = tables["platform_counts"].set_index("signal_platform")["n_videos"].to_dict()
    eng_platform = tables["engagement_by_platform"].set_index("signal_platform")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Measuring and Evaluating AI-Generated Content on TikTok\n\n")
        f.write("## Key Empirical Findings\n\n")
        f.write(f"- Sample size: **{total} videos**.\n")
        f.write(
            f"- Platform raw `is_aigc` prevalence: **{prev.loc['platform_raw_is_aigc', 'pct_true']:.2f}%** "
            f"({int(prev.loc['platform_raw_is_aigc', 'count_true'])}/{total}, "
            f"95% CI: {prev.loc['platform_raw_is_aigc', 'ci95_low_pct']:.2f}-{prev.loc['platform_raw_is_aigc', 'ci95_high_pct']:.2f}).\n"
        )
        f.write(
            f"- Platform derived-label prevalence (`signal_platform`): **{prev.loc['signal_platform', 'pct_true']:.2f}%** "
            f"({int(prev.loc['signal_platform', 'count_true'])}/{total}, "
            f"95% CI: {prev.loc['signal_platform', 'ci95_low_pct']:.2f}-{prev.loc['signal_platform', 'ci95_high_pct']:.2f}).\n"
        )
        f.write(
            f"- Creator AI-tag prevalence: **{prev.loc['signal_creator_tag', 'pct_true']:.2f}%** "
            f"({int(prev.loc['signal_creator_tag', 'count_true'])}/{total}).\n"
        )
        f.write(
            f"- Disclosure gap (creator-tagged AI but not platform-labeled): "
            f"**{prev.loc['disclosure_gap', 'pct_true']:.2f}%** "
            f"({int(prev.loc['disclosure_gap', 'count_true'])}/{total}).\n"
        )
        f.write(
            f"- Potential AI text signal (broad heuristic): **{prev.loc['signal_potential_text', 'pct_true']:.2f}%** "
            f"({int(prev.loc['signal_potential_text', 'count_true'])}/{total}).\n"
        )
        f.write(
            f"- Potential AI union signal (platform OR creator OR broad text): "
            f"**{prev.loc['signal_potential_ai_any', 'pct_true']:.2f}%** "
            f"({int(prev.loc['signal_potential_ai_any', 'count_true'])}/{total}).\n"
        )
        f.write(
            f"- Platform vs creator agreement: **{agree.loc['signals_platform_vs_creator', 'agreement_rate_pct']:.2f}%** "
            f"(n={int(agree.loc['signals_platform_vs_creator', 'n_non_null'])}).\n"
        )
        pos_ag = agree.loc["positive_agreement_platform_creator", "agreement_rate_pct"]
        pos_n = int(agree.loc["positive_agreement_platform_creator", "n_non_null"])
        pos_ag_str = f"{pos_ag:.2f}%" if pd.notna(pos_ag) else "N/A"
        f.write(
            f"- Positive agreement (platform vs creator, overlap among AI-positive cases): "
            f"**{pos_ag_str}** (n={pos_n}).\n"
        )
        all_agree_pct = agree.loc["signals_agree_all", "agreement_rate_pct"]
        all_agree_str = f"{all_agree_pct:.2f}%" if pd.notna(all_agree_pct) else "N/A"
        f.write(
            f"- All-signal agreement (where available): **{all_agree_str}** "
            f"(n={int(agree.loc['signals_agree_all', 'n_non_null'])}).\n"
        )
        f.write(
            f"- Era split: pre_ai={era_counts.get('pre_ai', 0)}, "
            f"post_ai={era_counts.get('post_ai', 0)}.\n\n"
        )
        if "diagnostics" in tables:
            diag = tables["diagnostics"].set_index("diagnostic")
            f.write("## Signal Coverage Diagnostics\n\n")
            f.write(
                f"- Captions with any AI token (broad text scan): **{int(diag.loc['contains_ai_token', 'count'])}** "
                f"({diag.loc['contains_ai_token', 'pct']:.4f}%).\n"
            )
            f.write(
                f"- Captions with model/tool names (e.g., Sora, Midjourney): **{int(diag.loc['contains_model_names', 'count'])}** "
                f"({diag.loc['contains_model_names', 'pct']:.4f}%).\n"
            )
            f.write(
                f"- Captions with AI-themed hashtags: **{int(diag.loc['contains_hashtag_ai', 'count'])}** "
                f"({diag.loc['contains_hashtag_ai', 'pct']:.4f}%).\n\n"
            )
        f.write("## Engagement Snapshot (Descriptive)\n\n")
        n_pos = int(platform_counts.get(True, 0))
        n_neg = int(platform_counts.get(False, 0))
        if True in eng_platform.index and False in eng_platform.index:
            f.write(
                f"- Median views (`play_count`): platform-labeled AI = **{eng_platform.loc[True, 'play_count']:.1f}** "
                f"(n={n_pos}) vs non-labeled = **{eng_platform.loc[False, 'play_count']:.1f}** (n={n_neg}).\n"
            )
            f.write(
                f"- Median like rate: platform-labeled AI = **{eng_platform.loc[True, 'like_rate']:.4f}** "
                f"vs non-labeled = **{eng_platform.loc[False, 'like_rate']:.4f}**.\n\n"
            )

        f.write("## Interpretation Notes\n\n")
        f.write("- Agreement values quantify consistency across indicators, not absolute truth.\n")
        f.write("- Overall agreement can be high when both signals are mostly negative; positive agreement is more informative under imbalance.\n")
        f.write("- Disclosure gap highlights potential under-labeling by platform mechanisms.\n")
        f.write("- Engagement differences are descriptive and should be interpreted with distribution-aware tests.\n")
        if prev.loc["platform_raw_is_aigc", "pct_true"] in (0.0, 100.0):
            f.write("- Raw platform AI flag appears degenerate (all/none), so pair this with your independent visual model before strong inference.\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="metadata_combined.csv")
    ap.add_argument("--outdir", default="research_outputs")
    args = ap.parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    os.makedirs(args.outdir, exist_ok=True)

    tables = build_core_tables(df)
    for name, table in tables.items():
        table.to_csv(os.path.join(args.outdir, f"{name}.csv"), index=False)

    save_figures(tables, args.outdir)
    write_findings_markdown(df, tables, os.path.join(args.outdir, "findings_summary.md"))

    print(f"Wrote analysis outputs to: {args.outdir}")


if __name__ == "__main__":
    main()
