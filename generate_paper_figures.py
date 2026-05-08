import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = "/Users/jenn/Documents/umass/ersp/idpi-tiktok-parser"
OUTDIR = os.path.join(ROOT, "paper_figures")
POSTER_OUTDIR = os.path.join(OUTDIR, "poster_ready")


def _to_bool(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool)


def _platform_labeled_from_badge(df: pd.DataFrame) -> pd.Series:
    if "signal_platform" in df.columns:
        return _to_bool(df["signal_platform"])
    badge = df.get("aigc_badge_type", pd.Series(index=df.index, dtype=object))
    badge = badge.astype(str).str.strip().str.lower()
    return ~badge.isin(["", "nan", "none"])


def _clean_plot_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "create_dt" in out.columns:
        out["create_dt"] = pd.to_datetime(out["create_dt"], errors="coerce", utc=True)
    for col in ["play_count", "like_count", "comment_count", "share_count", "like_rate", "comment_rate"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "engagement_total" not in out.columns and {"play_count", "like_count", "comment_count", "share_count"}.issubset(out.columns):
        out["engagement_total"] = out[["play_count", "like_count", "comment_count", "share_count"]].sum(axis=1)
    else:
        out["engagement_total"] = pd.to_numeric(out.get("engagement_total"), errors="coerce")
    return out


def _boxplot_two_groups(series_a: pd.Series, series_b: pd.Series, labels: list[str], title: str, ylabel: str, out_name: str) -> None:
    plt.figure(figsize=(7.2, 4.8))
    box = plt.boxplot([series_a.dropna(), series_b.dropna()], tick_labels=labels, patch_artist=True)
    for patch, color in zip(box["boxes"], ["#93c5fd", "#2563eb"]):
        patch.set_facecolor(color)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, out_name), dpi=220)
    plt.close()


def _apply_poster_style() -> None:
    plt.rcParams.update(
        {
            "figure.titlesize": 20,
            "axes.titlesize": 17,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
        }
    )


def make_prevalence_figure(df: pd.DataFrame) -> None:
    rows = [
        ("`video_is_ai_gc` true", _to_bool(df["video_is_ai_gc"]).sum()),
        ("`ai_gc_label_type` present", df["ai_gc_label_type"].notna().sum()),
        ("Derived platform-labeled", _platform_labeled_from_badge(df).sum()),
    ]
    plot_df = pd.DataFrame(rows, columns=["signal", "count"])
    plot_df["pct"] = 100 * plot_df["count"] / len(df)

    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(plot_df["signal"], plot_df["pct"], color=["#9ca3af", "#60a5fa", "#2563eb"])
    plt.ylabel("Percent of videos")
    plt.title("TikTok AI Label Prevalence by Field")
    plt.ylim(0, max(plot_df["pct"].max() * 1.25, 0.8))
    for bar, count, pct in zip(bars, plot_df["count"], plot_df["pct"]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{int(count)}\n({pct:.2f}%)",
                 ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig_01_platform_label_prevalence.png"), dpi=220)
    plt.close()


def make_label_type_breakdown_figure(df: pd.DataFrame) -> None:
    raw = df["ai_gc_label_type"]
    n_total = len(raw)
    n_present = int(raw.notna().sum())
    n_missing = n_total - n_present

    present_counts = (
        raw.dropna()
        .astype(str)
        .replace({"0.0": "0", "1.0": "1", "2.0": "2"})
        .value_counts()
        .reindex(["0", "1", "2"], fill_value=0)
    )
    present_pct = 100.0 * present_counts / max(n_present, 1)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.5))

    # Left panel: coverage of the raw field
    left_labels = ["Missing", "Present"]
    left_vals = [100.0 * n_missing / n_total, 100.0 * n_present / n_total]
    bars = axes[0].bar(left_labels, left_vals, color=["#94a3b8", "#2563eb"])
    axes[0].set_ylabel("Percent of all videos")
    axes[0].set_title("Field Coverage")
    for bar, c, p in zip(bars, [n_missing, n_present], left_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{c}\n({p:.2f}%)",
                     ha="center", va="bottom", fontsize=9)

    # Right panel: value mix among present rows only
    bars2 = axes[1].bar(present_counts.index, present_pct.values, color="#2563eb")
    axes[1].set_title(f"Value Mix Among Present Rows (n={n_present})")
    axes[1].set_ylabel("Percent within present rows")
    axes[1].set_xlabel("`ai_gc_label_type`")
    for bar, c, p in zip(bars2, present_counts.values, present_pct.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{int(c)}\n({p:.1f}%)",
                     ha="center", va="bottom", fontsize=9)

    fig.suptitle("TikTok `ai_gc_label_type` Overview", y=0.995, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(
        os.path.join(OUTDIR, "fig_02_aigc_label_type_breakdown.png"),
        dpi=220,
        bbox_inches="tight",
    )
    plt.close()


def make_cumulative_time_figure(df: pd.DataFrame) -> None:
    labeled = df.loc[_platform_labeled_from_badge(df)].copy()
    labeled["create_dt"] = pd.to_datetime(labeled["create_dt"], errors="coerce", utc=True)
    labeled = labeled.dropna(subset=["create_dt"]).sort_values("create_dt")
    labeled["cum_count"] = range(1, len(labeled) + 1)

    plt.figure(figsize=(9, 4.8))
    plt.plot(labeled["create_dt"], labeled["cum_count"], color="#2563eb", linewidth=2)
    plt.axvline(pd.Timestamp("2024-02-01", tz="UTC"), color="#dc2626", linestyle="--", linewidth=1.5)
    plt.text(pd.Timestamp("2024-02-01", tz="UTC"), max(len(labeled) * 0.08, 1), "Analysis cutoff (pre/post split)",
             rotation=90, va="bottom", ha="right", color="#dc2626")
    plt.ylabel("Cumulative platform-labeled videos")
    plt.title("Growth of TikTok AI-Labeled Videos Over Time")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig_03_cumulative_platform_labels_over_time.png"), dpi=220)
    plt.close()


def make_monthly_platform_labels_figure(df: pd.DataFrame) -> None:
    labeled = df.loc[_platform_labeled_from_badge(df)].copy()
    labeled["create_dt"] = pd.to_datetime(labeled["create_dt"], errors="coerce", utc=True)
    labeled = labeled.dropna(subset=["create_dt"])
    labeled["month"] = labeled["create_dt"].dt.to_period("M").dt.to_timestamp()
    monthly = labeled.groupby("month").size().reset_index(name="count")

    plt.figure(figsize=(9, 4.8))
    plt.plot(monthly["month"], monthly["count"], marker="o", linewidth=1.8, color="#2563eb")
    plt.axvline(pd.Timestamp("2024-02-01"), color="#dc2626", linestyle="--", linewidth=1.5)
    plt.ylabel("Platform-labeled videos per month (sample)")
    plt.title("Monthly Count of TikTok AI-Labeled Videos in Sample")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig_04_monthly_platform_labels.png"), dpi=220)
    plt.close()


def make_labeled_vs_not_figure(df: pd.DataFrame) -> None:
    platform = _platform_labeled_from_badge(df)
    counts = pd.Series(
        {
            "TikTok not labeled": int((~platform).sum()),
            "TikTok labeled AI": int(platform.sum()),
        }
    )
    pct = 100.0 * counts / len(df)

    plt.figure(figsize=(7.2, 4.6))
    bars = plt.bar(counts.index, pct.values, color=["#94a3b8", "#2563eb"])
    plt.ylabel("Percent of videos")
    plt.title("Labeled vs Not Labeled (Platform Signal)")
    for bar, c, p in zip(bars, counts.values, pct.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{int(c)}\n({p:.2f}%)",
                 ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig_13_labeled_vs_not_labeled.png"), dpi=220)
    plt.close()


def make_monthly_label_rate_figure(df: pd.DataFrame) -> None:
    plot_df = df.copy()
    plot_df["create_dt"] = pd.to_datetime(plot_df["create_dt"], errors="coerce", utc=True)
    plot_df = plot_df.dropna(subset=["create_dt"])
    # Remove obvious timestamp artifacts (e.g., epoch default 1970 rows).
    plot_df = plot_df[(plot_df["create_dt"].dt.year >= 2018) & (plot_df["create_dt"].dt.year <= 2030)]
    plot_df["month"] = plot_df["create_dt"].dt.tz_convert(None).dt.to_period("M").dt.to_timestamp()
    plot_df["platform_labeled"] = _platform_labeled_from_badge(plot_df)
    monthly = (
        plot_df.groupby("month")
        .agg(total=("platform_labeled", "size"), labeled=("platform_labeled", "sum"))
        .reset_index()
    )
    monthly["label_rate_pct"] = 100.0 * monthly["labeled"] / monthly["total"]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9.4, 6.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax1.plot(monthly["month"], monthly["label_rate_pct"], marker="o", linewidth=1.8, color="#2563eb")
    ax1.axvline(pd.Timestamp("2024-02-01"), color="#dc2626", linestyle="--", linewidth=1.3)
    ax1.set_ylabel("Label rate (%)")
    ax1.set_title("Monthly TikTok AI-Label Rate (Sample-Normalized)")

    ax2.bar(monthly["month"], monthly["total"], width=25, color="#94a3b8")
    ax2.set_ylabel("n / month")
    ax2.set_xlabel("Month")
    ax2.set_title("Monthly Sample Size (context for rate volatility)", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig_14_monthly_label_rate_pct.png"), dpi=220)
    plt.close()


def make_monthly_label_rate_figure_poster(df: pd.DataFrame) -> None:
    _apply_poster_style()
    plot_df = df.copy()
    plot_df["create_dt"] = pd.to_datetime(plot_df["create_dt"], errors="coerce", utc=True)
    plot_df = plot_df.dropna(subset=["create_dt"])
    plot_df = plot_df[(plot_df["create_dt"].dt.year >= 2018) & (plot_df["create_dt"].dt.year <= 2030)]
    plot_df["month"] = plot_df["create_dt"].dt.tz_convert(None).dt.to_period("M").dt.to_timestamp()
    plot_df["platform_labeled"] = _platform_labeled_from_badge(plot_df)
    monthly = (
        plot_df.groupby("month")
        .agg(total=("platform_labeled", "size"), labeled=("platform_labeled", "sum"))
        .reset_index()
    )
    monthly["label_rate_pct"] = 100.0 * monthly["labeled"] / monthly["total"]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11.2, 7.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax1.plot(monthly["month"], monthly["label_rate_pct"], marker="o", linewidth=2.6, color="#1d4ed8")
    ax1.axvline(pd.Timestamp("2024-02-01"), color="#b91c1c", linestyle="--", linewidth=2.0)
    ax1.text(
        pd.Timestamp("2024-02-01"),
        max(monthly["label_rate_pct"].max() * 0.85, 0.5),
        "Analysis cutoff",
        color="#b91c1c",
        rotation=90,
        va="center",
        ha="right",
        fontsize=12,
    )
    ax1.set_ylabel("Label rate (%)")
    ax1.set_title("Monthly TikTok AI-Label Rate (Sample-Normalized)")

    ax2.bar(monthly["month"], monthly["total"], width=25, color="#94a3b8")
    ax2.set_ylabel("n / month")
    ax2.set_xlabel("Month")
    ax2.set_title("Monthly Sample Size", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(POSTER_OUTDIR, "fig_14_monthly_label_rate_pct_poster.png"), dpi=260, bbox_inches="tight")
    plt.close(fig)


def make_summary_stats_figures(df: pd.DataFrame) -> None:
    plot_df = _clean_plot_df(df)
    plot_df["platform_labeled"] = _platform_labeled_from_badge(plot_df)

    era_stats = (
        plot_df.groupby("era")["play_count"]
        .agg(n="count", median="median", q1=lambda s: s.quantile(0.25), q3=lambda s: s.quantile(0.75), max="max")
        .reindex(["pre_ai", "post_ai"])
        .reset_index()
    )
    x = range(len(era_stats))
    yerr_low = era_stats["median"] - era_stats["q1"]
    yerr_high = era_stats["q3"] - era_stats["median"]
    plt.figure(figsize=(7.8, 4.8))
    bars = plt.bar(x, era_stats["median"], color=["#93c5fd", "#2563eb"])
    plt.errorbar(x, era_stats["median"], yerr=[yerr_low, yerr_high], fmt="none", ecolor="#1f2937", capsize=5)
    plt.xticks(list(x), ["Pre-AI", "Post-AI"])
    plt.ylabel("Median views (IQR whiskers)")
    plt.title("Summary Stats: Views by Era")
    for i, row in era_stats.iterrows():
        plt.text(i, row["median"], f"n={int(row['n'])}\nmax={int(row['max'])}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig_15_summary_stats_views_by_era.png"), dpi=220)
    plt.close()

    unlabeled = plot_df.loc[~plot_df["platform_labeled"], "play_count"].dropna()
    labeled = plot_df.loc[plot_df["platform_labeled"], "play_count"].dropna()

    plt.figure(figsize=(8.2, 5.1))
    box = plt.boxplot(
        [((unlabeled + 1).map(math.log10)), ((labeled + 1).map(math.log10))],
        tick_labels=["Not labeled", "TikTok labeled AI"],
        patch_artist=True,
    )
    for patch, color in zip(box["boxes"], ["#94a3b8", "#2563eb"]):
        patch.set_facecolor(color)
    plt.ylabel("log10(view_count + 1)")
    plt.title("Views by Platform Label (Log Scale)")

    u_med, u_q1, u_q3 = float(unlabeled.median()), float(unlabeled.quantile(0.25)), float(unlabeled.quantile(0.75))
    l_med, l_q1, l_q3 = float(labeled.median()), float(labeled.quantile(0.25)), float(labeled.quantile(0.75))
    plt.text(
        0.02, 0.97,
        f"Not labeled: n={len(unlabeled):,}, median={u_med:,.1f}, IQR={u_q1:,.1f}-{u_q3:,.1f}\n"
        f"TikTok labeled AI: n={len(labeled):,}, median={l_med:,.1f}, IQR={l_q1:,.1f}-{l_q3:,.1f}",
        transform=plt.gca().transAxes, ha="left", va="top", fontsize=8.8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "#d1d5db"},
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig_16_summary_stats_views_by_platform.png"), dpi=220)
    plt.close()


def make_summary_stats_platform_poster(df: pd.DataFrame) -> None:
    _apply_poster_style()
    plot_df = _clean_plot_df(df)
    plot_df["platform_labeled"] = _platform_labeled_from_badge(plot_df)
    unlabeled = plot_df.loc[~plot_df["platform_labeled"], "play_count"].dropna()
    labeled = plot_df.loc[plot_df["platform_labeled"], "play_count"].dropna()

    fig = plt.figure(figsize=(11.2, 6.8))
    ax = fig.add_subplot(111)
    box = ax.boxplot(
        [((unlabeled + 1).map(math.log10)), ((labeled + 1).map(math.log10))],
        tick_labels=["Not labeled", "Platform-labeled AI"],
        patch_artist=True,
        widths=0.45,
    )
    for patch, color in zip(box["boxes"], ["#94a3b8", "#1d4ed8"]):
        patch.set_facecolor(color)
    ax.set_ylabel("log10(view_count + 1)")
    ax.set_title("Views by Platform Label (Log Scale)")

    u_med, u_q1, u_q3 = float(unlabeled.median()), float(unlabeled.quantile(0.25)), float(unlabeled.quantile(0.75))
    l_med, l_q1, l_q3 = float(labeled.median()), float(labeled.quantile(0.25)), float(labeled.quantile(0.75))
    ax.text(
        0.02, 0.98,
        f"Not labeled: n={len(unlabeled):,}, median={u_med:,.1f}, IQR={u_q1:,.1f}-{u_q3:,.1f}\n"
        f"Platform-labeled AI: n={len(labeled):,}, median={l_med:,.1f}, IQR={l_q1:,.1f}-{l_q3:,.1f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=12.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cbd5e1"},
    )
    fig.tight_layout()
    fig.savefig(os.path.join(POSTER_OUTDIR, "fig_16_summary_stats_views_by_platform_poster.png"), dpi=260, bbox_inches="tight")
    plt.close(fig)


def make_platform_by_era_figure(df: pd.DataFrame) -> None:
    plot_df = df.copy()
    plot_df["platform_labeled"] = _platform_labeled_from_badge(plot_df)
    counts = (
        plot_df.groupby(["era", "platform_labeled"])
        .size()
        .reset_index(name="count")
    )
    totals = counts.groupby("era")["count"].transform("sum")
    counts["pct"] = 100 * counts["count"] / totals
    pivot = counts.pivot(index="era", columns="platform_labeled", values="pct").fillna(0)
    pivot = pivot.reindex(["pre_ai", "post_ai"]).fillna(0)

    ax = pivot.plot(kind="bar", stacked=True, figsize=(7.5, 4.8), color=["#94a3b8", "#2563eb"])
    ax.set_ylabel("Percent within era")
    ax.set_title("Share of Platform-Labeled Videos by Era")
    ax.legend(["Not labeled", "TikTok labeled AI"], title="")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig_05_platform_label_share_by_era.png"), dpi=220)
    plt.close()


def make_engagement_by_era_figures(df: pd.DataFrame) -> None:
    plot_df = _clean_plot_df(df)
    plot_df = plot_df.dropna(subset=["play_count"])
    plot_df["era"] = pd.Categorical(plot_df["era"], categories=["pre_ai", "post_ai"], ordered=True)
    pre_views = (plot_df.loc[plot_df["era"] == "pre_ai", "play_count"] + 1).map(math.log10)
    post_views = (plot_df.loc[plot_df["era"] == "post_ai", "play_count"] + 1).map(math.log10)
    _boxplot_two_groups(
        pre_views, post_views,
        ["Pre-AI", "Post-AI"],
        "View Distribution Before vs After AI Boom",
        "log10(view_count + 1)",
        "fig_06_views_by_era_boxplot.png",
    )

    pre_like = plot_df.loc[plot_df["era"] == "pre_ai", "like_rate"]
    post_like = plot_df.loc[plot_df["era"] == "post_ai", "like_rate"]
    _boxplot_two_groups(
        pre_like, post_like,
        ["Pre-AI", "Post-AI"],
        "Like Rate Before vs After AI Boom",
        "like_rate",
        "fig_07_like_rate_by_era_boxplot.png",
    )


def make_engagement_by_platform_figures(df: pd.DataFrame) -> None:
    plot_df = _clean_plot_df(df)
    plot_df["platform_labeled"] = _platform_labeled_from_badge(plot_df)
    unlabeled_views = (plot_df.loc[~plot_df["platform_labeled"], "play_count"] + 1).dropna().map(math.log10)
    labeled_views = (plot_df.loc[plot_df["platform_labeled"], "play_count"] + 1).dropna().map(math.log10)
    _boxplot_two_groups(
        unlabeled_views, labeled_views,
        ["Not labeled", "TikTok labeled AI"],
        "Views by TikTok AI Label Status",
        "log10(view_count + 1)",
        "fig_08_views_by_platform_label_boxplot.png",
    )

    unlabeled_like = plot_df.loc[~plot_df["platform_labeled"], "like_rate"]
    labeled_like = plot_df.loc[plot_df["platform_labeled"], "like_rate"]
    _boxplot_two_groups(
        unlabeled_like, labeled_like,
        ["Not labeled", "TikTok labeled AI"],
        "Like Rate by TikTok AI Label Status",
        "like_rate",
        "fig_09_like_rate_by_platform_label_boxplot.png",
    )


def make_manual_accuracy_figure(df: pd.DataFrame) -> None:
    plot_df = df.copy()
    plot_df["manual_label"] = plot_df["signal_manual"].map({1: "Manual AI", 0: "Manual Not AI"}).fillna("Missing")
    plot_df["platform_label"] = _to_bool(plot_df["signal_platform"]).map({True: "TikTok labeled AI", False: "TikTok not labeled"})
    ctab = pd.crosstab(plot_df["manual_label"], plot_df["platform_label"]).reindex(
        ["Manual AI", "Manual Not AI"], fill_value=0
    )

    ax = ctab.plot(kind="bar", stacked=True, figsize=(7.5, 4.8), color=["#94a3b8", "#2563eb"])
    ax.set_ylabel("Number of videos")
    ax.set_title("Manual Labels vs TikTok Platform Labels")
    ax.legend(title="")
    for container in ax.containers:
        ax.bar_label(container, label_type="center", color="white", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig_10_manual_vs_platform_labels.png"), dpi=220)
    plt.close()


def make_manual_engagement_figures(df: pd.DataFrame) -> None:
    plot_df = _clean_plot_df(df)
    ai_views = (plot_df.loc[plot_df["signal_manual"] == 1, "play_count"] + 1).dropna().map(math.log10)
    non_ai_views = (plot_df.loc[plot_df["signal_manual"] == 0, "play_count"] + 1).dropna().map(math.log10)
    _boxplot_two_groups(
        non_ai_views, ai_views,
        ["Manual Not AI", "Manual AI"],
        "Views in the Manual Audit Sample",
        "log10(view_count + 1)",
        "fig_11_manual_views_boxplot.png",
    )

    non_ai_like = plot_df.loc[plot_df["signal_manual"] == 0, "like_rate"]
    ai_like = plot_df.loc[plot_df["signal_manual"] == 1, "like_rate"]
    _boxplot_two_groups(
        non_ai_like, ai_like,
        ["Manual Not AI", "Manual AI"],
        "Like Rate in the Manual Audit Sample",
        "like_rate",
        "fig_12_manual_like_rate_boxplot.png",
    )


def write_manifest(metadata: pd.DataFrame, manual: pd.DataFrame) -> None:
    platform_labeled = _platform_labeled_from_badge(metadata)
    manifest = pd.DataFrame(
        [
            ["fig_01_platform_label_prevalence.png", "Prevalence", "Compares `video_is_ai_gc`, `ai_gc_label_type`, and derived platform labels."],
            ["fig_02_aigc_label_type_breakdown.png", "Prevalence", "Shows the distribution of raw `ai_gc_label_type` values."],
            ["fig_03_cumulative_platform_labels_over_time.png", "Time trend", "Cumulative count of TikTok-labeled videos over time."],
            ["fig_04_monthly_platform_labels.png", "Time trend", "Monthly counts of TikTok-labeled videos."],
            ["fig_05_platform_label_share_by_era.png", "Time trend", "Share of platform-labeled videos in `pre_ai` vs `post_ai` eras."],
            ["fig_06_views_by_era_boxplot.png", "Engagement", "View distribution before vs after the AI-boom cutoff."],
            ["fig_07_like_rate_by_era_boxplot.png", "Engagement", "Like-rate distribution before vs after the AI-boom cutoff."],
            ["fig_08_views_by_platform_label_boxplot.png", "Engagement", "Views for TikTok-labeled vs non-labeled videos."],
            ["fig_09_like_rate_by_platform_label_boxplot.png", "Engagement", "Like rates for TikTok-labeled vs non-labeled videos."],
            ["fig_10_manual_vs_platform_labels.png", "Accuracy / audit", "Manual labels crossed with TikTok platform labels in the 86-video audit set."],
            ["fig_11_manual_views_boxplot.png", "Accuracy / audit", "Views for manually labeled AI vs manually labeled non-AI videos."],
            ["fig_12_manual_like_rate_boxplot.png", "Accuracy / audit", "Like rates for manually labeled AI vs manually labeled non-AI videos."],
            ["fig_13_labeled_vs_not_labeled.png", "Prevalence", "Direct comparison of TikTok-labeled vs not-labeled counts/percentages."],
            ["fig_14_monthly_label_rate_pct.png", "Time trend", "Monthly percentage labeled (normalizes by monthly sample size)."],
            ["fig_15_summary_stats_views_by_era.png", "Engagement", "Median/IQR/max view summary for pre_ai vs post_ai groups."],
            ["fig_16_summary_stats_views_by_platform.png", "Engagement", "Median/IQR/max view summary for platform-labeled vs non-labeled groups."],
        ],
        columns=["file", "section", "description"],
    )
    manifest.to_csv(os.path.join(OUTDIR, "figure_manifest.csv"), index=False)

    summary = pd.DataFrame(
        [
            ["metadata_rows", len(metadata)],
            ["platform_labeled_rows", int(platform_labeled.sum())],
            ["manual_rows", len(manual)],
            ["manual_ai_rows", int((manual["signal_manual"] == 1).sum())],
            ["manual_not_ai_rows", int((manual["signal_manual"] == 0).sum())],
        ],
        columns=["metric", "value"],
    )
    summary.to_csv(os.path.join(OUTDIR, "figure_pack_summary.csv"), index=False)
    plt.close()


def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(POSTER_OUTDIR, exist_ok=True)

    metadata = pd.read_csv(os.path.join(ROOT, "metadata_combined.jsondir.csv"), low_memory=False)
    manual = pd.read_csv(os.path.join(ROOT, "tiktok_database_combined.csv"), low_memory=False)

    make_prevalence_figure(metadata)
    make_label_type_breakdown_figure(metadata)
    make_cumulative_time_figure(metadata)
    make_monthly_platform_labels_figure(metadata)
    make_platform_by_era_figure(metadata)
    make_engagement_by_era_figures(metadata)
    make_engagement_by_platform_figures(metadata)
    make_manual_accuracy_figure(manual)
    make_manual_engagement_figures(manual)
    make_labeled_vs_not_figure(metadata)
    make_monthly_label_rate_figure(metadata)
    make_summary_stats_figures(metadata)
    make_monthly_label_rate_figure_poster(metadata)
    make_summary_stats_platform_poster(metadata)
    write_manifest(metadata, manual)

    print(f"Wrote figures to {OUTDIR}")


if __name__ == "__main__":
    main()
