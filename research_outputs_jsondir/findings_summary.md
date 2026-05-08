# Measuring and Evaluating AI-Generated Content on TikTok

## Key Empirical Findings

- Sample size: **10200 videos**.
- Platform raw `is_aigc` prevalence: **0.00%** (0/10200, 95% CI: 0.00-0.04).
- Platform derived-label prevalence (`signal_platform`): **0.67%** (68/10200, 95% CI: 0.53-0.84).
- Creator AI-tag prevalence: **0.01%** (1/10200).
- Disclosure gap (creator-tagged AI but not platform-labeled): **0.01%** (1/10200).
- Potential AI text signal (broad heuristic): **0.09%** (9/10200).
- Potential AI union signal (platform OR creator OR broad text): **0.76%** (78/10200).
- Platform vs creator agreement: **99.32%** (n=10200).
- Positive agreement (platform vs creator, overlap among AI-positive cases): **0.00%** (n=69).
- All-signal agreement (where available): **N/A** (n=0).
- Era split: pre_ai=4316, post_ai=5884.

## Signal Coverage Diagnostics

- Captions with any AI token (broad text scan): **9** (0.0882%).
- Captions with model/tool names (e.g., Sora, Midjourney): **1** (0.0098%).
- Captions with AI-themed hashtags: **3** (0.0294%).

## Caption hashtag metadata (descriptive)

- Videos with at least one extracted hashtag (`hashtags`): **3506** / 10200 (34.37%).
- Videos matching the curated AI hashtag lexicon (`hashtag_ai_signal`): **1** (0.01%).
- Most frequent hashtags (case-insensitive, top 8): `capcut` (n=629), `fyp` (n=485), `foryou` (n=390), `foryoupage` (n=307), `fypシ` (n=283), `viral` (n=242), `duet` (n=187), `tiktok` (n=153).
- Saved tables: `hashtag_ai_by_platform.csv`, `hashtag_top_terms.csv`.

## Engagement Snapshot (Descriptive)

- Median views (`play_count`): platform-labeled AI = **201.5** (n=68) vs non-labeled = **162.0** (n=10132).
- Median like rate: platform-labeled AI = **0.0981** vs non-labeled = **0.1148**.

## Interpretation Notes

- Agreement values quantify consistency across indicators, not absolute truth.
- Overall agreement can be high when both signals are mostly negative; positive agreement is more informative under imbalance.
- Disclosure gap highlights potential under-labeling by platform mechanisms.
- Engagement differences are descriptive and should be interpreted with distribution-aware tests.
- Raw platform AI flag appears degenerate (all/none), so pair this with your independent visual model before strong inference.

## Future research

- Characterize the **full** hashtag and caption metadata distribution (frequencies, co-occurrence, drift over time) and relate it to `ai_gc_label_type`, manual labels, and external visual taxonomies (for example SightEngine categories once joined at row level).
- Extend beyond the fixed AI hashtag lexicon to data-driven tagging or embedding clusters, then test agreement with platform and human labels.
