# Measuring and Evaluating AI-Generated Content on TikTok

## Key Empirical Findings

- Sample size: **2865 videos**.
- Platform raw `is_aigc` prevalence: **0.00%** (0/2865, 95% CI: 0.00-0.13).
- Platform derived-label prevalence (`signal_platform`): **0.17%** (5/2865, 95% CI: 0.07-0.41).
- Creator AI-tag prevalence: **0.00%** (0/2865).
- Disclosure gap (creator-tagged AI but not platform-labeled): **0.00%** (0/2865).
- Platform vs creator agreement: **99.83%** (n=2865).
- Positive agreement (platform vs creator, overlap among AI-positive cases): **0.00%** (n=5).
- All-signal agreement (where available): **N/A** (n=0).
- Era split: pre_ai=1341, post_ai=1524.

## Engagement Snapshot (Descriptive)

- Median views (`play_count`): platform-labeled AI = **315.0** (n=5) vs non-labeled = **166.0** (n=2860).
- Median like rate: platform-labeled AI = **0.0286** vs non-labeled = **0.1198**.

## Interpretation Notes

- Agreement values quantify consistency across indicators, not absolute truth.
- Overall agreement can be high when both signals are mostly negative; positive agreement is more informative under imbalance.
- Disclosure gap highlights potential under-labeling by platform mechanisms.
- Engagement differences are descriptive and should be interpreted with distribution-aware tests.
- Raw platform AI flag appears degenerate (all/none), so pair this with your independent visual model before strong inference.
