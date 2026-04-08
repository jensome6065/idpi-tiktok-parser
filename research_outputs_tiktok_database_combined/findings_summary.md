# Measuring and Evaluating AI-Generated Content on TikTok

## Key Empirical Findings

- Sample size: **86 videos**.
- Platform raw `is_aigc` prevalence: **0.00%** (0/86, 95% CI: 0.00-4.28).
- Platform derived-label prevalence (`signal_platform`): **0.00%** (0/86, 95% CI: 0.00-4.28).
- Creator AI-tag prevalence: **11.63%** (10/86).
- Disclosure gap (creator-tagged AI but not platform-labeled): **11.63%** (10/86).
- Potential AI text signal (broad heuristic): **0.00%** (0/86).
- Potential AI union signal (platform OR creator OR broad text): **0.00%** (0/86).
- Platform vs creator agreement: **88.37%** (n=86).
- Positive agreement (platform vs creator, overlap among AI-positive cases): **0.00%** (n=10).
- All-signal agreement (where available): **N/A** (n=0).
- Era split: pre_ai=27, post_ai=59.

## Signal Coverage Diagnostics

- Captions with any AI token (broad text scan): **11** (12.7907%).
- Captions with model/tool names (e.g., Sora, Midjourney): **10** (11.6279%).
- Captions with AI-themed hashtags: **10** (11.6279%).

## Engagement Snapshot (Descriptive)

## Interpretation Notes

- Agreement values quantify consistency across indicators, not absolute truth.
- Overall agreement can be high when both signals are mostly negative; positive agreement is more informative under imbalance.
- Disclosure gap highlights potential under-labeling by platform mechanisms.
- Engagement differences are descriptive and should be interpreted with distribution-aware tests.
- Raw platform AI flag appears degenerate (all/none), so pair this with your independent visual model before strong inference.
