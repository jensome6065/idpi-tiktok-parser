# Measuring and Evaluating AI-Generated Content on TikTok

## Key Empirical Findings

- Sample size: **10200 videos**.
- Platform raw `is_aigc` prevalence: **0.00%** (0/10200, 95% CI: 0.00-0.04).
- Platform derived-label prevalence (`signal_platform`): **0.00%** (0/10200, 95% CI: 0.00-0.04).
- Creator AI-tag prevalence: **0.01%** (1/10200).
- Disclosure gap (creator-tagged AI but not platform-labeled): **0.01%** (1/10200).
- Potential AI text signal (broad heuristic): **0.09%** (9/10200).
- Potential AI union signal (platform OR creator OR broad text): **0.10%** (10/10200).
- Platform vs creator agreement: **99.99%** (n=10200).
- Positive agreement (platform vs creator, overlap among AI-positive cases): **0.00%** (n=1).
- All-signal agreement (where available): **N/A** (n=0).
- Era split: pre_ai=4316, post_ai=5884.

## Signal Coverage Diagnostics

- Captions with any AI token (broad text scan): **9** (0.0882%).
- Captions with model/tool names (e.g., Sora, Midjourney): **1** (0.0098%).
- Captions with AI-themed hashtags: **3** (0.0294%).

## Engagement Snapshot (Descriptive)

## Interpretation Notes

- Agreement values quantify consistency across indicators, not absolute truth.
- Overall agreement can be high when both signals are mostly negative; positive agreement is more informative under imbalance.
- Disclosure gap highlights potential under-labeling by platform mechanisms.
- Engagement differences are descriptive and should be interpreted with distribution-aware tests.
- Raw platform AI flag appears degenerate (all/none), so pair this with your independent visual model before strong inference.
