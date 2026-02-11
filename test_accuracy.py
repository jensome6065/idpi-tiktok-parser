import pandas as pd

df = pd.read_csv("tiktok_combined.csv")

# Normalize manual label: True if 'AI' (case-insensitive, ignore spaces)
df["manual_ai_flag"] = df["AI/NOT AI"].astype(str).str.strip().str.upper().eq("AI")

# Basic counts
print(df["manual_ai_flag"].value_counts(dropna=False))
print(df["tiktok_labeled_aigc"].value_counts(dropna=False))
print(df["aigc_badge_type"].value_counts(dropna=False))

# 2x2 between your labels and TikTok’s badges
print(pd.crosstab(df["manual_ai_flag"], df["tiktok_labeled_aigc"],
                  rownames=["manual_ai"], colnames=["tiktok_labeled_aigc"]))

# Breakdown by badge subtype among TikTok-labeled videos
mask_tiktok = df["tiktok_labeled_aigc"] == True
print(df.loc[mask_tiktok, "aigc_badge_type"].value_counts())