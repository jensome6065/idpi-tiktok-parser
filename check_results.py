import pandas as pd
import json

df = pd.read_csv("metadata_combined.jsondir.csv", low_memory=False)

# Show the 24 videos with AI value hits
vals = df[["video_id","description","is_aigc","ai_value_hits_json",
           "signal_platform","signal_creator_tag","signal_potential_ai_any"]].copy()
vals["ai_value_hits_json"] = vals["ai_value_hits_json"].fillna("[]").astype(str)
hits = vals[vals["ai_value_hits_json"] != "[]"]
print(f"=== {len(hits)} videos with AI string-value hits ===\n")
for _, row in hits.iterrows():
    vid = row["video_id"]
    desc = str(row["description"])[:80]
    try:
        pairs = json.loads(row["ai_value_hits_json"])
    except Exception:
        pairs = []
    print(f"Video: {vid}")
    print(f"  Desc: {desc}")
    for p in pairs[:5]:
        path = p.get("path", "?")
        value = str(p.get("value", ""))[:120]
        print(f"  HIT: {path} = {value}")
    print()

# Summary table
print("\n=== FULL FEATURE COMPARISON SUMMARY ===\n")
print(f"Total videos:                  {len(df)}")
print(f"IsAigc = True:                 {(df['is_aigc'] == True).sum()}")
print(f"signal_platform = True:        {(df['signal_platform'] == True).sum()}")
print(f"signal_creator_tag = True:     {(df['signal_creator_tag'] == True).sum()}")
print(f"hashtag_ai_signal = True:      {(df['hashtag_ai_signal'] == True).sum()}")
print(f"caption_ai_signal = True:      {(df['caption_ai_signal'] == True).sum()}")
print(f"potential_ai_text = True:      {(df['potential_ai_text_signal'] == True).sum()}")
print(f"signal_potential_ai_any:       {(df['signal_potential_ai_any'] == True).sum()}")
print(f"Videos with AI value hits:     {len(hits)}")
strict = df["ai_keys_strict"].dropna().astype(str)
strict_nonempty = strict[strict != ""]
print(f"Videos with strict AI keys:    {len(strict_nonempty)}")
print(f"  (all are: itemInfo.itemStruct.IsAigc)")
print(f"\nEra distribution:")
print(df["era"].value_counts().to_string())
print(f"\nDisclosure gap:                {(df['disclosure_gap'] == True).sum()}")
