import pandas as pd

df = pd.read_csv("metadata_combined.jsondir.csv", low_memory=False)
ai_vids = df[df["signal_potential_ai_any"] == True]

print(f"Found {len(ai_vids)} videos tagged as potential AI out of {len(df)} total files.")
print("=" * 60)

for _, row in ai_vids.iterrows():
    vid = row["video_id"]
    author = row.get("author_unique_id", "unknown")
    desc = str(row.get("description", "")).replace("\n", " ").strip()
    
    # Check what type of AI flag it caught
    flags = []
    if row.get("signal_platform"): flags.append("Platform Flagged")
    if row.get("signal_creator_tag"): flags.append("Creator Tag (#aifilter/etc)")
    if row.get("potential_ai_text_signal"): flags.append("Potential AI Text (caption broad match)")
    
    url = f"https://www.tiktok.com/@{author}/video/{vid}"
    
    print(f"URL:     {url}")
    print(f"Signals: {', '.join(flags)}")
    print(f"Caption: {desc}")
    print("-" * 60)
