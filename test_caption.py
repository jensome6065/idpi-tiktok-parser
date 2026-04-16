from metadata_parser import *

# This is the caption from the video
desc = "Tham trai san ni nhung #bactuday #thamtraisanphongkhach #thamtraisanninhung"

print("=== Caption ===")
print(desc)
print()

# Step 1: Extract all hashtags
tags = extract_hashtags(desc)
print("=== All hashtags found ===")
print(tags)
print()

# Step 2: Check against AI hashtag list
ht_signal, ht_matches = hashtag_ai_analysis(desc)
print("=== Hashtag AI signal ===")
print(f"  Signal: {ht_signal}")
print(f"  Matches: {ht_matches}")
print()

# Step 3: Caption AI phrase check
cap_signal, cap_match = caption_ai_analysis(desc)
print("=== Caption AI signal ===")
print(f"  Signal: {cap_signal}")
print(f"  Match: {cap_match}")
print()

# Step 4: Broad potential AI text
pot_signal, pot_match = potential_ai_text_analysis(desc)
print("=== Potential AI text signal ===")
print(f"  Signal: {pot_signal}")
print(f"  Match: {pot_match}")
print()

# Check each hashtag individually
print("=== Checking each hashtag against AI list ===")
for t in tags:
    in_list = t.lower() in AI_HASHTAGS
    status = "MATCH!" if in_list else "no match"
    print(f"  #{t} -> {status}")
print()

# Show the AI hashtag list for reference
print(f"=== AI Hashtags list ({len(AI_HASHTAGS)} total) ===")
for h in sorted(AI_HASHTAGS):
    print(f"  #{h}")
