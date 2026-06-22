import os
import csv
import random
from glob import glob

PREFIX = "https://wandr-userstudy.s3.eu-central-1.amazonaws.com/fusion/"
 
folders = {
    "Catch_Trials": [],
    "FUSION_Generation": [],
    "Merged_Dataset": [],
    "Raw_Dataset": [],
}

OUTPUT_CSV = "user_study_videos.csv"

# Collect mp4 files
for folder in folders:
    folders[folder] = [
        os.path.join('perceptual_study', folder, f)
        for f in os.listdir(os.path.join('perceptual_study', folder))
        if f.endswith(".mp4")
    ]

# --- Step 1: sample first 5 ---
# 1 from Catch_Trials
first_catch = random.sample(folders["Catch_Trials"], 2)

# 4 from the remaining folders
other_folders = (
    folders["FUSION_Generation"]
    + folders["Merged_Dataset"]
    + folders["Raw_Dataset"]
)
first_others = random.sample(other_folders, 3)

first_five = first_catch + first_others
random.shuffle(first_five)

# --- Step 2: collect remaining videos ---
used = set(first_five)

remaining = []
for vids in folders.values():
    for v in vids:
        if v not in used:
            remaining.append(v)

random.shuffle(remaining)

# --- Step 3: build final row with prefix ---
final_list = first_five + remaining
final_list = [PREFIX + path for path in final_list]

# --- Step 4: write single-line CSV ---
with open("perceptual_study/perceptual_study_videos.csv", "w", newline="") as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerow(final_list)
    
print(f"Wrote {len(final_list)} entries to output.csv")


 