import numpy as np 
import pandas as pd 

df_results = pd.read_csv('FinalResults.csv')
df_videos = pd.read_csv('perceptual_study/perceptual_study_videos.csv')


data = df_results.values
data_videos = df_videos.columns.array[0].split(';')

user_ratings = []


for user_info in data:

    user_ratings.append(list(map(lambda x: float(x), user_info[-3].split(';')[:-1])))


user_ratings = np.array(user_ratings).astype(np.float32)
user_ratings_mean = user_ratings.mean(axis=0)


catch_trials_idx = []
fusion_idx = []
random_merging_idx = []
raw_data_idx = []

CATCH_TRIALS_THRESHOLD = 2.5

for idx, video_name in enumerate(data_videos):

    if "FUSION_Generation" in video_name:
        fusion_idx.append(idx)
    elif "Raw_Dataset" in video_name:
        raw_data_idx.append(idx)
    elif "Catch_Trials" in video_name:
        catch_trials_idx.append(idx)
    else:
        random_merging_idx.append(idx)

# catch the bad ones whoe give more than 2 to the catch trials and discard them
catch_trial_mean_by_user = user_ratings[:, catch_trials_idx].mean(axis=1)

bad_idx = np.where(catch_trial_mean_by_user > CATCH_TRIALS_THRESHOLD)[0]
good_idx = np.where(catch_trial_mean_by_user <= CATCH_TRIALS_THRESHOLD)[0]

print(f'Number of users: {len(user_ratings)}, number of bad users: {len(bad_idx)}, number of good users: {len(good_idx)}')

valid_evaluations = user_ratings[good_idx].mean(axis=0)

print(f'FUSION MEAN {valid_evaluations[fusion_idx].mean():.2f} FUSION STD {valid_evaluations[fusion_idx].std():.2f}')
print(f'RANDOM MERGING MEAN {valid_evaluations[random_merging_idx].mean():.2f} RANDOM MERGING STD {valid_evaluations[random_merging_idx].std():.2f}')
print(f'RAW DATA MEAN {valid_evaluations[raw_data_idx].mean():.2f} RAW DATA STD {valid_evaluations[raw_data_idx].std():.2f}')
 

