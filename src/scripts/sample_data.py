import joblib
import numpy as np 

idx = 0 
split_dict = {"GRAB": "test", "ARCTIC": "test", "OMOMO": "val"}

# make sure the augment flag is false, because smplx is not perfeclty symmmetric.
for datasetname in ["ARCTIC",  "OMOMO", "GRAB"]:
    
    temp = joblib.load(f'data/motion/Body_Processed/{datasetname.lower()}_{split_dict[datasetname]}.p')
    print(f"Loaded {datasetname} {split_dict[datasetname]} dataset successfully.")
    
    not_augmented = True
    index = [list(temp.keys())[elem] for elem in np.random.choice(len(temp.keys()), 2, replace=False)]
    
    # Set augment flag to False for the selected indices    
    while not_augmented:
        if any([temp[_index_]['augment_flag'] for _index_ in index]):
            index = [list(temp.keys())[elem] for elem in np.random.choice(len(temp.keys()), 2, replace=False)]
        else:
            not_augmented = False
    
    for _index_ in index:
        joblib.dump({_index_: temp[_index_]}, f'data/sample_data/{idx:03d}.p')
        idx += 1