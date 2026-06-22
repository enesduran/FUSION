import os 
import sys
import glob
import torch
import joblib
import numpy as np
from omegaconf import OmegaConf

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))


from src.utils.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from src.utils.process_utils import LEFT_WRIST_BASE_LOC, RIGHT_WRIST_BASE_LOC

cfg = OmegaConf.load('configs/optimize_clutch.yaml')

clutch_filenames = sorted(glob.glob('data/clutch/*.npy'))

clutch_dict_all = {}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

idx = 0 


# R_v2s = torch.tensor([[1., 0., 0.],
#                       [0., 0., 1.],
#                       [0., 1., 0.]]).reshape(3,3).float()
R_v2s = torch.tensor([[-1., 0., 0.],
                      [0., -1., 0.],
                      [0., 0., 1.]]).reshape(3,3).float()


def rotate_global(global_rot_aa, global_trans, root_offset):
 
    origin2root = global_trans + torch.from_numpy(root_offset).float()
        
    global_rot_aa = matrix_to_axis_angle(R_v2s.T @ axis_angle_to_matrix(global_rot_aa.float()))

    global_transl = torch.matmul(origin2root, R_v2s) - root_offset
 
    return global_rot_aa, global_transl


for clutch_filename in clutch_filenames:

    # 'R_w2c_sla_all', 't_w2c_sla_all', 'R_c2w_sla_all', 't_c2w_sla_all' 
    clutch_dict_idx = np.load(clutch_filename, allow_pickle=True).item()


    frames = torch.arange(clutch_dict_idx['pred_trans'].shape[1])
    chunks = frames.unfold(dimension=0, size=cfg.window, step=cfg.window) 
    

    for chunk in chunks:

        clutch_dict_idx['pred_rot'][0], clutch_dict_idx['pred_trans'][0] = rotate_global(clutch_dict_idx['pred_rot'][0], clutch_dict_idx['pred_trans'][0], LEFT_WRIST_BASE_LOC)
        clutch_dict_idx['pred_rot'][1], clutch_dict_idx['pred_trans'][1] = rotate_global(clutch_dict_idx['pred_rot'][1], clutch_dict_idx['pred_trans'][1], RIGHT_WRIST_BASE_LOC)
         

        clutch_dict_all[idx] = {"left": {"hand_pose": clutch_dict_idx['pred_hand_pose'][0, chunk].reshape(-1, 45).to(device),
                                         "global_orient": clutch_dict_idx['pred_rot'][0, chunk].to(device),
                                          "betas": clutch_dict_idx['pred_betas'][0, chunk].to(device),
                                         "transl": clutch_dict_idx['pred_trans'][0, chunk].to(device)}, 

                                "right": {"hand_pose": clutch_dict_idx['pred_hand_pose'][1, chunk].reshape(-1, 45).to(device),
                                            "global_orient": clutch_dict_idx['pred_rot'][1, chunk].to(device),
                                            "betas": clutch_dict_idx['pred_betas'][1, chunk].to(device),
                                            "transl": clutch_dict_idx['pred_trans'][1, chunk].to(device)}}

    
        idx += 1
    
joblib.dump(clutch_dict_all, 'data/clutch/clutch_dict_processed.p')
