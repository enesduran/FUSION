"""
SAMP Hand Dataset Processor - Refactored version using unified architecture.
Extracts hand motions from SAMP dataset using SMPLX FK for wrist orientation.
Fixed version of the original broken script.
"""
import os
import sys
import glob
import smplx
import torch
import pickle
import numpy as np
from typing import Dict, List, Optional, Any

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

from src.scripts.process.base_processor import HandDatasetProcessor
from src.scripts.process.config import ProcessingConfig, SMPLX_JOINT_MIRROR_ARR
from src.scripts.process.processor_utils import ModelLoader
from src.utils.transforms import (
    axis_angle_to_quaternion, quaternion_to_axis_angle, quat_fk
)
from src.utils.process_utils import RIGHT_WRIST_BASE_LOC, LEFT_WRIST_BASE_LOC, SMPLX_JOINTS

SAMP_PATH = 'data/motion/Body_Raw/SAMP/1'
RIGHT_MANO_PATH = 'data/body_models/mano/MANO_RIGHT.pkl'
LEFT_MANO_PATH = 'data/body_models/mano/MANO_LEFT.pkl'

class SampHandProcessor(HandDatasetProcessor):
    """Processor for SAMP hand motion dataset."""
    
    # No motion variance filtering in original
    LOWEST_PERCENT = 0.50
    HIGHEST_PERCENT = 1.00
    
    def __init__(self):
        super().__init__(
            dataset_name='SAMP',
            dataset_path=SAMP_PATH,
            output_path='data/motion/Hand_Processed/samp_train.p'
        )
        
        self.device = self.get_device()
        self.body_models = ModelLoader.load_smplx_models(
            batch_size=1,  # Dynamic batch size
            num_betas=self.config.NUM_BETAS,
            device=self.device,
            config=self.config
        )

         # Load MANO mean poses (flat_hand_mean=False)
        self.r_mean_pose = smplx.create(
            model_path=RIGHT_MANO_PATH, model_type='mano',
            is_rhand=True, use_pca=False, flat_hand_mean=False, ext='pkl'
        ).hand_mean.detach().cpu().numpy()
        
        self.l_mean_pose = smplx.create(
            model_path=LEFT_MANO_PATH, model_type='mano',
            is_rhand=False, use_pca=False, flat_hand_mean=False, ext='pkl'
        ).hand_mean.detach().cpu().numpy()

        
    
    def load_sequences(self) -> List[str]:
        """Load all SAMP pkl file paths."""
        return glob.glob(os.path.join(self.dataset_path, "*.pkl"))
    
    def filter_sequences(self, sequences: List[str]) -> Dict[str, List[str]]:
        """All sequences go to train."""
        return {'train': sequences}
    
    def process_sequence(self, pkl_filepath: str) -> Optional[List[Dict[str, Any]]]:
        """Process a single SAMP sequence for hand data."""
        with open(pkl_filepath, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        
        sample_freq = int(data['mocap_framerate'] / self.config.TARGET_FPS)
        
        betas = torch.tensor(data['shape_est_betas'][:self.config.NUM_BETAS]).to(self.device).float()
        gender = data['ps']['gender']
        trans = torch.tensor(data['pose_est_trans'][::sample_freq]).to(self.device).float()
        fullpose = torch.tensor(data['pose_est_fullposes'][::sample_freq]).to(self.device).float()
        
        rhand_pose = fullpose[:, 120:]
        lhand_pose = fullpose[:, 75:120]
        
        T = len(fullpose)
        if T < self.config.WINDOW:
            return None
        
        # FK for global wrist orientations
        bm = self.body_models[gender]
        bm_rest_out = bm(betas=betas[None, :],
                         expression=torch.zeros(1, 10).float().to(self.device))
        rest_joint_pos = bm_rest_out.joints.detach()[:, :len(SMPLX_JOINT_MIRROR_ARR)]
        
        global_rot_quat, _ = quat_fk(
            axis_angle_to_quaternion(fullpose.reshape(T, -1, 3)),
            rest_joint_pos.expand(T, -1, -1),
            bm.parents
        )
        global_rot_aa = quaternion_to_axis_angle(global_rot_quat)
        
        rhand_root_orient = global_rot_aa[:, SMPLX_JOINTS['right_wrist']]
        lhand_root_orient = global_rot_aa[:, SMPLX_JOINTS['left_wrist']]
        
        # Get wrist translations
        bm_out = bm(
            betas=betas[None, :].expand(T, self.config.NUM_BETAS),
            global_orient=fullpose[:, :3],
            body_pose=fullpose[:, 3:66],
            transl=trans,
            expression=torch.zeros((T, 10)).to(self.device).float(),
            batch_size=T,
            left_hand_pose=lhand_pose,
            right_hand_pose=rhand_pose,
            jaw_pose=fullpose[:, 66:69],
            leye_pose=fullpose[:, 69:72],
            reye_pose=fullpose[:, 72:75]
        )
        
        lhand_transl = bm_out.joints[:, SMPLX_JOINTS['left_wrist']].cpu() - LEFT_WRIST_BASE_LOC
        rhand_transl = bm_out.joints[:, SMPLX_JOINTS['right_wrist']].cpu() - RIGHT_WRIST_BASE_LOC
        
        seq_name = "_".join(pkl_filepath.split('/')[-1:])
        
        results = []
        for pose_augment_flag in [True, False]:
            for time_augment_flag in [True, False]:
                if pose_augment_flag:
                    hand_trans, hand_pose, hand_orient, _ = self.mirror_left_to_right(
                        lhand_transl.numpy(), lhand_pose.cpu().numpy(),
                        lhand_root_orient.cpu().numpy()
                    )
                else:
                    hand_trans = rhand_transl.cpu().numpy()
                    hand_pose = rhand_pose.cpu().numpy()
                    hand_orient = rhand_root_orient.cpu().numpy()
                
                if time_augment_flag:
                    trans_time, pose_time, orient_time = \
                        self.flip_time(hand_trans, hand_pose, hand_orient)
                else:
                    trans_time = hand_trans.copy()
                    pose_time = hand_pose.copy()
                    orient_time = hand_orient.copy()
                
                frames = torch.arange(T).to(torch.long)
                chunks = frames.unfold(dimension=0, size=self.config.WINDOW, step=self.config.WINDOW)
                
                for chunk in chunks:
                    chunk_data = {
                        'betas': betas.cpu().numpy(),
                        'gender': gender,
                        'seq_name': seq_name,
                        'datasetname': 'SAMP',
                        'pose_augment_flag': pose_augment_flag,
                        'time_augment_flag': time_augment_flag,
                        'trans': trans_time[chunk],
                        'pose_rhand': pose_time[chunk],
                        'root_orient': orient_time[chunk],
                    }
                    results.append(chunk_data)
        
        return results if results else None


if __name__ == '__main__':
    processor = SampHandProcessor()
    processor.run()
    print('Done processing SAMP')
