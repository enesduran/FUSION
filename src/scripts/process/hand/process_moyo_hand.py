"""
MOYO Hand Dataset Processor - Refactored version using unified architecture.
Extracts hand motions from MOYO dataset using SMPLX FK for wrist orientation.
"""
import os
import sys
import glob
import smplx
import torch
import numpy as np
from typing import Dict, List, Optional, Any

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

from src.scripts.process.base_processor import HandDatasetProcessor
from src.scripts.process.config import ProcessingConfig, SMPLX_JOINT_MIRROR_ARR, DATASET_PATHS
from src.scripts.process.processor_utils import ModelLoader
from src.utils.transforms3d import loc2vel
from src.utils.transforms import (
    axis_angle_to_matrix, axis_angle_to_quaternion,
    quaternion_to_axis_angle, quat_fk
)
from src.utils.process_utils import RIGHT_WRIST_BASE_LOC, LEFT_WRIST_BASE_LOC, SMPLX_JOINTS

MOYO_PATH = DATASET_PATHS.get('moyo', 'data/motion/Hand_Raw/MOYO/')
VICON_FPS = 60


class MoyoHandProcessor(HandDatasetProcessor):
    """Processor for MOYO hand motion dataset."""
    
    LOWEST_PERCENT = 0.30
    HIGHEST_PERCENT = 0.90
    
    def __init__(self):
        super().__init__(
            dataset_name='MOYO',
            dataset_path=MOYO_PATH,
            output_path='data/motion/Hand_Processed/moyo_train.p'
        )
        
        self.device = self.get_device()
        
        # MOYO uses 300 betas
        self.num_betas_moyo = 300
        self.body_models = self._load_body_models()
    
    def _load_body_models(self):
        """Load SMPLX body models with MOYO-specific num_betas."""
        models = {}
        for gender in ['male', 'female', 'neutral']:
            path = getattr(self.config, f'SMPLX_{gender.upper()}_PATH')
            bm = smplx.create(
                model_path=path,
                gender=gender,
                num_betas=self.num_betas_moyo,
                flat_hand_mean=True,
                use_pca=False
            ).to(self.device)
            for p in bm.parameters():
                p.requires_grad = False
            models[gender] = bm
        return models
    
    @staticmethod
    def filter_grasp_frames(data):
        """Filter frames based on hand velocity."""
        len_motion = data['poses'].shape[0]
        
        mocap_time_length = data['mocap_time_length']
        mocap_frame_rate = data['mocap_frame_rate']
        assert len_motion == np.round(mocap_frame_rate * mocap_time_length), \
            f"{len_motion} {mocap_time_length * mocap_frame_rate}"
        
        idxs = np.arange(len_motion)
        
        hand = data['poses'][:, 17 * 3:18 * 3]
        hand_rotmat = axis_angle_to_matrix(torch.tensor(hand))
        hand_ang_vel = loc2vel(hand_rotmat, fps=30).abs().norm(dim=-1).norm(dim=-1).squeeze()
        
        start_frame = idxs[hand_ang_vel > 0.6][1]
        start_fil = idxs > start_frame
        
        skip_frame = int(data['mocap_frame_rate'] / 30)
        fps_fil = (idxs % skip_frame) == 0
        
        grasp_motion_frames = fps_fil * start_fil
        return grasp_motion_frames
    
    def load_sequences(self) -> List[str]:
        """Load all MOYO npz file paths."""
        return glob.glob(os.path.join(self.dataset_path, "*/*/*"))
    
    def filter_sequences(self, sequences: List[str]) -> Dict[str, List[str]]:
        """All sequences go to train."""
        return {'train': sequences}
    
    def process_sequence(self, npz_filepath: str) -> Optional[List[Dict[str, Any]]]:
        """Process a single MOYO sequence for hand data."""
        data = np.load(npz_filepath, allow_pickle=True)
        
        # Only process smplx_locked_head
        if data['surface_model_type'].item() != 'smplx_locked_head':
            return None
        
        filter_frames = self.filter_grasp_frames(data)
        
        betas = torch.tensor(data['betas']).float().to(self.device)
        gender = data['gender'].item()
        
        trans = torch.tensor(data['trans'][filter_frames]).float().to(self.device)
        fullpose = torch.tensor(data['poses'][filter_frames]).float().to(self.device)
        
        T = len(fullpose)
        if T < self.config.WINDOW:
            return None
        
        rhand_pose = fullpose[:, 120:]
        lhand_pose = fullpose[:, 75:120]
        
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
            betas=betas[None, :].expand(T, self.num_betas_moyo),
            global_orient=fullpose[:, :3],
            body_pose=fullpose[:, 3:66],
            transl=trans,
            expression=torch.zeros((T, 10)).to(self.device).float(),
            left_hand_pose=lhand_pose,
            right_hand_pose=rhand_pose,
            jaw_pose=fullpose[:, 66:69],
            leye_pose=fullpose[:, 69:72],
            reye_pose=fullpose[:, 72:75]
        )
        
        lhand_transl = bm_out.joints[:, SMPLX_JOINTS['left_wrist']].cpu() - LEFT_WRIST_BASE_LOC
        rhand_transl = bm_out.joints[:, SMPLX_JOINTS['right_wrist']].cpu() - RIGHT_WRIST_BASE_LOC
        
        seq_name = "_".join(npz_filepath.split('/')[-3:])
        
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
                        'seq_name': self.motion_idx,
                        'datasetname': 'MOYO',
                        'pose_augment_flag': pose_augment_flag,
                        'time_augment_flag': time_augment_flag,
                        'trans': trans_time[chunk],
                        'pose_rhand': pose_time[chunk],
                        'root_orient': orient_time[chunk],
                    }
                    results.append(chunk_data)
        
        return results if results else None


if __name__ == '__main__':
    processor = MoyoHandProcessor()
    processor.run()
    print('Done processing MOYO')
