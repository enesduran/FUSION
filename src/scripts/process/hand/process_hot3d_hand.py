"""
HOT3D Hand Dataset Processor - Refactored version using unified architecture.
Multi-stage pipeline: purge JSONL → temporal assembly → chunking + filtering.
"""
import os
import sys
import glob
import copy
import json
import smplx
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Any

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

from src.scripts.process.base_processor import HandDatasetProcessor
from src.scripts.process.config import ProcessingConfig, DATASET_PATHS
from src.utils.transforms import quaternion_to_axis_angle

HOT3D_PATH = '/is/cluster/fast/eduran2/hot3d/hot3d/dataset'
RIGHT_MANO_PATH = 'data/body_models/mano/MANO_RIGHT.pkl'
LEFT_MANO_PATH = 'data/body_models/mano/MANO_LEFT.pkl'

# 30 fps, time in nanoseconds
THRESHOLD_COEF = 1.1
THRESHOLD_NS = 1e9 / 30
NUM_PCA_COMPS = 15


class Hot3dHandProcessor(HandDatasetProcessor):
    """Processor for HOT3D hand motion dataset."""
    
    LOWEST_PERCENT = 0.25
    HIGHEST_PERCENT = 0.99
    
    FUTURE = 0
    PAST = 0
    
    def __init__(self):
        super().__init__(
            dataset_name='HOT3D',
            dataset_path=HOT3D_PATH,
            output_path='data/motion/Hand_Processed/hot3d_train.p'
        )
        
        # Load PCA components
        self.sbj_rh_pca = smplx.create(
            model_path=RIGHT_MANO_PATH, model_type='mano',
            is_rhand=True, use_pca=True, batch_size=self.config.WINDOW,
            num_pca_comps=NUM_PCA_COMPS, ext='pkl'
        ).np_hand_components
        
        self.sbj_lh_pca = smplx.create(
            model_path=LEFT_MANO_PATH, model_type='mano',
            is_rhand=False, use_pca=True, batch_size=self.config.WINDOW,
            num_pca_comps=NUM_PCA_COMPS, ext='pkl'
        ).np_hand_components
        
        # Load mean poses
        self.r_mean_pose = smplx.create(
            model_path=RIGHT_MANO_PATH, model_type='mano',
            is_rhand=True, use_pca=True, flat_hand_mean=False, ext='pkl'
        ).hand_mean
        
        self.l_mean_pose = smplx.create(
            model_path=LEFT_MANO_PATH, model_type='mano',
            is_rhand=False, use_pca=True, num_pca_comps=NUM_PCA_COMPS,
            flat_hand_mean=False, ext='pkl'
        ).hand_mean
    
    def _purge_sequences(self):
        """Parse JSONL files and assemble temporal hand pose sequences."""
        all_jsonl_files = glob.glob(os.path.join(self.dataset_path, '*/mano_hand_pose_trajectory.jsonl'))
        
        processed_datalist = []
        raw_datalist = []
        mocap_frame_rate = 30
        
        # Load raw JSONL data
        for json_filepath in tqdm(sorted(all_jsonl_files), desc="Loading HOT3D JSONL"):
            with open(json_filepath, 'r', encoding='utf-8') as file:
                json_list = [json.loads(line) for line in file]
            
            if len(json_list) > 0:
                raw_datalist.append(json_list)
        
        # Process each sequence
        for seq in tqdm(raw_datalist, desc="Purging HOT3D sequences"):
            # Validate temporal continuity
            timesteps_seq = np.array([v['timestamp_ns'] for v in seq])
            timesteps_seq_sorted = copy.deepcopy(timesteps_seq)
            timesteps_seq_sorted.sort()
            
            assert (timesteps_seq_sorted == timesteps_seq).all()
            
            timesteps_seq_diff = np.diff(timesteps_seq)
            invalid_indices = (timesteps_seq_diff > THRESHOLD_COEF * THRESHOLD_NS)
            invalid_indices = invalid_indices.nonzero()[0]
            
            if sum(invalid_indices) == 1 and invalid_indices[0]:
                seq = seq[1:]
            elif sum(timesteps_seq_diff > THRESHOLD_COEF * THRESHOLD_NS) > 1:
                continue
            
            # Process left (0) and right (1) hands
            for k_str in ['0', '1']:
                temporal_pose_dict = {
                    'poses': [], 'betas': [], 'trans': [],
                    'root_orient': [], 'pose_augment_flag': None,
                    'mocap_frame_rate': mocap_frame_rate
                }
                temporal_pose_list = []
                
                for _i_, v in enumerate(seq[::2]):
                    handedness = 'right' if k_str == '1' else 'left'
                    
                    if k_str not in v['hand_poses'].keys() or _i_ == len(seq) - 1:
                        # End of continuous segment - process accumulated data
                        if len(temporal_pose_list) > self.config.WINDOW:
                            hand_pca_template = self.sbj_rh_pca if handedness == 'right' else self.sbj_lh_pca
                            hand_mean_pose = self.r_mean_pose if handedness == 'right' else self.l_mean_pose
                            
                            hand_poses_pca = np.array(temporal_pose_dict['poses'])
                            hand_poses_fullspace = hand_poses_pca @ hand_pca_template + hand_mean_pose.numpy()
                            root_orient = quaternion_to_axis_angle(
                                torch.tensor(temporal_pose_dict['root_orient'])
                            ).numpy()
                            
                            trans = np.array(temporal_pose_dict['trans'])
                            
                            # Mirror left hand to right
                            if handedness == 'left':
                                hand_poses_fullspace_ = hand_poses_fullspace.copy().reshape(-1, 15, 3)
                                hand_poses_fullspace_[:, :, 1:] *= -1
                                hand_poses_fullspace_ = hand_poses_fullspace_.reshape(-1, 45)
                                
                                root_orient_ = root_orient.copy()
                                root_orient_[:, 1:] *= -1
                                
                                trans_ = trans.copy()
                                trans_[:, 0] *= -1
                            else:
                                hand_poses_fullspace_ = hand_poses_fullspace.copy()
                                root_orient_ = root_orient.copy()
                                trans_ = trans.copy()
                            
                            temporal_pose_dict['trans'] = trans_
                            temporal_pose_dict['poses'] = hand_poses_fullspace_
                            temporal_pose_dict['betas'] = np.array(temporal_pose_dict['betas'])
                            temporal_pose_dict['root_orient'] = np.array(root_orient_)
                            temporal_pose_dict['pose_augment_flag'] = (handedness == 'left')
                            
                            processed_datalist.append(temporal_pose_dict)
                        
                        # Reset for next segment
                        temporal_pose_dict = {
                            'poses': [], 'betas': [], 'trans': [],
                            'root_orient': [], 'pose_augment_flag': None,
                            'mocap_frame_rate': mocap_frame_rate
                        }
                        temporal_pose_list = []
                    else:
                        temporal_pose_dict['poses'].append(v['hand_poses'][k_str]['pose'])
                        temporal_pose_dict['betas'].append(v['hand_poses'][k_str]['betas'])
                        temporal_pose_dict['trans'].append(v['hand_poses'][k_str]['wrist_xform']['t_xyz'])
                        temporal_pose_dict['root_orient'].append(v['hand_poses'][k_str]['wrist_xform']['q_wxyz'])
                        temporal_pose_dict['pose_augment_flag'] = (handedness == 'left')
                        temporal_pose_list.append(v['hand_poses'][k_str])
        
        return processed_datalist
    
    def load_sequences(self) -> List[Dict]:
        """Load and purge HOT3D sequences from JSONL files."""
        print("Purging HOT3D sequences...")
        return self._purge_sequences()
    
    def filter_sequences(self, sequences: List[Dict]) -> Dict[str, List[Dict]]:
        """All go to train."""
        return {'train': sequences}
    
    def process_sequence(self, data: Dict) -> Optional[List[Dict[str, Any]]]:
        """Process a single pre-processed HOT3D temporal sequence."""
        sample_freq = int(data['mocap_frame_rate'] / self.config.TARGET_FPS)
        
        fullpose_ = data['poses'][::sample_freq]
        root_orient_ = data['root_orient'][::sample_freq]
        betas = data['betas']
        trans = data['trans'][::sample_freq]
        pose_augment_flag = data['pose_augment_flag']
        
        T = len(fullpose_)
        if T < self.config.WINDOW - self.FUTURE - self.PAST:
            return None
        
        trans_ = trans.copy()
        
        frames = torch.arange(T).to(torch.long)
        
        if T % self.config.WINDOW >= self.config.WINDOW - self.FUTURE - self.PAST:
            frames = torch.cat([
                torch.zeros(self.PAST),
                frames,
                torch.ones(self.FUTURE) * (T - 1)
            ]).to(torch.long)
        
        chunks = frames.unfold(dimension=0, size=self.config.WINDOW, step=self.config.WINDOW)
        
        assert fullpose_.shape[0] == T
        assert fullpose_.shape[1] == 45
        
        results = []
        for time_augment_flag in [True, False]:
            if time_augment_flag:
                trans_time, fullpose_time, root_orient_time = \
                    self.flip_time(trans_, fullpose_, root_orient_)
            else:
                trans_time = trans_.copy()
                fullpose_time = fullpose_.copy()
                root_orient_time = root_orient_.copy()
            
            for chunk in chunks:
                chunk_data = {
                    'betas': betas,
                    'gender': 'neutral',
                    'seq_name': self.motion_idx,
                    'datasetname': 'HOT3D',
                    'pose_augment_flag': pose_augment_flag,
                    'time_augment_flag': time_augment_flag,
                    'trans': trans_time[chunk],
                    'pose_rhand': fullpose_time[chunk],
                    'root_orient': root_orient_time[chunk],
                }
                results.append(chunk_data)
        
        return results if results else None


if __name__ == '__main__':
    processor = Hot3dHandProcessor()
    processor.run()
    print('Done processing HOT3D')
