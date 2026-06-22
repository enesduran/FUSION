"""
ReInterHands Hand Dataset Processor - Refactored version using unified architecture.
Multi-stage pipeline: congregate → process (slerp interpolation) → data (chunking + filtering).
"""
import os
import sys
import json
import glob
import copy
import smplx
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Any

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

from src.scripts.process.base_processor import HandDatasetProcessor
from src.scripts.process.config import ProcessingConfig, DATASET_PATHS
from src.utils.data_utils import slerp
from src.utils.transforms import axis_angle_to_quaternion, quaternion_to_axis_angle

REINTERHANDS_PATH = DATASET_PATHS.get('reinterhands', 'data/motion/Hand_Raw/Reinterhands')
RIGHT_MANO_PATH = 'data/body_models/mano/MANO_RIGHT.pkl'
CONSECUTIVE_FRAMES_THRESHOLD = 6


class ReinterhandsHandProcessor(HandDatasetProcessor):
    """Processor for ReInterHands hand motion dataset."""
    
    LOWEST_PERCENT = 0.20
    HIGHEST_PERCENT = 1.00
    
    FUTURE = 1
    PAST = 1
    
    def __init__(self):
        super().__init__(
            dataset_name='ReInterhands',
            dataset_path=REINTERHANDS_PATH,
            output_path='data/motion/Hand_Processed/reinterhands_train.p'
        )
        
        self.r_mean_pose = smplx.create(
            model_path=RIGHT_MANO_PATH, model_type='mano',
            is_rhand=True, use_pca=False, flat_hand_mean=False, ext='pkl'
        ).hand_mean
    
    def _congregate_sequences(self):
        """Load per-frame JSON data and organize by subject."""
        all_seqs = glob.glob(os.path.join(self.dataset_path, '*/mano_fits/params'))
        
        split_seqs_purged_right = {}
        split_seqs_purged_left = {}
        
        for seq in tqdm(all_seqs, desc="Congregating sequences"):

            subj_id = seq.split('/')[-3].split("--")[3]
            
            split_seqs_purged_right[subj_id] = {}
            split_seqs_purged_left[subj_id] = {}
            
            subj_seq_r = sorted(glob.glob(os.path.join(seq, '*_right.json')))
            subj_seq_l = sorted(glob.glob(os.path.join(seq, '*_left.json')))
            
            assert len(subj_seq_r) == len(subj_seq_l), "Right and left hand sequences are not equal"
            
            for subj_seq_file_r, subj_seq_file_l in zip(subj_seq_r, subj_seq_l):
                bad_condition = False
                
                with open(subj_seq_file_r, 'r') as f:
                    seq_data_r = json.load(f)
                    if seq_data_r["pose"] is None or len(seq_data_r["pose"]) != 48:
                        bad_condition = True
                
                with open(subj_seq_file_l, 'r') as f:
                    seq_data_l = json.load(f)
                    if seq_data_l["pose"] is None or len(seq_data_l["pose"]) != 48:
                        bad_condition = True
                
                if bad_condition:
                    continue
                
                frame_idx_r = int(subj_seq_file_r.split("/")[-1].split("_")[0])
                frame_idx_l = int(subj_seq_file_l.split("/")[-1].split("_")[0])
                assert frame_idx_r == frame_idx_l, "Right and left frame indices don't match"
                
                split_seqs_purged_right[subj_id][frame_idx_r] = {
                    "pose": seq_data_r["pose"],
                    "shape": seq_data_r["shape"],
                    "trans": seq_data_r["trans"]
                }
                split_seqs_purged_left[subj_id][frame_idx_l] = {
                    "pose": seq_data_l["pose"],
                    "shape": seq_data_l["shape"],
                    "trans": seq_data_l["trans"]
                }
        
        return split_seqs_purged_right, split_seqs_purged_left
    
    def _process_raw_sequences(self, split_seqs_purged_right, split_seqs_purged_left):
        """Process raw frame-level data into temporal sequences with slerp interpolation."""
        processed_list = []
        purged_list = [split_seqs_purged_right, split_seqs_purged_left]
        
        for _i_, split_seqs_purged_hand in enumerate(purged_list):
            for subj_id, split_subj_id in split_seqs_purged_hand.items():
                int_timesteps = list(map(int, list(split_subj_id.keys())))
                int_timesteps_sorted = sorted(copy.deepcopy(int_timesteps))
                
                int_timesteps_diff = np.diff(int_timesteps)
                int_timesteps_sorted_diff = np.diff(int_timesteps_sorted)
                
                endpoints = np.argwhere(abs(int_timesteps_diff) > CONSECUTIVE_FRAMES_THRESHOLD)[:, 0]
                endpoints_sorted = np.argwhere(int_timesteps_sorted_diff > CONSECUTIVE_FRAMES_THRESHOLD)[:, 0]
                
                endpoints_ = sorted(set([0] + list(endpoints) + [len(int_timesteps) - 1]))
                endpoints_sorted_ = sorted(set([0] + list(endpoints_sorted) + [len(int_timesteps) - 1]))
                
                end_pts = endpoints_sorted_
                timesteps = int_timesteps_sorted
                
                for idx in range(len(end_pts) - 1):
                    start_idx, end_idx = end_pts[idx] + 1, end_pts[idx + 1]
                    
                    if start_idx == end_idx:
                        continue
                    
                    temporal_pose_dict, temporal_pose_list = {}, []
                    
                    for timestep in range(start_idx, end_idx):
                        t = timesteps[timestep]
                        temporal_pose_list.append(split_subj_id[t])
                    
                    for keyval in ["pose", "shape", "trans"]:
                        temporal_pose_dict[keyval] = [elem[keyval] for elem in temporal_pose_list]
                    
                    # Mirror left hand
                    if _i_ == 1:
                        temporal_pose_dict['pose_augment_flag'] = True
                        temporal_pose_dict['pose'] = list(
                            (np.array(temporal_pose_dict['pose']).reshape(-1, 16, 3)
                             * np.array([[1, -1, -1]])).reshape(-1, 48)
                        )
                        temporal_pose_dict['trans'] = list(
                            np.array(temporal_pose_dict['trans']) * np.array([[-1, 1, 1]])
                        )
                    else:
                        temporal_pose_dict['pose_augment_flag'] = False
                    
                    # Slerp interpolation
                    T = timesteps[end_idx] - timesteps[start_idx]
                    quat_padded = torch.zeros((T, 16, 4))
                    trans_padded = torch.zeros((T, 3))
                    
                    known_indices = np.array(timesteps[start_idx:end_idx]) - timesteps[start_idx]
                    all_indices = np.arange(0, known_indices.max() + 1)
                    
                    if len(known_indices) > 1:
                        quat_padded[known_indices] = axis_angle_to_quaternion(
                            torch.tensor(np.array(temporal_pose_dict['pose'])).float().reshape(-1, 16, 3)
                        )
                        trans_padded[known_indices] = torch.tensor(
                            np.array(temporal_pose_dict['trans'])
                        ).float()
                        
                        slerp_quat, slerp_trans = slerp(
                            quat=quat_padded, trans=trans_padded,
                            key_times=known_indices, times=all_indices, mask=True
                        )
                        
                        temporal_pose_dict['pose'] = quaternion_to_axis_angle(slerp_quat).reshape(-1, 48)
                        temporal_pose_dict['trans'] = slerp_trans
                        temporal_pose_dict['shape'] = np.array(
                            [temporal_pose_dict['shape'][0]] * slerp_trans.shape[0]
                        )
                    
                    processed_list.append({
                        "subject_id": subj_id,
                        "mocap_frame_rate": 60,
                        'pose_augment_flag': temporal_pose_dict['pose_augment_flag'],
                        "poses": np.array(temporal_pose_dict['pose']),
                        "betas": np.array(temporal_pose_dict['shape']),
                        "trans": np.array(temporal_pose_dict['trans']),
                        "n_frames": len(temporal_pose_dict['pose'])
                    })
        
        return processed_list
    
    def load_sequences(self) -> List[Dict]:
        """Load, congregate, and interpolate ReInterHands sequences."""
        print("Congregating ReInterHands sequences...")
        purged_right, purged_left = self._congregate_sequences()
        
        print("Processing raw sequences (slerp interpolation)...")
        all_seqs = self._process_raw_sequences(purged_right, purged_left)
        
        return all_seqs
    
    def filter_sequences(self, sequences: List[Dict]) -> Dict[str, List[Dict]]:
        """All sequences go to train."""
        return {'train': sequences}
    
    def process_sequence(self, data: Dict) -> Optional[List[Dict[str, Any]]]:
        """Process a single pre-processed ReInterHands temporal sequence."""
        sample_freq = int(data['mocap_frame_rate'] / self.config.TARGET_FPS)
        
        fullpose = data['poses'][::sample_freq]
        betas = data['betas']
        trans = data['trans'][::sample_freq]
        pose_augment_flag = data['pose_augment_flag']
        
        T = len(fullpose)
        if T < self.config.WINDOW - self.FUTURE - self.PAST:
            return None
        
        trans_ = trans.copy()
        fullpose_ = fullpose.copy()[:, 3:] + self.r_mean_pose.reshape(1, -1).numpy()
        root_orient_ = fullpose.copy()[:, :3]
        
        frames = torch.arange(T).to(torch.long)
        
        if T % self.config.WINDOW >= self.config.WINDOW - self.FUTURE - self.PAST:
            frames = torch.cat([
                torch.zeros(self.PAST),
                frames,
                torch.ones(self.FUTURE) * (T - 1)
            ]).to(torch.long)
        
        chunks = frames.unfold(dimension=0, size=self.config.WINDOW, step=self.config.WINDOW)
        
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
                    'datasetname': 'Interhands',  # Same as original
                    'pose_augment_flag': pose_augment_flag,
                    'time_augment_flag': time_augment_flag,
                    'trans': trans_time[chunk],
                    'pose_rhand': fullpose_time[chunk],
                    'root_orient': root_orient_time[chunk],
                }
                results.append(chunk_data)
        
        return results if results else None


if __name__ == '__main__':
    processor = ReinterhandsHandProcessor()
    processor.run()
    print('Done processing ReInterHands')
