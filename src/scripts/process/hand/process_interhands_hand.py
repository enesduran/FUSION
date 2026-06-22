"""
InterHands Hand Dataset Processor - Refactored version using unified architecture.
Multi-stage pipeline: purge → process (slerp interpolation) → data (chunking + filtering).
"""
import os
import sys
import json
import copy
import smplx
import torch
import numpy as np
from typing import Dict, List, Optional, Any

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

from src.scripts.process.base_processor import HandDatasetProcessor
from src.scripts.process.config import ProcessingConfig, DATASET_PATHS
from src.scripts.process.processor_utils import ModelLoader
from src.utils.data_utils import slerp
from src.utils.transforms import axis_angle_to_quaternion, quaternion_to_axis_angle

INTERHANDS_PATH = DATASET_PATHS.get('interhands', 'data/motion/Hand_Raw/Interhands')
RIGHT_MANO_PATH = 'data/body_models/mano/MANO_RIGHT.pkl'
CONSECUTIVE_FRAMES_THRESHOLD = 6


class InterhandsHandProcessor(HandDatasetProcessor):
    """Processor for InterHands hand motion dataset."""
    
    LOWEST_PERCENT = 0.45
    HIGHEST_PERCENT = 0.98
    
    FUTURE = 1
    PAST = 1
    
    def __init__(self):
        super().__init__(
            dataset_name='Interhands',
            dataset_path=INTERHANDS_PATH,
            output_path='data/motion/Hand_Processed/interhands_train.p'
        )
        
        # Load MANO mean pose (flat_hand_mean=False)
        self.r_mean_pose = smplx.create(
            model_path=RIGHT_MANO_PATH, model_type='mano',
            is_rhand=True, use_pca=False, flat_hand_mean=False, ext='pkl'
        ).hand_mean
    
    def _purge_sequences(self):
        """Load and separate left/right hand data from JSON annotations."""
        train_path = os.path.join(self.dataset_path, "train/InterHand2.6M_train_MANO_NeuralAnnot.json")
        test_path = os.path.join(self.dataset_path, "test/InterHand2.6M_test_MANO_NeuralAnnot.json")
        val_path = os.path.join(self.dataset_path, "val/InterHand2.6M_val_MANO_NeuralAnnot.json")
        
        split_seqs_purged_right = {'train': {}, 'val': {}, 'test': {}}
        split_seqs_purged_left = {'train': {}, 'val': {}, 'test': {}}
        
        split_paths = {'train': train_path, 'test': test_path, 'val': val_path}
        
        for split in split_seqs_purged_right.keys():
            split_30fps = json.load(open(split_paths[split], 'r'))
            split_seqs_purged_right[split] = copy.deepcopy(split_30fps)
            split_seqs_purged_left[split] = copy.deepcopy(split_30fps)
            
            for subj_id in split_30fps.keys():
                split_subj_id = split_30fps[subj_id]
                
                for k, v in split_subj_id.items():
                    if v["right"] is None:
                        split_seqs_purged_right[split][subj_id].pop(k)
                    elif len(v["right"]["pose"]) != 48:
                        split_seqs_purged_right[split][subj_id].pop(k)
                    else:
                        split_seqs_purged_right[split][subj_id][k].pop("left")
                        split_seqs_purged_right[split][subj_id][k] = \
                            split_seqs_purged_right[split][subj_id][k].pop("right")
                    
                    if v["left"] is None:
                        split_seqs_purged_left[split][subj_id].pop(k)
                    elif len(v["left"]["pose"]) != 48:
                        split_seqs_purged_left[split][subj_id].pop(k)
                    else:
                        split_seqs_purged_left[split][subj_id][k].pop("right")
                        split_seqs_purged_left[split][subj_id][k] = \
                            split_seqs_purged_left[split][subj_id][k].pop("left")
        
        return split_seqs_purged_right, split_seqs_purged_left
    
    def _process_raw_sequences(self, split_seqs_purged_right, split_seqs_purged_left):
        """Process raw frame-level data into temporal sequences with slerp interpolation."""
        split_seqs_processed = {'train': [], 'val': [], 'test': []}
        purged_list = [split_seqs_purged_right, split_seqs_purged_left]
        
        for _i_, split_seqs_purged_hand in enumerate(purged_list):
            for split_name, split_30fps in split_seqs_purged_hand.items():
                for subj_id in split_30fps.keys():
                    split_subj_id = split_30fps[subj_id]
                    
                    int_timesteps = list(map(int, list(split_subj_id.keys())))
                    int_timesteps_sorted = sorted(copy.deepcopy(int_timesteps))
                    int_timesteps_diff = np.diff(int_timesteps)
                    int_timesteps_sorted_diff = np.diff(int_timesteps_sorted)
                    
                    # Find discontinuities
                    endpoints = np.argwhere(abs(int_timesteps_diff) > CONSECUTIVE_FRAMES_THRESHOLD)[:, 0]
                    endpoints_sorted = np.argwhere(int_timesteps_sorted_diff > CONSECUTIVE_FRAMES_THRESHOLD)[:, 0]
                    
                    endpoints_ = sorted(set([0] + list(endpoints) + [len(int_timesteps) - 1]))
                    endpoints_sorted_ = sorted(set([0] + list(endpoints_sorted) + [len(int_timesteps) - 1]))
                    
                    end_pts = endpoints_sorted_
                    timesteps = int_timesteps_sorted
                    
                    for idx in range(len(end_pts) - 1):
                        start_idx, end_idx = end_pts[idx], end_pts[idx + 1]
                        
                        temporal_pose_dict, temporal_pose_list = {}, []
                        
                        for timestep in range(start_idx, end_idx):
                            t = str(timesteps[timestep])
                            temporal_pose_list.append(split_subj_id[t])
                        
                        # Concatenate
                        for keyval in ["pose", "shape", "trans"]:
                            temporal_pose_dict[keyval] = [elem[keyval] for elem in temporal_pose_list]
                        
                        # Mirror left hand to right hand convention
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
                    
                    # Slerp interpolation for the last segment
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
                    
                    split_seqs_processed[split_name].append({
                        "subject_id": subj_id,
                        "mocap_frame_rate": 30,
                        'pose_augment_flag': temporal_pose_dict['pose_augment_flag'],
                        "poses": np.array(temporal_pose_dict['pose']),
                        "betas": np.array(temporal_pose_dict['shape']),
                        "trans": np.array(temporal_pose_dict['trans']),
                        "n_frames": len(temporal_pose_dict['pose'])
                    })
        
        return split_seqs_processed
    
    def load_sequences(self) -> List[Dict]:
        """Load, purge, interpolate and merge InterHands sequences."""
        print("Purging InterHands sequences...")
        purged_right, purged_left = self._purge_sequences()
        
        print("Processing raw sequences (slerp interpolation)...")
        seqs_processed = self._process_raw_sequences(purged_right, purged_left)
        
        # Merge all splits into train
        all_seqs = (seqs_processed.get('train', []) +
                    seqs_processed.get('test', []) +
                    seqs_processed.get('val', []))
        
        return all_seqs
    
    def filter_sequences(self, sequences: List[Dict]) -> Dict[str, List[Dict]]:
        """All sequences go to train."""
        return {'train': sequences}
    
    def process_sequence(self, data: Dict) -> Optional[List[Dict[str, Any]]]:
        """Process a single pre-processed InterHands temporal sequence."""
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
        
        # Padding for boundary frames
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
                    'datasetname': 'Interhands',
                    'pose_augment_flag': pose_augment_flag,
                    'time_augment_flag': time_augment_flag,
                    'trans': trans_time[chunk],
                    'pose_rhand': fullpose_time[chunk],
                    'root_orient': root_orient_time[chunk],
                }
                results.append(chunk_data)
        
        return results if results else None


if __name__ == '__main__':
    processor = InterhandsHandProcessor()
    processor.run()
    print('Done processing InterHands')
