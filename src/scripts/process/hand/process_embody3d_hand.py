"""
EMBODY3D Hand Dataset Processor - Refactored version using unified architecture.
Extracts hand motions from EMBODY3D dataset using SMPLX FK for wrist orientation.
"""
import os
import sys
import json
import torch
import smplx
import numpy as np
from tqdm import tqdm
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

from src.scripts.process.base_processor import HandDatasetProcessor
from src.scripts.process.config import ProcessingConfig, SMPLX_JOINT_MIRROR_ARR, DATASET_PATHS
from src.scripts.process.processor_utils import ModelLoader
from src.utils.transforms import (
    axis_angle_to_matrix, matrix_to_axis_angle,
    axis_angle_to_quaternion, quaternion_to_axis_angle, quat_fk
)
from src.utils.process_utils import RIGHT_WRIST_BASE_LOC, LEFT_WRIST_BASE_LOC, SMPLX_JOINTS


class FeatName(Enum):
    """Feature names for EMBODY3D dataset."""
    BODY = "smplx_mesh_body_pose"
    ROT = "smplx_mesh_global_orient"
    TRANS = "smplx_mesh_transl"
    SHAPE = "smplx_mesh_betas"
    LEFT_HAND = "smplx_mesh_left_hand_pose"
    RIGHT_HAND = "smplx_mesh_right_hand_pose"
    RIGHT_EYE = "smplx_mesh_reye_pose"
    LEFT_EYE = "smplx_mesh_leye_pose"
    JAW = "smplx_mesh_jaw_pose"
    SUBJECT_ID = "subject_id"
    SEQUENCE_ID = "sequence_id"
    START_FRAME = "start_frame"
    LENGTH = "length"


class EMBODY3DDataset:
    """Simple dataset loader for EMBODY3D that preloads sequences into memory."""
    
    def __init__(self, split, data_dir, seq_len=120, train_ratio=0.07,
                 val_ratio=0.07, features_to_load=None):
        self.split = split
        self.seq_len = seq_len
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.features_to_load = features_to_load or []
        self.data_dir = data_dir
        self.dataset_info = self._load_and_split_dataset()
        self._create_index_mapping()
        self._preload_dataset()
    
    def _load_and_split_dataset(self):
        dataset_info_path = os.path.join(self.data_dir, "dataset.json")
        with open(dataset_info_path, "r") as f:
            full_dataset_info = json.load(f)
        
        all_subjects = list(full_dataset_info.keys())
        train_end = int(len(all_subjects) * self.train_ratio)
        val_end = train_end + int(len(all_subjects) * self.val_ratio)
        
        if self.split == "val":
            selected_subjects = all_subjects[train_end:val_end]
        elif self.split == "test":
            selected_subjects = all_subjects[val_end:]
        elif self.split == "train":
            selected_subjects = all_subjects[:train_end]
        else:
            raise ValueError(f"Invalid split: {self.split}")
        
        split_dataset = {k: full_dataset_info[k] for k in selected_subjects}
        n_seqs = sum([len(v) for v in split_dataset.values()])
        print(f"EMBODY3D {self.split}: {len(split_dataset)} subjects, {n_seqs} sequences")
        return split_dataset
    
    def _create_index_mapping(self):
        seqs = [
            (subject_id, sequence_name)
            for subject_id in self.dataset_info.keys()
            for sequence_name in self.dataset_info[subject_id].keys()
        ]
        
        idx = 0
        self.idx2seq = {}
        self.idx_per_seq = []
        
        for seq in seqs:
            self.idx2seq[idx] = seq
            self.idx_per_seq.append(idx)
            num_segments = self.dataset_info[seq[0]][seq[1]]["length"] // self.seq_len
            idx += num_segments
        
        self.num_valid_segments = idx
    
    def _preload_dataset(self):
        self.dataset = {}
        seqs = [
            (subject_id, sequence_name)
            for subject_id in self.dataset_info.keys()
            for sequence_name in self.dataset_info[subject_id].keys()
        ]
        
        for subject_id, sequence_name in tqdm(seqs, desc=f"Preloading EMBODY3D {self.split}"):
            if subject_id not in self.dataset:
                self.dataset[subject_id] = {}
            
            base_sequence_name = os.path.splitext(sequence_name)[0]
            seq_dict = {}
            
            for feat_name in self.features_to_load:
                feat_path = os.path.join(
                    self.data_dir, sequence_name, subject_id,
                    feat_name, f"{base_sequence_name}.npy"
                )
                data_feature = np.load(feat_path)
                seq_dict[feat_name] = data_feature.astype(np.float32)
            
            seq_dict['missing'] = np.load(os.path.join(self.data_dir, sequence_name, 
                subject_id, "missing", f"{base_sequence_name}.npy"))

            self.dataset[subject_id][sequence_name] = seq_dict
    
    def _idx2segment(self, idx):
        closest_idx = np.searchsorted(list(self.idx2seq.keys()), idx, side="right") - 1
        closest_idx = list(self.idx2seq.keys())[closest_idx]
        subject_id, sequence_name = self.idx2seq[closest_idx]
        frame_offset = (idx - closest_idx) * self.seq_len
        return subject_id, sequence_name, frame_offset
    
    def __len__(self):
        return self.num_valid_segments
    
    def __getitem__(self, idx):
        subject_id, sequence_name, start_frame = self._idx2segment(idx)
        end_frame = start_frame + self.seq_len
        
        data = {
            FeatName.SUBJECT_ID.value: subject_id,
            FeatName.SEQUENCE_ID.value: sequence_name,
            FeatName.START_FRAME.value: start_frame,
            FeatName.LENGTH.value: self.seq_len,
        }
        
        for feat_name in self.features_to_load:
            feat_data = self.dataset[subject_id][sequence_name][feat_name]
            data[feat_name] = torch.from_numpy(feat_data[start_frame:end_frame].copy())
        
        return data


EMBODY3D_PATH = DATASET_PATHS.get('embody3d', 'data/motion/Body_Raw/Embody3D_subset')


class Embody3dHandProcessor(HandDatasetProcessor):
    """Processor for EMBODY3D hand motion dataset."""
    
    # No motion variance filtering for EMBODY3D
    LOWEST_PERCENT = 0.15
    HIGHEST_PERCENT = 0.95
    
    SMPLX_KEYS = [
        FeatName.BODY.value, 
        FeatName.ROT.value, 
        FeatName.TRANS.value,
        FeatName.SHAPE.value, 
        FeatName.LEFT_HAND.value, 
        FeatName.RIGHT_HAND.value,
        'missing'
    ]
    
    def __init__(self):
        super().__init__(
            dataset_name='EMBODY3D',
            dataset_path=EMBODY3D_PATH,
            output_path='data/motion/Hand_Processed/embody3d_train.p'
        )
        
        self.device = self.get_device()
        self.body_models = ModelLoader.load_smplx_models(
            batch_size=self.config.WINDOW,
            num_betas=self.config.NUM_BETAS,
            device=self.device,
            config=self.config
        )
        
        # Freeze parameters
        for gender_model in self.body_models.values():
            for p in gender_model.parameters():
                p.requires_grad = False
        
        # Discover all subfolders with a dataset.json
        self.subfolders = self._discover_subfolders()
        self._train_entries = []  # list of (subfolder, dataset, local_idx)
    
    def _discover_subfolders(self) -> List[str]:
        """Find all subfolders under dataset_path that contain a dataset.json."""
        subfolders = []
        for entry in sorted(os.listdir(self.dataset_path)):
            subfolder_path = os.path.join(self.dataset_path, entry)
            if os.path.isdir(subfolder_path) and os.path.exists(
                os.path.join(subfolder_path, 'dataset.json')
            ):
                subfolders.append(entry)
        print(f"Found {len(subfolders)} EMBODY3D subfolders: {subfolders}")
        return subfolders
    
    def load_sequences(self) -> List[int]:
        """Load EMBODY3D datasets from all subfolders and return sequence indices."""
        self._train_entries = []
        
        for subfolder in self.subfolders:
            subfolder_path = os.path.join(self.dataset_path, subfolder)
            dataset_kwargs = dict(
                data_dir=subfolder_path,
                seq_len=self.config.WINDOW,
                train_ratio=0.86,
                val_ratio=0.07,
                features_to_load=self.SMPLX_KEYS
            )
            try:
                ds = EMBODY3DDataset(split='train', **dataset_kwargs)
                for local_idx in range(len(ds)):
                    self._train_entries.append((subfolder, ds, local_idx))
            except Exception as e:
                print(f"Warning: Failed to load {subfolder}/train: {e}")
                continue
        
        print(f"Total train sequences across all subfolders: {len(self._train_entries)}")
        return list(range(len(self._train_entries)))
    
    def filter_sequences(self, sequences: List[int]) -> Dict[str, List[int]]:
        """All go to train."""
        return {'train': sequences}
    
    def process_sequence(self, sequence_idx: int) -> Optional[List[Dict[str, Any]]]:
        """Process a single EMBODY3D sequence and extract hand motions."""
        subfolder, dataset, local_idx = self._train_entries[sequence_idx]
        data = dataset[local_idx]
        data['mocap_frame_rate'] = 30  # From documentation
        
        # Extract parameters
        betas = data[FeatName.SHAPE.value][:, :self.config.NUM_BETAS].float().to(self.device)
        gender = 'neutral'
        
        sample_freq = int(data['mocap_frame_rate'] / self.config.TARGET_FPS)
        trans = data[FeatName.TRANS.value][::sample_freq].to(self.device)
        

        if data['missing'].sum().item() != self.config.WINDOW:
            print(f"Skipping sequence {sequence_idx} due to missing frames")
            return None 

        # Concatenate full pose
        fullpose = torch.cat([
            data[FeatName.ROT.value],
            data[FeatName.BODY.value],
            torch.zeros((self.config.WINDOW, 3)),   # jaw pose
            torch.zeros((self.config.WINDOW, 3)),   # left eye pose
            torch.zeros((self.config.WINDOW, 3)),   # right eye pose
            data[FeatName.LEFT_HAND.value],
            data[FeatName.RIGHT_HAND.value]
        ], dim=-1)[::sample_freq].to(self.device)
        
        T = len(fullpose)
        if T < self.config.WINDOW:
            return None
        
        # Extract hand poses
        rhand_pose = fullpose[:, 120:]
        lhand_pose = fullpose[:, 75:120]
        
        # Compute rest joint positions
        bm = self.body_models[gender]

        rest_joint_pos_placeholder = torch.zeros((1, len(SMPLX_JOINT_MIRROR_ARR), 3)).to(self.device)

        
        # Compute global joint rotations using FK
        global_rot_quat, _ = quat_fk(
            axis_angle_to_quaternion(fullpose.reshape(T, -1, 3)),
            rest_joint_pos_placeholder.expand(T, -1, -1),
            bm.parents
        )
        global_rot_aa = quaternion_to_axis_angle(global_rot_quat)
        
        rhand_root_orient = global_rot_aa[:, SMPLX_JOINTS['right_wrist']]
        lhand_root_orient = global_rot_aa[:, SMPLX_JOINTS['left_wrist']]
        
        # Get wrist translations from full body output
        bm_out = bm(
            betas=betas,
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
        
        seq_name = f"{subfolder}_{str(sequence_idx).zfill(6)}"
        
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
                    trans_time, hand_pose_time, hand_orient_time = \
                        self.flip_time(hand_trans, hand_pose, hand_orient)
                else:
                    trans_time = hand_trans.copy()
                    hand_pose_time = hand_pose.copy()
                    hand_orient_time = hand_orient.copy()
                
                frames = torch.arange(T).to(torch.long)
                chunks = frames.unfold(dimension=0, size=self.config.WINDOW, step=self.config.WINDOW)
                
                for chunk in chunks:
                    chunk_data = {
                        'betas': betas.cpu().numpy(),
                        'gender': gender,
                        'seq_name': seq_name,
                        'datasetname': 'EMBODY3D',
                        'pose_augment_flag': pose_augment_flag,
                        'time_augment_flag': time_augment_flag,
                        'trans': trans_time[chunk],
                        'pose_rhand': hand_pose_time[chunk],
                        'root_orient': hand_orient_time[chunk],
                    }
                    results.append(chunk_data)
        
        return results if results else None


if __name__ == '__main__':
    processor = Embody3dHandProcessor()
    processor.run()
    print('Done processing EMBODY3D')
