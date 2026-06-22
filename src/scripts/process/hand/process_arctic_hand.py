"""
ARCTIC Hand Dataset Processor - Refactored version using unified architecture.
Extracts hand motions from ARCTIC body-object interaction dataset.
"""
import os
import sys
import json
import smplx
import torch
import trimesh
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

from src.scripts.process.base_processor import HandDatasetProcessor
from src.scripts.process.config import ProcessingConfig, DATASET_PATHS
from src.scripts.process.processor_utils import ModelLoader
from src.utils.transforms3d import loc2vel
from src.utils.transforms import axis_angle_to_matrix

ARCTIC_PATH = DATASET_PATHS.get('arctic', 'data/motion/Hand_Raw/ARCTIC')
RIGHT_MANO_PATH = 'data/body_models/mano/MANO_RIGHT.pkl'
LEFT_MANO_PATH = 'data/body_models/mano/MANO_LEFT.pkl'


class ArcticHandProcessor(HandDatasetProcessor):
    """Processor for ARCTIC hand motion dataset."""
    
    LOWEST_PERCENT = 0.15
    HIGHEST_PERCENT = 1.00
    
    def __init__(self):
        super().__init__(
            dataset_name='ARCTIC',
            dataset_path=ARCTIC_PATH,
            output_path='data/motion/Hand_Processed/arctic_train.p'
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
        
        # Subject info cache for hand vtemplates
        self.sbj_info = {}
        
        # Subject metadata
        self.subject_metadata = self._load_subject_metadata()
    
    def _load_subject_metadata(self) -> Dict:
        """Load ARCTIC subject metadata."""
        metadata_path = os.path.join(
            self.dataset_path, 'data/arctic_data/data/meta/misc.json'
        )
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}
    
    def load_sbj_verts(self, subj_id, betas_r, betas_l):
        """Load or generate subject-specific hand vtemplates."""
        if subj_id not in self.sbj_info:
            sbj_rh_gt = smplx.create(
                model_path=RIGHT_MANO_PATH, model_type='mano',
                is_rhand=True, v_template=None, use_pca=False,
                flat_hand_mean=True, batch_size=1, ext='pkl'
            )
            sbj_lh_gt = smplx.create(
                model_path=LEFT_MANO_PATH, model_type='mano',
                is_rhand=False, v_template=None, use_pca=False,
                flat_hand_mean=True, batch_size=1, ext='pkl'
            )
            
            vtemp_dir = f"{self.dataset_path}/data/arctic_data/data/meta/hand_vtemplates"
            os.makedirs(vtemp_dir, exist_ok=True)
            
            rh_vtemp = sbj_rh_gt(betas=torch.tensor(betas_r[None])).vertices[0].detach().numpy()
            lh_vtemp = sbj_lh_gt(betas=torch.tensor(betas_l[None])).vertices[0].detach().numpy()
            
            rh_path = f"{vtemp_dir}/{subj_id}_rh.obj"
            lh_path = f"{vtemp_dir}/{subj_id}_lh.obj"
            
            trimesh.Trimesh(vertices=rh_vtemp, faces=sbj_rh_gt.faces).export(rh_path)
            trimesh.Trimesh(vertices=lh_vtemp, faces=sbj_lh_gt.faces).export(lh_path)
            
            self.sbj_info[subj_id] = [rh_path, lh_path]
        
        return self.sbj_info[subj_id]
    
    @staticmethod
    def filter_grasp_frames(seq_data) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Filter frames based on hand velocity."""
        len_motion = seq_data['params']['pose_r'].shape[0]
        idxs = np.arange(len_motion)
        fil = np.array([True] * len_motion)
        
        hand = seq_data['params']['smplx_body_pose'][:, 17 * 3:18 * 3]
        hand_rotmat = axis_angle_to_matrix(torch.tensor(hand))
        hand_ang_vel = loc2vel(hand_rotmat, fps=30).abs().norm(dim=-1).norm(dim=-1).squeeze()
        
        start_fil = hand_ang_vel > 0.5
        if start_fil.sum() < 2:
            return None, fil
        
        start_frame = idxs[start_fil][1]
        start_fil = idxs > start_frame
        
        if fil.sum() < 1:
            return None, fil
        
        skip_frame = int(30. / 30)
        fps_fil = (idxs % skip_frame) == 0
        
        grasp_motion_frames = fps_fil * start_fil
        obj_moving_frames = fps_fil * fil
        
        return grasp_motion_frames, obj_moving_frames
    
    def load_sequences(self) -> List[Tuple[str, Dict]]:
        """Load ARCTIC sequences from split files (train only)."""
        sequences = []
        
        # we only use train split to prevent data leakage.
        for split_name in ['train']:
            split_path = os.path.join(
                self.dataset_path,
                f'data/arctic_data/data/splits/p1_{split_name}.npy'
            )
            if not os.path.exists(split_path):
                print(f"Warning: Split file not found: {split_path}")
                continue
            
            split_data = np.load(split_path, allow_pickle=True).item()['data_dict']
            for datapath, data in split_data.items():
                sequences.append((datapath, data))
        
        return sequences
    
    def filter_sequences(self, sequences: List[Tuple]) -> Dict[str, List[Tuple]]:
        """All sequences go to train."""
        return {'train': sequences}
    
    def process_sequence(self, sequence_data: Tuple) -> Optional[List[Dict[str, Any]]]:
        """Process a single ARCTIC sequence for hand data."""
        datapath, data = sequence_data
        subj_id = datapath.split('/')[0]
        gender = self.subject_metadata.get(subj_id, {}).get('gender', 'neutral')
        
        try:
            grasp_motion_frames, _ = self.filter_grasp_frames(data)
        except Exception:
            return None
        
        if grasp_motion_frames is None:
            return None
        
        # Extract hand data with mean pose correction
        rhand_transl = data['params']['trans_r'][grasp_motion_frames]
        rhand_pose = data['params']['pose_r'][grasp_motion_frames] + self.r_mean_pose[None, :]
        rhand_orient = data['params']['rot_r'][grasp_motion_frames]
        
        lhand_transl = data['params']['trans_l'][grasp_motion_frames]
        lhand_pose = data['params']['pose_l'][grasp_motion_frames] + self.l_mean_pose[None, :]
        lhand_orient = data['params']['rot_l'][grasp_motion_frames]
        
        r_relative_wrist_orient = data['params']['smplx_body_pose'][:, -3:]
        l_relative_wrist_orient = data['params']['smplx_body_pose'][:, -6:-3]
        
        assert rhand_pose.shape == lhand_pose.shape
        assert rhand_transl.shape == lhand_transl.shape
        
        # Load subject hand vtemplates
        rhand_vtemp_path, lhand_vtemp_path = self.load_sbj_verts(
            subj_id,
            data['params']['shape_r'].mean(0),
            data['params']['shape_l'].mean(0)
        )
        
        T = len(rhand_pose)
        if T < self.config.WINDOW:
            return None
        
        # Object data
        obj_rot = data['params']['obj_rot'][grasp_motion_frames]
        obj_trans = data['params']['obj_trans'][grasp_motion_frames]
        object_mesh_path = 0  # Placeholder as in original
        
        results = []
        
        for pose_augment_flag in [True, False]:
            if pose_augment_flag:
                trans_, fullpose_, root_orient_, extra = self.mirror_left_to_right(
                    lhand_transl, lhand_pose, lhand_orient,
                    relative_wrist_orient=l_relative_wrist_orient
                )
                relative_wrist_orient_ = extra['relative_wrist_orient']
            else:
                trans_ = rhand_transl.copy()
                fullpose_ = rhand_pose.copy()
                root_orient_ = rhand_orient.copy()
                relative_wrist_orient_ = r_relative_wrist_orient.copy()
            
            for time_augment_flag in [True, False]:
                if time_augment_flag:
                    trans_time, fullpose_time, root_orient_time, rel_wrist_time = \
                        self.flip_time(trans_, fullpose_, root_orient_, relative_wrist_orient_)
                else:
                    trans_time = trans_.copy()
                    fullpose_time = fullpose_.copy()
                    root_orient_time = root_orient_.copy()
                    rel_wrist_time = relative_wrist_orient_.copy()
                
                frames = torch.arange(T).to(torch.long)
                chunks = frames.unfold(dimension=0, size=self.config.WINDOW, step=self.config.WINDOW)
                
                for chunk in chunks:
                    chunk_data = {
                        'betas': np.zeros(10),
                        'gender': gender,
                        'seq_name': datapath,
                        'datasetname': 'ARCTIC',
                        'pose_augment_flag': pose_augment_flag,
                        'time_augment_flag': time_augment_flag,
                        'trans': trans_time[chunk],
                        'pose_rhand': fullpose_time[chunk],
                        'root_orient': root_orient_time[chunk],
                        'relative_wrist_orient': rel_wrist_time[chunk],
                        'obj_trans': obj_trans[chunk] / 1000,
                        'obj_orient': obj_rot[chunk],
                        'obj_mesh': object_mesh_path,
                        'rhand_vtemp': rhand_vtemp_path,
                        'lhand_vtemp': lhand_vtemp_path,
                    }
                    results.append(chunk_data)
        
        return results if results else None


if __name__ == '__main__':
    processor = ArcticHandProcessor()
    processor.run()
    print('Done processing ARCTIC')
