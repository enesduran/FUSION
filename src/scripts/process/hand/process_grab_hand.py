"""
GRAB Hand Dataset Processor - Refactored version using unified architecture.
Extracts hand motions from GRAB body-object interaction dataset.
"""
import os
import sys
import glob
import torch
import trimesh
import numpy as np
from typing import Dict, List, Optional, Any
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

from src.scripts.process.base_processor import HandDatasetProcessor
from src.scripts.process.config import ProcessingConfig, DATASET_PATHS
from src.scripts.process.processor_utils import ModelLoader
from src.utils.transforms3d import loc2vel
from src.utils.transforms import axis_angle_to_matrix


# GRAB object splits (train only for hand processing)
GRAB_SPLITS = {
    'train': [
        'airplane', 'alarmclock', 'banana', 'bowl', 'cubelarge',
        'cubemedium', 'cubesmall', 'cup', 'cylinderlarge',
        'cylindermedium', 'cylindersmall', 'doorknob', 'duck',
        'eyeglasses', 'flashlight', 'flute', 'gamecontroller', 'hammer',
        'headphones', 'knife', 'lightbulb', 'mouse', 'phone', 'piggybank',
        'pyramidlarge', 'pyramidmedium', 'pyramidsmall', 'scissors',
        'spherelarge', 'spheremedium', 'spheresmall', 'stamp',
        'stanfordbunny', 'stapler', 'teapot', 'toruslarge', 'torusmedium',
        'torussmall', 'train', 'watch', 'waterbottle', 'wineglass',
        'fryingpan', 'toothbrush', 'elephant', 'hand'
    ]
}


class GrabHandProcessor(HandDatasetProcessor):
    """Processor for GRAB hand motion dataset."""
    
    LOWEST_PERCENT = 0.15
    HIGHEST_PERCENT = 1.00
    
    def __init__(self):
        dataset_path = DATASET_PATHS.get('grab', 'data/motion/Hand_Raw/GRAB/grab')
        output_path = 'data/motion/Hand_Processed/grab_train.p'
        
        super().__init__(
            dataset_name='GRAB',
            dataset_path=dataset_path,
            output_path=output_path
        )
        
        # Object info cache
        self.obj_info = {}
    
    def load_sequences(self) -> List[str]:
        """Load all GRAB sequence files."""
        return glob.glob(os.path.join(self.dataset_path, '*/*.npz'))
    
    def filter_sequences(self, sequences: List[str]) -> Dict[str, List[str]]:
        """Filter sequences by object name into train split."""
        selected = []
        for seq in sequences:
            object_name = os.path.basename(seq).split('_')[0]
            if object_name in GRAB_SPLITS['train']:
                selected.append(seq)
        return {'train': selected}
    
    def filter_grasp_frames(self, seq_data) -> tuple:
        """Filter frames based on object contact/height and hand velocity."""
        table_height = seq_data['object'].item()['params']['transl'][0, 2]
        obj_height = seq_data['object'].item()['params']['transl'][:, 2]
        contact_array = seq_data['contact'].item()['object']
        idxs = np.arange(obj_height.shape[0])
        
        # Special objects use contact-based filtering
        if seq_data["obj_name"] in ['coffeemug', 'rubberduck', 'wristwatch', 'doorknob', 'mouse']:
            fil = (contact_array.mean(axis=1) > 0)
        else:
            fil = obj_height > (table_height + 0.004)
        
        # Hand velocity based start detection
        hand = seq_data['body'].item()['params']['body_pose'][:, 17 * 3:18 * 3]
        hand_rotmat = axis_angle_to_matrix(torch.tensor(hand))
        hand_ang_vel = loc2vel(hand_rotmat, fps=120).abs().norm(dim=-1).norm(dim=-1).squeeze()
        
        start_fil = hand_ang_vel > 0.6
        if start_fil.sum() < 2:
            return None, fil
        
        start_frame = idxs[start_fil][1]
        start_fil = idxs > start_frame
        
        if fil.sum() < 1:
            return None, fil
        
        skip_frame = int(seq_data['framerate'].item() // self.config.TARGET_FPS)
        fps_fil = (idxs % skip_frame) == 0
        
        grasp_motion_frames = fps_fil * start_fil
        obj_moving_frames = fil * fps_fil
        
        return grasp_motion_frames, obj_moving_frames
    
    def load_and_simplify_obj_verts(self, seq_data, n_verts_sample=2048, scale=1):
        """Load and simplify object mesh with caching."""
        obj_name = seq_data['obj_name'].item()
        mesh_path = os.path.join(self.dataset_path, seq_data['object'].item()['object_mesh'])
        
        if obj_name not in self.obj_info:
            np.random.seed(100)
            obj_mesh = trimesh.load(file_obj=mesh_path)
            verts_obj = np.array(obj_mesh.vertices)
            faces_obj = np.array(obj_mesh.faces)
            
            n_faces = max(faces_obj.shape[0] // 10, n_verts_sample)
            mesh_simplified = obj_mesh.simplify_quadric_decimation(n_faces)
            
            simple_mesh_path = mesh_path.replace('contact_meshes', 'contact_meshes_simplified')
            os.makedirs(os.path.dirname(simple_mesh_path), exist_ok=True)
            mesh_simplified.export(simple_mesh_path)
            
            verts_obj *= scale
            obj_com_offset = verts_obj.mean(axis=0)
            
            self.obj_info[obj_name] = {
                'verts': verts_obj,
                'obj_com_offset': obj_com_offset,
                'faces': faces_obj,
                'verts_sample': mesh_simplified.vertices,
                'faces_sample': mesh_simplified.faces,
                'obj_mesh_file': mesh_path
            }
        
        return self.obj_info[obj_name]
    
    def process_sequence(self, data_path: str) -> Optional[List[Dict[str, Any]]]:
        """Process a single GRAB sequence for hand data."""
        data = np.load(data_path, allow_pickle=True)
        
        # Get hand vtemp paths
        rhand_vtemp_path = data["rhand"].item()['vtemp']
        lhand_vtemp_path = data["lhand"].item()['vtemp']
        
        # Get relative wrist orientations from body pose
        r_relative_wrist_orient = data["body"].item()['params']['body_pose'][:, -3:]
        l_relative_wrist_orient = data["body"].item()['params']['body_pose'][:, -6:-3]
        
        seq_name = f'{self.motion_idx:06d}_{data["sbj_id"]}_{data["obj_name"]}_{data["motion_intent"]}'
        
        # Filter grasp frames
        grasp_motion_frames, _ = self.filter_grasp_frames(data)
        if grasp_motion_frames is None:
            return None
        
        # Extract hand data
        rhand_transl = data["rhand"].item()['params']['transl'][grasp_motion_frames]
        rhand_pose = data["rhand"].item()['params']['fullpose'][grasp_motion_frames]
        rhand_orient = data["rhand"].item()['params']['global_orient'][grasp_motion_frames]
        
        lhand_transl = data["lhand"].item()['params']['transl'][grasp_motion_frames]
        lhand_pose = data["lhand"].item()['params']['fullpose'][grasp_motion_frames]
        lhand_orient = data["lhand"].item()['params']['global_orient'][grasp_motion_frames]
        
        betas = np.zeros(10)
        
        assert rhand_pose.shape == lhand_pose.shape
        assert rhand_transl.shape == lhand_transl.shape
        
        # Object data
        object_mesh_path = data['object'].item()['object_mesh']
        obj_rot = data['object'].item()['params']['global_orient'][grasp_motion_frames]
        obj_trans = data['object'].item()['params']['transl'][grasp_motion_frames]
        
        self.load_and_simplify_obj_verts(data)
        
        T = sum(grasp_motion_frames)
        if T < self.config.WINDOW:
            return None
        
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
                
                assert fullpose_.shape[0] == T
                assert fullpose_.shape[1] == 45
                
                for chunk in chunks:
                    chunk_data = {
                        'betas': betas,
                        'gender': 'neutral',
                        'seq_name': seq_name,
                        'datasetname': 'GRAB',
                        'pose_augment_flag': pose_augment_flag,
                        'time_augment_flag': time_augment_flag,
                        'trans': trans_time[chunk],
                        'pose_rhand': fullpose_time[chunk],
                        'root_orient': root_orient_time[chunk],
                        'relative_wrist_orient': rel_wrist_time[chunk],
                        'obj_trans': obj_trans[chunk],
                        'obj_orient': obj_rot[chunk],
                        'obj_mesh': object_mesh_path,
                        'rhand_vtemp': rhand_vtemp_path,
                        'lhand_vtemp': lhand_vtemp_path,
                    }
                    results.append(chunk_data)
        
        return results if results else None


if __name__ == '__main__':
    processor = GrabHandProcessor()
    processor.run()
    print('Done processing GRAB')
