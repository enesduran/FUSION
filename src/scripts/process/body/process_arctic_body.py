"""
ARCTIC Dataset Processor - Refactored version using new architecture.
ARCTIC contains body-object interaction with articulated objects and subject-specific body templates.
"""
import os
import sys
import json
import torch
import trimesh
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

from src.scripts.process.base_processor import BodyDatasetProcessor
from src.scripts.process.config import (
    ProcessingConfig, 
    SMPLX_JOINT_MIRROR_ARR, 
    DATASET_PATHS
)
from src.scripts.process.processor_utils import ModelLoader, SequenceChunker
from src.utils.process_utils import BRANCH_NAME, determine_floor_height_and_contacts
from src.utils.transforms3d import loc2vel
from src.utils.transforms import axis_angle_to_matrix


class ArcticBodyProcessor(BodyDatasetProcessor):
    """Processor for ARCTIC body motion dataset (body-only, no object data)."""
    
    def __init__(self):
        dataset_path = DATASET_PATHS.get('arctic', 'data/motion/Hand_Raw/ARCTIC')
        output_path = 'data/motion/Body_Processed/arctic.p'
        
        super().__init__(
            dataset_name='ARCTIC',
            dataset_path=dataset_path,
            output_path=output_path
        )
        
        # Initialize models
        self.device = self.get_device()
        self.body_models = ModelLoader.load_smplx_models(
            batch_size=self.config.WINDOW,
            num_betas=self.config.NUM_BETAS,
            device=self.device,
            config=self.config
        )
        self.default_vtemplates = ModelLoader.get_default_vtemplates(self.body_models)
        self.renderer = ModelLoader.create_renderer(self.config)
        
        # Load MANO mean poses
        self.r_mean_pose, self.l_mean_pose = ModelLoader.load_mano_mean_poses(self.config)
        
        # Load subject metadata
        self.subject_metadata = self._load_subject_metadata()
    
    def _load_subject_metadata(self) -> Dict:
        """Load ARCTIC subject metadata (gender, etc.)."""
        metadata_path = os.path.join(
            self.dataset_path,
            'data/arctic_data/data/meta/misc.json'
        )
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}
    
    def cleanup_data(self):
        """ARCTIC doesn't require specific cleanup."""
        print('No cleanup needed for ARCTIC')
    
    def load_sequences(self) -> List[Tuple[str, str, Dict]]:
        """
        Load ARCTIC sequences from split files.
        Returns list of (split_name, datapath, data) tuples.
        """
        sequences = []
        
        # Define splits
        splits = {
            'train': os.path.join(self.dataset_path, 'data/arctic_data/data/splits/p1_train.npy'),
            'test': os.path.join(self.dataset_path, 'data/arctic_data/data/splits/p1_val.npy')
        }
        
        for split_name, split_path in splits.items():
            if not os.path.exists(split_path):
                print(f"Warning: Split file not found: {split_path}")
                continue
            
            # Load split data
            split_data = np.load(split_path, allow_pickle=True).item()['data_dict']
            
            # Each entry in split_data is a datapath -> data mapping
            for datapath, data in split_data.items():
                sequences.append((split_name, datapath, data))
        
        return sequences
    
    def filter_sequences(self, sequences: List[Tuple]) -> Dict[str, List[Tuple]]:
        """
        Filter sequences by split name.
        ARCTIC sequences are already pre-split.
        """
        filtered = {}
        
        for split_name, datapath, data in sequences:
            if split_name not in filtered:
                filtered[split_name] = []
            filtered[split_name].append((datapath, data))
        
        return filtered
    
    def filter_grasp_frames(self, seq_data: Dict) -> Optional[np.ndarray]:
        """
        Filter frames based on hand velocity to find grasp motion.
        
        Returns:
            grasp_motion_frames array or None if invalid
        """
        try:
            len_motion = seq_data['params']['pose_r'].shape[0]
            idxs = np.arange(len_motion)
            
            # Hand velocity based filtering
            hand = seq_data['params']['smplx_body_pose'][:, 17 * 3:18 * 3]
            hand_rotmat = axis_angle_to_matrix(torch.tensor(hand))
            hand_ang_vel = loc2vel(hand_rotmat, fps=30).abs().norm(dim=-1).norm(dim=-1).squeeze()
            
            # Find start frame based on velocity
            start_fil = hand_ang_vel > 0.5
            if start_fil.sum() < 2:
                return None
            
            start_frame = idxs[start_fil][1]
            start_fil = idxs > start_frame
            
            # Downsample to target FPS
            skip_frame = int(30.0 / self.config.TARGET_FPS)
            fps_fil = (idxs % skip_frame) == 0
            
            grasp_motion_frames = fps_fil * start_fil
            
            return grasp_motion_frames
            
        except Exception as e:
            print(f"Error filtering grasp frames: {e}")
            return None
    
    def process_sequence(self, sequence_data: Tuple) -> Optional[List[Dict[str, Any]]]:
        """
        Process a single ARCTIC sequence.
        
        Args:
            sequence_data: Tuple of (datapath, data)
        """
        datapath, data = sequence_data
        
        # Extract subject and object info from datapath
        subj_id = datapath.split('/')[0]
        object_name = datapath.split('/')[1].split('_')[0]
        
        # Filter grasp frames
        grasp_motion_frames = self.filter_grasp_frames(data)
        if grasp_motion_frames is None:
            return None
        
        T = sum(grasp_motion_frames)
        if T < self.config.WINDOW:
            return None
        
        # Get gender from metadata
        gender = self.subject_metadata.get(subj_id, {}).get('gender', 'neutral')
        
        # Extract motion data
        transl = data['params']['smplx_transl'][grasp_motion_frames]
        
        # Concatenate full pose
        fullpose = np.concatenate([
            data['params']['smplx_global_orient'],
            data['params']['smplx_body_pose'],
            data['params']['smplx_jaw_pose'],
            data['params']['smplx_leye_pose'],
            data['params']['smplx_reye_pose'],
            data['params']['smplx_left_hand_pose'],
            data['params']['smplx_right_hand_pose']
        ], axis=1)
        
        # Get paths
        body_vtemp_path = os.path.join(
            self.dataset_path,
            f'data/arctic_data/data/meta/subject_vtemplates/{subj_id}.obj'
        )
        object_mesh_path = os.path.join(
            self.dataset_path,
            f'data/arctic_data/data/meta/object_vtemplates/{object_name}/mesh.obj'
        )


        bm_rest_out = self.body_models[gender](
            v_template= torch.from_numpy(trimesh.load(body_vtemp_path).vertices
            ).float().to(self.device)
        )
        rest_joint_pos = bm_rest_out.joints.cpu().detach().numpy()[0, :len(SMPLX_JOINT_MIRROR_ARR)]
        root_offset = rest_joint_pos[0]
        
        # Compute joint offsets
        pos_offset = [[0, 0, 0]]
        for child, parent in enumerate(self.body_models[gender].parents):
            if parent == -1:
                continue
            pos_offset.append(rest_joint_pos[child] - rest_joint_pos[parent])
        pos_offset = np.vstack(pos_offset)
        
        # Extract object data
        obj_trans = data['params']['obj_trans'][grasp_motion_frames]
        obj_rot = data['params']['obj_rot'][grasp_motion_frames]
        obj_arti = data['params']['obj_arti'][grasp_motion_frames]
        
        # Process both original and augmented versions
        results = []
        for augment_flag in [False, True]:
            result = self._process_single_variant(
                fullpose, transl, obj_trans, obj_rot, obj_arti,
                grasp_motion_frames, gender, datapath, subj_id, object_name,
                body_vtemp_path, object_mesh_path, augment_flag, pos_offset, root_offset, T
            )
            if result:
                results.extend(result)
                        
        return results if results else None
            

    
    def _process_single_variant(
        self,
        fullpose: np.ndarray,
        transl: np.ndarray,
        obj_trans: np.ndarray,
        obj_rot: np.ndarray,
        obj_arti: np.ndarray,
        grasp_motion_frames: np.ndarray,
        gender: str,
        datapath: str,
        subj_id: str,
        object_name: str,
        body_vtemp_path: str,
        object_mesh_path: str,
        augment_flag: bool,
        pos_offset: np.ndarray,
        root_offset: np.ndarray,
        T: int
    ) -> Optional[List[Dict[str, Any]]]:
        """Process original or augmented version of a sequence."""
        
        # Apply augmentation if needed
        if augment_flag:
            trans_ = transl.copy()
            trans_[:, 0] *= -1
            
            fullpose_ = fullpose.copy().reshape(-1, len(SMPLX_JOINT_MIRROR_ARR), 3)
            fullpose_ = fullpose_[:, SMPLX_JOINT_MIRROR_ARR]
            fullpose_[..., 1:] *= -1
            fullpose_ = fullpose_.reshape(-1, len(SMPLX_JOINT_MIRROR_ARR) * 3)
            
            # Object YZ-plane reflection: negate X of translation,
            # conjugate rotation by M = diag(-1,1,1).  At load time
            # load_object_geometry with reflect_x=True undoes the
            # augmentation, runs the articulated forward pass with
            # original params, then reflects the output vertices.
            obj_trans_ = obj_trans.copy()
            obj_trans_[:, 0] *= -1
            obj_rot_ = obj_rot.copy()
            obj_rot_[:, 1:] *= -1
            obj_arti_ = obj_arti.copy()
        else:
            trans_ = transl.copy()
            fullpose_ = fullpose.copy()
            obj_trans_ = obj_trans.copy()
            obj_rot_ = obj_rot.copy()
            obj_arti_ = obj_arti.copy()
        
        # Split into body parts
        root_orient_ = fullpose_[..., :3]
        pose_body_ = fullpose_[..., 3:66]
        pose_jaw_ = fullpose_[..., 66:69]
        pose_eye_ = fullpose_[..., 69:75]
        pose_lhand_ = fullpose_[..., 75:120]
        pose_rhand_ = fullpose_[..., 120:]
        
        # ARCTIC uses default betas (subject-specific shape is in vtemp)
        betas = np.zeros(self.config.NUM_BETAS)
        
        # Chunk sequence into windows
        chunks = SequenceChunker.chunk_sequence(T, self.config.WINDOW, overlap=0)
        
        results = []
        for start, end in chunks:
            chunk_slice = slice(start, end)
            chunk_len = end - start
            
            if chunk_len < self.config.WINDOW:
                continue
            
            # Prepare motion parameters
            motion_params_gt = {
                'betas': torch.from_numpy(betas[None, :]).float().to(self.device),
                'global_orient': torch.from_numpy(root_orient_[chunk_slice]).float().to(self.device),
                'body_pose': torch.from_numpy(pose_body_[chunk_slice]).float().to(self.device),
                'left_hand_pose': torch.from_numpy(pose_lhand_[chunk_slice]).float().to(self.device),
                'right_hand_pose': torch.from_numpy(pose_rhand_[chunk_slice]).float().to(self.device),
                'jaw_pose': torch.from_numpy(pose_jaw_[chunk_slice]).float().to(self.device),
                'leye_pose': torch.from_numpy(pose_eye_[chunk_slice][:, :3]).float().to(self.device),
                'reye_pose': torch.from_numpy(pose_eye_[chunk_slice][:, 3:]).float().to(self.device),
                'transl': torch.from_numpy(trans_[chunk_slice]).float().to(self.device)
            }
            
            # Update body template for subject-specific shape
            self.body_models['neutral'].v_template = torch.from_numpy(
                trimesh.load(body_vtemp_path).vertices
            ).float().to(self.device)
            
            # Generate mesh
            bm_out = self.body_models[gender](**motion_params_gt)
            joints = bm_out.joints.cpu().detach().numpy()[:, :len(SMPLX_JOINT_MIRROR_ARR)]
            
            # Determine contacts
            offset_floor_height, contacts, discard_seq = \
                determine_floor_height_and_contacts(joints, self.config.TARGET_FPS)
            
            if discard_seq:
                continue
            
            # Get grasp_motion_frames indices for this chunk
            grasp_indices = np.where(grasp_motion_frames)[0]
            chunk_grasp_slice = slice(start, end)
            
            # Store processed data
            chunk_data = {
                'betas': betas,
                'gender': gender,
                'seq_name': datapath,
                'body_dataset_name': 'ARCTIC',
                'motion_no': self.motion_idx,
                'augment_flag': augment_flag,
                'body_vtemp': body_vtemp_path,

                'trans': trans_[chunk_slice],
                'root_orient': root_orient_[grasp_indices[chunk_grasp_slice]],
                'pose_eye': pose_eye_[grasp_indices[chunk_grasp_slice]],
                'pose_jaw': pose_jaw_[grasp_indices[chunk_grasp_slice]],
                'pose_body': pose_body_[grasp_indices[chunk_grasp_slice]],
                'pose_lhand': pose_lhand_[grasp_indices[chunk_grasp_slice]],
                'pose_rhand': pose_rhand_[grasp_indices[chunk_grasp_slice]],
                
                'contacts_mask': contacts,
                'pos_offset': pos_offset,
                'root_offset': root_offset,

                # Object-specific data
                'obj_name': object_name,
                'obj_trans': obj_trans_[chunk_slice] / 1000.0,  # Convert to meters
                'obj_orient': obj_rot_[chunk_slice],
                'obj_mesh': object_mesh_path,
                'obj_arti': obj_arti_[chunk_slice],
                'obj_scale': np.ones(self.config.WINDOW)  # Placeholder, ARCTIC doesn't have scale variation
            }
            
            results.append(chunk_data)
            self.motion_idx += 1
        
        # Restore default template
        self.body_models['neutral'].v_template = self.default_vtemplates['neutral']
        
        return results if results else None
    
    def _visualize_sequence(
        self,
        motion_params_gt: Dict[str, torch.Tensor],
        vertices: np.ndarray,
        seq_name: str,
        dataset_name: Optional[str] = None,
        root_orient: Optional[np.ndarray] = None,
        trans: Optional[np.ndarray] = None
    ):
        """Visualize a sequence for debugging."""
        dataset_name = dataset_name or self.dataset_name
        filename = f"fusion_runs/{BRANCH_NAME}/body_dataset_vis/{dataset_name}/{seq_name}"
        if filename.endswith('.npz') or filename.endswith('.pkl'):
            filename = filename[:-4]
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        mesh_dict = {
            'transl': motion_params_gt['transl'].cpu(),
            'global_orient': motion_params_gt['global_orient'].cpu(),
            'faces': self.body_models['neutral'].faces,
            'vertices': vertices
        }
        
        # Use provided root_orient/trans if available, otherwise use from motion_params
        camera_dict = {
            'camera_rot': torch.from_numpy(root_orient).float() if root_orient is not None else motion_params_gt['global_orient'].cpu(),
            'camera_transl': torch.from_numpy(trans).float() if trans is not None else motion_params_gt['transl'].cpu(),
            'coef': 1.9
        }
        
        self.renderer.render_motion(
            mesh_dict, filename,
            camera_dict=camera_dict,
            color=(255/255, 160/255, 0/255, 1)
        )
    
    def add_to_data_dict(self, processed_data: Any):
        """Override to handle list of chunks."""
        if isinstance(processed_data, list):
            for chunk_data in processed_data:
                super().add_to_data_dict(chunk_data)
        else:
            super().add_to_data_dict(processed_data)


if __name__ == '__main__':
    processor = ArcticBodyProcessor()
    processor.run()
    print('Done processing ARCTIC')
