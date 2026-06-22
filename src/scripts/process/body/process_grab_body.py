"""
GRAB Dataset Processor - Refactored version using new architecture.
"""
import os
import sys
import glob
import torch
import trimesh
import numpy as np
from typing import Dict, List, Optional, Any

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

from src.scripts.process.base_processor import BodyDatasetProcessor
from src.scripts.process.config import ProcessingConfig, SMPLX_JOINT_MIRROR_ARR, SMPLX_JOINT_MIRROR_DICT, DATASET_PATHS
from src.scripts.process.processor_utils import ModelLoader, ObjectLoader, SequenceChunker
from src.utils.process_utils import BRANCH_NAME, CONTACT_INDICES, determine_floor_height_and_contacts
from src.utils.transforms import axis_angle_to_matrix


from src.utils.data_utils import apply_transformation_to_obj_geometry as grab_obj_forward_method

# GRAB object splits
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
    ],
    'test': ['mug', 'camera', 'binoculars', 'apple', 'toothpaste']
}


class GrabProcessor(BodyDatasetProcessor):
    """Processor for GRAB body-object interaction dataset."""
    
    def __init__(self):
        dataset_path = DATASET_PATHS.get('grab', 'data/motion/Hand_Raw/GRAB/grab')
        output_path = 'data/motion/Body_Processed/grab.p'
        
        super().__init__(
            dataset_name='GRAB',
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
        
        # Object loader with caching
        self.obj_loader = ObjectLoader()
    
    def load_sequences(self) -> List[str]:
        """Load all GRAB sequence files."""
        return glob.glob(os.path.join(self.dataset_path, '*/*.npz'))
    
    def filter_sequences(self, sequences: List[str]) -> Dict[str, List[str]]:
        """Filter sequences by object type into train/test splits."""
        filtered = {'train': [], 'test': []}
        
        for sequence in sequences:
            # Extract object name from filename
            action_name = os.path.basename(sequence)
            object_name = action_name.split('_')[0]
            
            # Assign to split based on object
            for split_name, split_objects in GRAB_SPLITS.items():
                if object_name in split_objects:
                    filtered[split_name].append(sequence)
                    break
        
        return filtered
    
    def filter_grasp_frames(self, seq_data: Dict) -> tuple:
        """
        Filter frames to get grasp motion.
        
        Returns:
            (grasp_motion_frames, obj_moving_frames) or (None, frames)
        """
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
        hand_ang_vel = hand_rotmat.diff(dim=0).abs().norm(dim=-1).norm(dim=-1)
        
        from src.utils.transforms3d import loc2vel
        hand_ang_vel = loc2vel(hand_rotmat, fps=120).abs().norm(dim=-1).norm(dim=-1).squeeze()
        
        start_fil = hand_ang_vel > 0.6
        if start_fil.sum() < 2:
            return None, fil
        
        start_frame = idxs[start_fil][1]
        start_fil = idxs > start_frame
        
        if fil.sum() < 1:
            return None, fil
        
        # Downsample to target FPS
        skip_frame = int(seq_data['framerate'].item() // self.config.TARGET_FPS)
        fps_fil = (idxs % skip_frame) == 0
        
        grasp_motion_frames = fps_fil * start_fil
        obj_moving_frames = fil * fps_fil
        
        return grasp_motion_frames, obj_moving_frames
    
    def process_sequence(self, sequence_path: str) -> Optional[List[Dict[str, Any]]]:
        """Process a single GRAB sequence file."""
        try:
            data = np.load(sequence_path, allow_pickle=True)
            
            # Filter frames
            grasp_motion_frames, obj_moving_frames = self.filter_grasp_frames(data)
            if grasp_motion_frames is None:
                return None
            
            # Extract basic info
            object_name = data['obj_name'].item()
            gender = data['gender'].item()
            betas = np.zeros(16)  # GRAB uses default betas
            
            # Get paths
            object_mesh_path = os.path.join(
                self.dataset_path,
                data["object"].item()["object_mesh"]
            ).replace('contact_meshes', 'contact_meshes_simplified')
            
            body_vtemp_path = os.path.join(
                self.dataset_path,
                data["body"].item()["vtemp"]
            )
            
            # Load and cache object mesh
            self.obj_loader.load_and_simplify_mesh(
                mesh_path=object_mesh_path,
                cache_key=object_name
            )
            
            # Extract motion data
            trans = data['body'].item()['params']['transl'][grasp_motion_frames]
            fullpose = data['body'].item()['params']['fullpose'][grasp_motion_frames]
            obj_trans = data['object'].item()['params']['transl'][grasp_motion_frames]
            obj_rot = -data['object'].item()['params']['global_orient'][grasp_motion_frames]
            obj_moving_frames = obj_moving_frames[grasp_motion_frames]
            
            T = sum(grasp_motion_frames)
            
            # Skip short sequences
            if T < self.config.WINDOW:
                return None
            
            # Process both original and augmented versions
            results = []
            for augment_flag in [False, True]:
                result = self._process_single_variant(
                    fullpose, trans, obj_trans, obj_rot, obj_moving_frames,
                    betas, gender, object_name, object_mesh_path, 
                    body_vtemp_path, augment_flag, T
                )
                if result:
                    results.extend(result)
            
            return results if results else None
            
        except Exception as e:
            print(f"Error processing {sequence_path}: {e}")
            return None
    
    def _process_single_variant(
        self,
        fullpose: np.ndarray,
        trans: np.ndarray,
        obj_trans: np.ndarray,
        obj_rot: np.ndarray,
        obj_moving_frames: np.ndarray,
        betas: np.ndarray,
        gender: str,
        object_name: str,
        object_mesh_path: str,
        body_vtemp_path: str,
        augment_flag: bool,
        T: int
    ) -> Optional[List[Dict[str, Any]]]:
        """Process original or augmented version of a sequence."""
        
        # Apply augmentation if needed (mirror across YZ plane)
        if augment_flag:
            trans_ = trans.copy()
            trans_[:, 0] *= -1
            
            fullpose_ = fullpose.copy().reshape(-1, len(SMPLX_JOINT_MIRROR_DICT), 3)
            fullpose_ = fullpose_[:, SMPLX_JOINT_MIRROR_ARR]
            fullpose_[..., 1:] *= -1
            fullpose_ = fullpose_.reshape(-1, len(SMPLX_JOINT_MIRROR_DICT) * 3)
            
            # Object YZ-plane reflection: negate X of translation,
            # conjugate rotation by M = diag(-1,1,1).  The mesh template
            # must also be X-reflected at load time (reflect_x=True in
            # load_object_geometry) so that the final vertices are
            # M @ (R @ v_local + t) rather than (M@R@M) @ v_local + M@t.
            obj_trans_ = obj_trans.copy()
            obj_trans_[:, 0] *= -1

            obj_rot_ = obj_rot.copy()
            obj_rot_[:, 1:] *= -1
        else:
            trans_ = trans.copy()
            fullpose_ = fullpose.copy()
            obj_trans_ = obj_trans.copy()
            obj_rot_ = obj_rot.copy()
        
        # Object articulation (always zero for GRAB)
        obj_arti_ = np.zeros(obj_trans_.shape[0])
        
        # Split into body parts
        root_orient_ = fullpose_[..., :3]
        pose_body_ = fullpose_[..., 3:66]
        pose_jaw_ = fullpose_[..., 66:69]
        pose_eye_ = fullpose_[..., 69:75]
        pose_lhand_ = fullpose_[..., 75:120]
        pose_rhand_ = fullpose_[..., 120:]
        
        # Compute rest pose offsets
        import smplx
        rest_joint_pos = smplx.lbs.vertices2joints(
            self.body_models[gender].J_regressor,
            self.body_models[gender].v_template.view(1, -1, 3)
        )[0, :len(SMPLX_JOINT_MIRROR_ARR)].cpu().numpy()
        
        root_offset = rest_joint_pos[0]
        
        pos_offset = [[0, 0, 0]]
        for child, parent in enumerate(self.body_models[gender].parents):
            if parent == -1:
                continue
            pos_offset.append(rest_joint_pos[child] - rest_joint_pos[parent])
        pos_offset = np.vstack(pos_offset)
        
        # Object rotation matrix
        obj_rotmat = axis_angle_to_matrix(torch.from_numpy(obj_rot_))
        
        # Chunk sequence into windows
        chunks = SequenceChunker.chunk_sequence(T, self.config.WINDOW, overlap=0)
        
        results = []
        for start, end in chunks:
            chunk_slice = slice(start, end)
            chunk_len = end - start
            
            if chunk_len < self.config.WINDOW:
                continue
            
            betas = np.concatenate([betas, np.zeros(self.config.NUM_BETAS - len(betas))]) if len(betas) < self.config.NUM_BETAS else betas
            
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
            self.body_models[gender].v_template = torch.from_numpy(
                trimesh.load(body_vtemp_path).vertices
            ).float().to(self.device)
            
            # Generate mesh
            bm_out = self.body_models[gender](**motion_params_gt)
            joints = bm_out.joints.cpu().detach().numpy()[:, :len(SMPLX_JOINT_MIRROR_ARR)]
            vertices = bm_out.vertices.cpu().detach().numpy()
            
            # Determine contacts
            offset_floor_height, contacts, discard_seq = \
                determine_floor_height_and_contacts(joints, self.config.TARGET_FPS)
            
            if discard_seq:
                continue
            
            # Compute object mesh and pelvis-to-object offset
            pelvis = joints[:, 0:1]
            obj_inp_dict = {
                'obj_mesh_path': object_mesh_path,
                'obj_scale': torch.tensor([1]),
                'obj_rot': obj_rotmat[chunk_slice],
                'obj_trans': torch.from_numpy(obj_trans_)[chunk_slice],
                'reflect_x': augment_flag,
            }
            obj_mesh_verts, obj_mesh_faces = grab_obj_forward_method(**obj_inp_dict)
            obj_mesh_verts = obj_mesh_verts.numpy()
        
            # Store processed data
            seq_name = f'{self.motion_idx:06d}_{object_name}'
            chunk_data = {
                'betas': betas,
                'gender': gender,
                'seq_name': seq_name,
                'body_dataset_name': 'GRAB',
                'motion_no': self.motion_idx,
                'augment_flag': augment_flag,
                'trans': trans_[chunk_slice],
                'pose_jaw': pose_jaw_[chunk_slice],
                'pose_eye': pose_eye_[chunk_slice],
                'pose_body': pose_body_[chunk_slice],
                'pose_lhand': pose_lhand_[chunk_slice],
                'pose_rhand': pose_rhand_[chunk_slice],
                'root_orient': root_orient_[chunk_slice],
                'root_offset': root_offset,
                'pos_offset': pos_offset,
                'contacts_mask': contacts,
                # Object-specific data
                'obj_name': object_name,
                'obj_trans': obj_trans_[chunk_slice],
                'obj_orient': obj_rot_[chunk_slice],
                'obj_mesh': object_mesh_path,
                'obj_arti': obj_arti_[chunk_slice],
                'obj_scale': np.ones(self.config.WINDOW),
                'obj_moving_frames': obj_moving_frames[chunk_slice],
                'body_vtemp': body_vtemp_path
            }
            
            results.append(chunk_data)
            self.motion_idx += 1
        
        # Restore default template
        self.body_models[gender].v_template = self.default_vtemplates[gender]
        
        return results if results else None
    
    def _visualize_sequence(
        self,
        motion_params_gt: Dict[str, torch.Tensor],
        vertices: np.ndarray,
        seq_name: str,
        dataset_name: Optional[str] = None,
        root_orient: Optional[np.ndarray] = None,
        trans: Optional[np.ndarray] = None,
        joints: Optional[np.ndarray] = None,
        contacts: Optional[np.ndarray] = None,
        object_dict: Optional[Dict[str, Any]] = None,
        status: str = 'discard'
    ):
        """Visualize a sequence for debugging."""
        dataset_name = dataset_name or self.dataset_name
        filename = f"fusion_runs/{BRANCH_NAME}/contact_vis_{status}/{dataset_name}/{seq_name}"
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
            'coef': 2.0
        }
        
        # Add skeleton and object visualization if provided
        kwargs = {'camera_dict': camera_dict, 'color': (60/255, 160/255, 0/255, 1.0)}
        
        if object_dict is not None:
            kwargs['object_dict'] = object_dict
        
        if joints is not None and contacts is not None:
            kwargs['skeleton_dict'] = {
                'positions': joints - np.array([[[-0.6, 0, 0]]]),
                'contact_masks': contacts[:, CONTACT_INDICES],
                'color': (160/255, 60/255, 0/255, 0.9)
            }
        
        self.renderer.render_motion(mesh_dict, filename, **kwargs)
    
    def add_to_data_dict(self, processed_data: Any):
        """Override to handle list of chunks."""
        if isinstance(processed_data, list):
            for chunk_data in processed_data:
                super().add_to_data_dict(chunk_data)
        else:
            super().add_to_data_dict(processed_data)


if __name__ == '__main__':
    processor = GrabProcessor()
    processor.run()
    print('Done processing GRAB')
