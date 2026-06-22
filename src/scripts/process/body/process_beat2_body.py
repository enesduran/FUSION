"""
BEAT2 Dataset Processor - Refactored version using new architecture.
"""
import os
import sys
import glob
import torch
import numpy as np
from typing import Dict, List, Optional, Any

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

from src.scripts.process.base_processor import BodyDatasetProcessor
from src.scripts.process.config import ProcessingConfig, SMPLX_JOINT_MIRROR_ARR, DATASET_PATHS
from src.scripts.process.processor_utils import ModelLoader, SequenceChunker
from src.utils.process_utils import BRANCH_NAME, CONTACT_INDICES, determine_floor_height_and_contacts
from src.utils.transforms import axis_angle_to_matrix, matrix_to_axis_angle


# Transformation from Vicon to SMPLX coordinate frame
R_V2S = torch.tensor([[1., 0., 0.],
                      [0., 0., 1.],
                      [0., -1., 0.]]).reshape(3, 3)


def rotate_global(global_rot_aa, global_trans, root_offset):
 
    origin2root = torch.from_numpy(global_trans + root_offset)
    global_rot_aa = matrix_to_axis_angle(R_V2S.T @ axis_angle_to_matrix(torch.from_numpy(global_rot_aa).float()))
    global_transl = torch.matmul(origin2root, R_V2S) - root_offset
 
    return global_rot_aa.numpy(), global_transl.numpy()


class Beat2Processor(BodyDatasetProcessor):
    """Processor for BEAT2 body motion dataset."""
    
    def __init__(self):
        dataset_path = DATASET_PATHS.get('beat2', 'data/motion/Body_Raw/BEAT2')
        output_path = 'data/motion/Body_Processed/beat2.p'
        
        super().__init__(
            dataset_name='BEAT2',
            dataset_path=dataset_path,
            output_path=output_path
        )
        
        self.LOWEST_PERCENT = 0.35 
        self.HIGHEST_PERCENT = 1.00
        
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
    
    def cleanup_data(self):
        """BEAT2 doesn't require specific cleanup."""
        print('No cleanup needed for BEAT2')
    
    def load_sequences(self) -> List[str]:
        """Load all BEAT2 sequence files."""
        # Pattern: BEAT2_path/**/*.npz
        all_sequences = []
        
        # Search all subdirectories for npz files
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith('.npz'):
                    all_sequences.append(os.path.join(root, file))
        
        return all_sequences
    
    def filter_sequences(self, sequences: List[str]) -> Dict[str, List[str]]:
        """Filter sequences into splits (BEAT2 only has train split)."""
        return {'train': sequences}
    
    def process_sequence(self, sequence_path: str) -> Optional[List[Dict[str, Any]]]:
        """Process a single BEAT2 sequence file."""
 
        data = np.load(sequence_path, allow_pickle=True)
        
        # Extract basic info
        seq_name = os.path.basename(sequence_path).replace('.npz', '')
        dataset_name = 'BEAT2'
                
        
        # Sample to target FPS
        sample_freq = int(data['mocap_frame_rate'].item() / self.config.TARGET_FPS)  # BEAT2 is at 30 FPS
        
        betas = data['betas']
        gender = data['gender'].item()
        trans = data['trans'][::sample_freq]
        

        root_orient = data['root_orient'][::sample_freq]

        # Full pose parameters
        pose_body = data['pose_body'][::sample_freq]
        pose_jaw = data['pose_jaw'][::sample_freq]
        pose_eye = data['pose_eye'][::sample_freq]
        pose_lhand = data['pose_hand'][::sample_freq][:, :45]
        pose_rhand = data['pose_hand'][::sample_freq][:, 45:]
        
        # Compute rest pose offsets
        bm_rest_out = self.body_models[gender](
            betas=torch.from_numpy(betas[None, :]).float().to(self.device))
        rest_joint_pos = bm_rest_out.joints.cpu().detach().numpy()[0, :len(SMPLX_JOINT_MIRROR_ARR)]
        root_offset = rest_joint_pos[0]
        
        # Compute joint offsets
        pos_offset = [[0, 0, 0]]
        for child, parent in enumerate(self.body_models[gender].parents):
            if parent == -1:
                continue
            pos_offset.append(rest_joint_pos[child] - rest_joint_pos[parent])
        pos_offset = np.vstack(pos_offset)

        T = len(trans)

        root_orient, trans = rotate_global(root_orient.copy(), 
                                                      data['trans'][::sample_freq].astype(np.float32), 
                                                      root_offset)   
    
        
        # Skip short sequences
        if T < self.config.WINDOW:
            return None
        
        # Process both original and augmented versions
        results = []
        for augment_flag in [False, True]:
            result = self._process_single_variant(
                root_orient, pose_body, pose_jaw, pose_eye,
                pose_lhand, pose_rhand, trans, betas, gender,
                dataset_name, seq_name, augment_flag, 
                pos_offset, root_offset, T
            )
            if result:
                results.extend(result)
        
        return results if results else None
               
    def _process_single_variant(
        self,
        root_orient: np.ndarray,
        pose_body: np.ndarray,
        pose_jaw: np.ndarray,
        pose_eye: np.ndarray,
        pose_lhand: np.ndarray,
        pose_rhand: np.ndarray,
        trans: np.ndarray,
        betas: np.ndarray,
        gender: str,
        dataset_name: str,
        seq_name: str,
        augment_flag: bool,
        pos_offset: np.ndarray,
        root_offset: np.ndarray,
        T: int
    ) -> Optional[List[Dict[str, Any]]]:
        """Process original or augmented version of a sequence."""
                
        # Apply augmentation if needed
        if augment_flag:
            trans_ = trans.copy()
            trans_[:, 0] *= -1

            fullpose = np.hstack([root_orient, pose_body, pose_jaw, pose_eye, pose_lhand, pose_rhand])
            
            fullpose_ = fullpose.copy().reshape(-1, len(SMPLX_JOINT_MIRROR_ARR), 3)
            fullpose_ = fullpose_[:, SMPLX_JOINT_MIRROR_ARR]
            fullpose_[..., 1:] *= -1
            fullpose_ = fullpose_.reshape(-1, len(SMPLX_JOINT_MIRROR_ARR) * 3)


            # Split into body parts
            root_orient_ = fullpose_[..., :3]
            pose_body_ = fullpose_[..., 3:66]
            pose_jaw_ = fullpose_[..., 66:69]
            pose_eye_ = fullpose_[..., 69:75]
            pose_lhand_ = fullpose_[..., 75:120]
            pose_rhand_ = fullpose_[..., 120:]


        else:
            trans_ = trans.copy()
            pose_body_ = pose_body.copy()
            root_orient_ = root_orient.copy()
            pose_lhand_ = pose_lhand.copy()
            pose_rhand_ = pose_rhand.copy()
            pose_jaw_ = pose_jaw.copy()
            pose_eye_ = pose_eye.copy()
        
        
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
                'transl': torch.from_numpy(trans_[chunk_slice]).float().to(self.device),
                'expression': torch.zeros((chunk_len, 10)).float().to(self.device)
            }
            
            # Generate mesh
            bm_out = self.body_models[gender](**motion_params_gt)
            joints = bm_out.joints.cpu().detach().numpy()[:, :len(SMPLX_JOINT_MIRROR_ARR)]
            vertices = bm_out.vertices.cpu().detach().numpy()
            
            # Determine floor height and contacts
            offset_floor_height, contacts, discard_seq = \
                determine_floor_height_and_contacts(joints, self.config.TARGET_FPS)
            
            # BEAT2 has many sequences with unusual poses, might want to keep some
            # that would otherwise be discarded
            if discard_seq:
                # Optionally visualize discarded sequences
                self._visualize_sequence(motion_params_gt, vertices, seq_name,
                                        dataset_name=dataset_name,
                                        joints=joints,
                                        contacts=contacts,
                                        status='discard')

         
            # Store processed data
            chunk_data = {
                'betas': betas,
                'gender': gender,
                'seq_name': seq_name,
                'motion_no': self.motion_idx,
                'body_dataset_name': dataset_name,
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
                'contacts_mask': contacts
            }
            
            results.append(chunk_data)
            self.motion_idx += 1
        
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
        status: str = 'accept'
    ):
        """Visualize a sequence for debugging."""
        dataset_name = dataset_name or self.dataset_name
        
        # Use different directory for discarded sequences
        if status == 'discard':
            filename = f"fusion_runs/{BRANCH_NAME}/contact_vis_{status}/{dataset_name}/{seq_name}_{self.motion_idx}"
        else:
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
            'coef': 1.9 if status == 'accept' else 2.0
        }
        
        # Add skeleton visualization if joints and contacts provided
        if joints is not None and contacts is not None:
            skeleton_dict = {
                'positions': joints,
                'contact_masks': contacts[:, CONTACT_INDICES],
                'color': (16/255, 60/255, 160/255, 0.9)
            }
            self.renderer.render_motion(
                mesh_dict, filename,
                skeleton_dict=skeleton_dict,
                camera_dict=camera_dict,
                color=(255/255, 160/255, 0/255, 1)
            )
        else:
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
    processor = Beat2Processor()
    processor.run()
    print('Done processing BEAT2')
