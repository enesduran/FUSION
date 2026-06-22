"""
SAMP Dataset Processor - Refactored version using new architecture.
"""
import os
import sys
import glob
import torch
import pickle
import numpy as np
from typing import Dict, List, Optional, Any

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

from src.scripts.process.base_processor import BodyDatasetProcessor
from src.scripts.process.config import ProcessingConfig, SMPLX_JOINT_MIRROR_ARR, DATASET_PATHS
from src.scripts.process.processor_utils import ModelLoader, SequenceChunker
from src.utils.process_utils import BRANCH_NAME, determine_floor_height_and_contacts


class SampProcessor(BodyDatasetProcessor):
    """Processor for SAMP body motion dataset."""
    
    def __init__(self):
        dataset_path = DATASET_PATHS.get(
            'samp',
            'data/motion/Body_Raw/SAMP/1'
        )

        output_path = 'data/motion/Body_Processed/samp.p'
        
        super().__init__(
            dataset_name='SAMP',
            dataset_path=dataset_path,
            output_path=output_path
        )
        
        self.LOWEST_PERCENT = 0.40
        self.HIGHEST_PERCENT = 1.0
        
        # Initialize models
        self.device = self.get_device()
        self.body_models = ModelLoader.load_smplx_models(
            batch_size=self.config.WINDOW,
            num_betas=self.config.NUM_BETAS,
            device=self.device,
            config=self.config
        )
        self.renderer = ModelLoader.create_renderer(self.config)
    
    def load_sequences(self) -> List[str]:
        """Load all SAMP sequence files."""
        datapaths = glob.glob(os.path.join(self.dataset_path, "*", "*.pkl"))
        
        # Filter out sequences with "lie" in the name
        datapaths = [p for p in datapaths if "lie" not in p]
        
        return datapaths
    
    def filter_sequences(self, sequences: List[str]) -> Dict[str, List[str]]:
        """Filter sequences into splits (SAMP only has train split)."""
        return {'train': sequences}
    
    def process_sequence(self, sequence_path: str) -> Optional[Dict[str, Any]]:
        """Process a single SAMP sequence file."""
   
        # Load pickle file
        with open(sequence_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        
        # Extract basic info
        sample_freq = int(data['mocap_framerate'] / self.config.TARGET_FPS)
        betas = data['shape_est_betas'][:self.config.NUM_BETAS]
        
        gender = data['ps']['gender']
        trans = data['pose_est_trans'][::sample_freq]
        fullpose = data['pose_est_fullposes'][::sample_freq]
        
        T = len(fullpose)
        
        # Skip short sequences
        if T < self.config.WINDOW:
            return None
        
        seq_name = "_".join(sequence_path.split('/')[-1:])
        
        # Process both original and augmented versions
        results = []
        for augment_flag in [True, False]:
            result = self._process_single_variant(
                fullpose, trans, betas, gender, 
                seq_name, augment_flag, T
            )
            if result:
                results.extend(result)
        
        return results if results else None
            

    
    def _process_single_variant(
        self,
        fullpose: np.ndarray,
        trans: np.ndarray,
        betas: np.ndarray,
        gender: str,
        seq_name: str,
        augment_flag: bool,
        T: int
    ) -> Optional[List[Dict[str, Any]]]:
        """Process original or augmented version of a sequence."""
        
        # Apply augmentation if needed
        if augment_flag:
            trans_ = trans.copy()
            trans_[:, 0] *= -1
            
            fullpose_ = fullpose.copy().reshape(-1, len(SMPLX_JOINT_MIRROR_ARR), 3)
            fullpose_ = fullpose_[:, SMPLX_JOINT_MIRROR_ARR]
            fullpose_[..., 1:] *= -1
            fullpose_ = fullpose_.reshape(-1, len(SMPLX_JOINT_MIRROR_ARR) * 3)
        else:
            fullpose_ = fullpose.copy()
            trans_ = trans.copy()
        
        # Split into body parts
        root_orient_ = fullpose_[..., :3]
        pose_body_ = fullpose_[..., 3:66]
        pose_jaw_ = fullpose_[..., 66:69]
        pose_eye_ = fullpose_[..., 69:75]
        pose_lhand_ = fullpose_[..., 75:120]
        pose_rhand_ = fullpose_[..., 120:]
        
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
        
        # Chunk sequence into non-overlapping windows
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
            
            # Generate joints
            bm_out = self.body_models[gender](**motion_params_gt)
            joints = bm_out.joints.cpu().detach().numpy()[:, :len(SMPLX_JOINT_MIRROR_ARR)]
            vertices = bm_out.vertices.cpu().detach().numpy()

 
            # Determine floor height and contacts
            offset_floor_height, contacts, discard_seq = \
                determine_floor_height_and_contacts(joints, self.config.TARGET_FPS)
            
            # Visualization
            if discard_seq:
                self._visualize_sequence(
                    motion_params_gt, vertices, seq_name,
                    dataset_name=self.dataset_name,
                    root_orient=root_orient_[chunk_slice],
                    trans=trans_[chunk_slice]
                )
                continue
            
            # Store processed data
            chunk_data = {
                'betas': betas,
                'gender': gender,
                'seq_name': seq_name,
                'motion_no': self.motion_idx,
                'body_dataset_name': self.dataset_name,
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
    processor = SampProcessor()
    processor.run()
    print('Done processing SAMP')
