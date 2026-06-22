"""
AMASS Dataset Processor - Refactored version using new architecture.
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
from src.utils.process_utils import BRANCH_NAME, determine_floor_height_and_contacts


class AmassProcessor(BodyDatasetProcessor):
    """Processor for AMASS body motion dataset."""
    
    def __init__(self):
        dataset_path = DATASET_PATHS.get('amass', '/data/motion/Body_Raw/AMASS/')
        output_path = 'data/motion/Body_Processed/amass.p'
        
        self.LOWEST_PERCENT = 0.10 
        self.HIGHEST_PERCENT = 1.0
        
        super().__init__(
            dataset_name='AMASS',
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
        self.renderer = ModelLoader.create_renderer(self.config)
        
        # Load MANO mean poses
        self.r_mean_pose, self.l_mean_pose = ModelLoader.load_mano_mean_poses(self.config)
    
    def cleanup_data(self):
        """Remove problematic sequences from AMASS dataset."""
        print("Cleaning up AMASS data...")
        
        # Remove treadmill and move clips from BMLrub
        self._cleanup_bmlrub()
        
        # Remove ice skating clips from MPI_HDM05
        self._cleanup_mpi_hdm05()
        
        # Remove stairs sequences from KIT
        self._cleanup_kit()
        
        # Remove specific sequences from EKUT
        self._cleanup_ekut()

        # Remove specific sequences from CMU
        self._cleanup_cmu()
        
        print("Cleanup complete")
    
    def _cleanup_bmlrub(self):
        """Remove treadmill sequences from BMLrub dataset."""
        dataset_path = os.path.join(self.dataset_path, 'BMLrub')
        if not os.path.exists(dataset_path):
            print('Could not find BMLrub data, skipping...')
            return
        
        for subj_dir in glob.glob(os.path.join(dataset_path, '*')):
            if not os.path.isdir(subj_dir):
                continue
            
            for motion_file in glob.glob(os.path.join(subj_dir, '*.npz')):
                motion_name = os.path.basename(motion_file)
                motion_type = motion_name.split('_')[1] if '_' in motion_name else ''
                
                if motion_type in ['treadmill', 'normal']:
                    print(f"Removing: {motion_file}")
                    os.remove(motion_file)
    
    def _cleanup_mpi_hdm05(self):
        """Remove inline skating sequences from MPI_HDM05."""
        dataset_path = os.path.join(self.dataset_path, 'MPI_HDM05')
        if not os.path.exists(dataset_path):
            print('Could not find MPI_HDM05 data, skipping...')
            return
        
        subj_path = os.path.join(dataset_path, 'dg')
        if not os.path.exists(subj_path):
            return
        
        for motion_file in glob.glob(os.path.join(subj_path, 'HDM_dg_07-01*')):
            print(f"Removing: {motion_file}")
            os.remove(motion_file)

    
    def _cleanup_cmu(self):
        """Remove specific sequences from CMU dataset."""
        dataset_path = os.path.join(self.dataset_path, 'CMU')

        if not os.path.exists(dataset_path):
            print('Could not find CMU data, skipping...')
            return
        
        
        
        for subj_dir in glob.glob(os.path.join(dataset_path, '*')):
            if not os.path.isdir(subj_dir):
                continue

            for motion_file in glob.glob(os.path.join(subj_dir, '*.npz')):
                if any(term in motion_file for term in ['/36/36_24_stageii', '/55/55_14_stageii', '/125/125_02_stageii']):
                    print(f"Removing: {motion_file}")
                    os.remove(motion_file)
    
    def _cleanup_kit(self):
        """Remove stairs sequences from KIT dataset."""
        dataset_path = os.path.join(self.dataset_path, 'KIT')
        if not os.path.exists(dataset_path):
            print('Could not find KIT data, skipping...')
            return
        
        for subj_dir in glob.glob(os.path.join(dataset_path, '*')):
            if not os.path.isdir(subj_dir):
                continue
            
            for motion_file in glob.glob(os.path.join(subj_dir, '*.npz')):
                # Remove stairs sequences
                if any(term in motion_file for term in ['downstairs', 'downstaris', 'upstairs']):
                    print(f"Removing: {motion_file}")
                    os.remove(motion_file)

                
                if subj_dir.endswith('3'):
                    problematic = ['jump_back03_stageii']
                    if any(prob in motion_file for prob in problematic):
                        print(f"Removing: {motion_file}")
                        os.remove(motion_file)

                # Remove specific problematic sequences
                if subj_dir.endswith('912') or subj_dir.split("/")[-1] == '3':
                    problematic = ['912_3_11_stageii', '912_3_12_stageii', '912_3_14_stageii',
                                   '912_3_15_stageii', '912_3_17_stageii', '912_3_18_stageii']
                    if any(prob in motion_file for prob in problematic):
                        print(f"Removing: {motion_file}")
                        os.remove(motion_file)
    
    def _cleanup_ekut(self):
        """Remove specific sequences from EKUT dataset."""
        dataset_path = os.path.join(self.dataset_path, 'EKUT')
        if not os.path.exists(dataset_path):
            print('Could not find EKUT data, skipping...')
            return
        
        problematic_patterns = ['WSUF', 'WSDF', 'BEAM', 'WSDB', 'WSUB', 
                                'WSU5', 'WSU4', 'WSU3', 'WSU2', 'WSU1', 'WSD']
        
        for subj_dir in glob.glob(os.path.join(dataset_path, '*')):
            if not os.path.isdir(subj_dir):
                continue
            
            for motion_file in glob.glob(os.path.join(subj_dir, '*.npz')):
                if any(pattern in motion_file for pattern in problematic_patterns):
                    print(f"Removing: {motion_file}")
                    os.remove(motion_file)
    
    def load_sequences(self) -> List[str]:
        """Load all AMASS sequence files."""
        all_sequences = []
        
        # Get all dataset subdirectories
        dataset_dirs = glob.glob(os.path.join(self.dataset_path, "*"))
        
        # Exclude GRAB (processed separately)
        dataset_dirs = [d for d in dataset_dirs if 'GRAB' not in d and os.path.isdir(d)]
        
        for dataset_dir in dataset_dirs:
            for sub_dir in glob.glob(os.path.join(dataset_dir, "*")):
                if not os.path.isdir(sub_dir):
                    continue
                
                sequences = glob.glob(os.path.join(sub_dir, "*.npz"))
                # Exclude neutral_stagei files
                sequences = [s for s in sequences if not s.endswith('neutral_stagei.npz')]
                all_sequences.extend(sequences)
        
        return all_sequences
    
    def filter_sequences(self, sequences: List[str]) -> Dict[str, List[str]]:
        """Filter sequences into splits (AMASS only has train split)."""
        return {'train': sequences}
    
    def process_sequence(self, sequence_path: str) -> Optional[Dict[str, Any]]:
        """Process a single AMASS sequence file."""
   
        data = np.load(sequence_path, allow_pickle=True)
        
        # Check if it's SMPLX format
        if data['surface_model_type'].item() != 'smplx':
            return None
        
        # Extract basic info
        dataset_name = sequence_path.split('/')[-3]
        seq_name = "_".join(sequence_path.split('/')[-3:])
        
        sample_freq = int(data['mocap_frame_rate'] / self.config.TARGET_FPS)
        betas = data['betas']
        gender = data['gender'].item()
        trans = data['trans'][::sample_freq]
        fullpose = data['poses'][::sample_freq]
        T = len(fullpose)
        
        # Skip short sequences
        if T < self.config.WINDOW:
            return None
        
        # Process both original and augmented versions
        results = []
        for augment_flag in [False, True]:
            result = self._process_single_variant(
                fullpose, trans, betas, gender, dataset_name, 
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
        dataset_name: str,
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
            trans_ = trans.copy()
            fullpose_ = fullpose.copy()
        
        # Split into body parts
        root_orient_ = fullpose_[..., :3]
        pose_body_ = fullpose_[..., 3:66]
        pose_jaw_ = fullpose_[..., 66:69]
        pose_eye_ = fullpose_[..., 69:75]
        pose_lhand_ = fullpose_[..., 75:120]
        pose_rhand_ = fullpose_[..., 120:]
        
        # Compute rest pose offsets
        bm_rest_out = self.body_models[gender](
            betas=torch.from_numpy(betas[None, :]).float().to(self.device)
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
            
            # Special case: EKUT SLP sequences should not be discarded
            if dataset_name == 'EKUT' and "SLP" in seq_name:
                discard_seq = False
            
            if discard_seq:
                continue
            
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
            
            # Visualization 
            if discard_seq:
                self._visualize_sequence(
                    motion_params_gt, vertices, seq_name,
                    dataset_name=dataset_name,
                    root_orient=root_orient_[chunk_slice],
                    trans=trans_[chunk_slice]
                )
            
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
    processor = AmassProcessor()
    processor.run()
    print('Done processing AMASS')
