"""
EMBODY3D Dataset Processor - Refactored version using new architecture.
"""
import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

from src.scripts.process.base_processor import BodyDatasetProcessor
from src.scripts.process.config import ProcessingConfig, SMPLX_JOINT_MIRROR_ARR, DATASET_PATHS
from src.scripts.process.processor_utils import ModelLoader, SequenceChunker
from src.utils.process_utils import BRANCH_NAME, determine_floor_height_and_contacts
from src.utils.transforms import axis_angle_to_matrix, matrix_to_axis_angle


class FeatName(Enum):
    """List of possible features for all datasets"""
    # SMPLX body features
    BODY = "smplx_mesh_body_pose"
    ROT = "smplx_mesh_global_orient"
    TRANS = "smplx_mesh_transl"
    SHAPE = "smplx_mesh_betas"
    LEFT_HAND = "smplx_mesh_left_hand_pose"
    RIGHT_HAND = "smplx_mesh_right_hand_pose"
    RIGHT_EYE = "smplx_mesh_reye_pose"
    LEFT_EYE = "smplx_mesh_leye_pose"
    JAW = "smplx_mesh_jaw_pose"
    
    # Metadata
    SUBJECT_ID = "subject_id"
    SEQUENCE_ID = "sequence_id"
    START_FRAME = "start_frame"
    LENGTH = "length"


class EMBODY3DDataset:
    """Simple dataset loader for EMBODY3D that preloads sequences into memory."""
    
    def __init__(
        self,
        split: str,
        data_dir: str,
        seq_len: int = 120,
        train_ratio: float = 0.07,
        val_ratio: float = 0.07,
        features_to_load: Optional[List[str]] = None
    ):
        """
        Args:
            split: 'train' or 'val'
            data_dir: Root directory containing dataset.json
            seq_len: Sequence length (window size)
            train_ratio: Fraction of subjects for training
            val_ratio: Fraction of subjects for validation
            features_to_load: List of feature names to load
        """
        self.split = split
        self.seq_len = seq_len
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.features_to_load = features_to_load or []
        
        # Load dataset info
        self.data_dir = data_dir
        self.dataset_info = self._load_and_split_dataset()
        
        # Create index mapping
        self._create_index_mapping()
        
        # Preload all data into memory
        self._preload_dataset()
    
    def _load_and_split_dataset(self) -> Dict[str, Any]:
        """Load dataset.json and split by subjects."""
        dataset_info_path = os.path.join(self.data_dir, "dataset.json")
        with open(dataset_info_path, "r") as f:
            full_dataset_info = json.load(f)
        
        # Split by subjects
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
        """Create mapping from index to (subject_id, sequence_name)."""
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
            # Number of non-overlapping windows in this sequence
            num_segments = self.dataset_info[seq[0]][seq[1]]["length"] // self.seq_len
            idx += num_segments
        
        self.num_valid_segments = idx
    
    def _preload_dataset(self):
        """Preload all data into memory."""
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
            
            # Load each feature
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
    
    def _idx2segment(self, idx: int) -> Tuple[str, str, int]:
        """Convert global index to (subject_id, sequence_name, frame_offset)."""
        closest_idx = np.searchsorted(list(self.idx2seq.keys()), idx, side="right") - 1
        closest_idx = list(self.idx2seq.keys())[closest_idx]
        subject_id, sequence_name = self.idx2seq[closest_idx]
        frame_offset = (idx - closest_idx) * self.seq_len
        return subject_id, sequence_name, frame_offset
    
    def __len__(self) -> int:
        return self.num_valid_segments
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sequence window."""
        subject_id, sequence_name, start_frame = self._idx2segment(idx)
        end_frame = start_frame + self.seq_len
        
        data = {
            FeatName.SUBJECT_ID.value: subject_id,
            FeatName.SEQUENCE_ID.value: sequence_name,
            FeatName.START_FRAME.value: start_frame,
            FeatName.LENGTH.value: self.seq_len,
        }
        
        # Load features for this window
        for feat_name in self.features_to_load:
            feat_data = self.dataset[subject_id][sequence_name][feat_name]
            data[feat_name] = torch.from_numpy(feat_data[start_frame:end_frame].copy())
        
        return data





class Embody3DProcessor(BodyDatasetProcessor):
    """Processor for EMBODY3D body motion dataset.
    
    Iterates over all subfolders under the embody-3d/data directory
    (acting, aiagent, charades, contact_charades, etc.).
    """
    
    def __init__(self):
        dataset_path = DATASET_PATHS.get(
            'embody3d', 
            'external/embody-3d/data'
        )
        output_path = 'data/motion/Body_Processed/embody3d.p'
        
        self.LOWEST_PERCENT = 0.75    
        self.HIGHEST_PERCENT = 1.0
        
        self.total_discarded = 0
        
        super().__init__(
            dataset_name='EMBODY3D',
            dataset_path=dataset_path,
            output_path=output_path
        )
        
        # Initialize models
        self.device = self.get_device()
        self.body_models = ModelLoader.load_smplx_models(
            batch_size=self.config.WINDOW,
            num_betas=self.config.NUM_BETAS,
            device=self.device,
            config=self.config)
        
        self.renderer = ModelLoader.create_renderer(self.config)
        
        # Define features to load
        self.smplx_keys = [
            FeatName.BODY.value, 
            FeatName.ROT.value,
            FeatName.TRANS.value,
            FeatName.SHAPE.value,
            FeatName.LEFT_HAND.value,
            FeatName.RIGHT_HAND.value, 
            'missing'
        ]
        
        # Discover all subfolders with a dataset.json
        self.subfolders = self._discover_subfolders()
        
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
        """Return list of sequence indices from all EMBODY3D subfolders."""
        # Build per-subfolder, per-split datasets
        # Structure: {split: [(subfolder_name, dataset, local_idx), ...]}
        self._split_datasets = {split: [] for split in ['train', 'val', 'test']}
        
        for subfolder in self.subfolders[:]:
            subfolder_path = os.path.join(self.dataset_path, subfolder)
            dataset_kwargs = dict(data_dir=subfolder_path,
                                    seq_len=self.config.WINDOW,
                                    train_ratio=0.93,
                                    val_ratio=0.08,
                                    features_to_load=self.smplx_keys)
            
            for split in ['train', 'val', 'test']:
                ds = EMBODY3DDataset(split=split, **dataset_kwargs)
                for local_idx in range(len(ds)):
                    self._split_datasets[split].append((subfolder, ds, local_idx))
            
        total = sum(len(v) for v in self._split_datasets.values())
        print(f"Total sequences across all subfolders: {total}")
        return list(range(len(self._split_datasets['train'])))
    
    def filter_sequences(self, sequences: List[int]) -> Dict[str, List[int]]:
        """Split sequences into train/val/test."""
        return {
            split: list(range(len(items)))
            for split, items in self._split_datasets.items()
        }
    
    def run(self):
        """Override run to track the current split being processed."""
        print(f"Starting {self.dataset_name} dataset processing...")

        self.clear_existing_processed_outputs()
        
        self.cleanup_data()
        
        print("Loading sequences...")
        sequences = self.load_sequences()
        print(f"Found {len(sequences)} sequences")
        
        print("Filtering sequences into splits...")
        filtered_splits = self.filter_sequences(sequences)
        
        for split_name, split_sequences in filtered_splits.items():
            print(f"\nProcessing {split_name} split ({len(split_sequences)} sequences)...")
            
            self.current_split = split_name
            self.reset_data_dict()
            
            for seq_idx in tqdm(split_sequences, desc=f"Processing {split_name}"):
         
                self.process_sequence(seq_idx)
            
            # Apply motion variance filter (std-based discard)
            self.apply_motion_variance_filter(split_name)

            if len(self.data_dict) > 0:
                self.save_data(split=split_name)
            else:
                print(f"Warning: No valid sequences in {split_name} split")
        
        print(f"\n{self.dataset_name} processing complete!")
    
    def process_sequence(self, sequence_idx: int) -> Optional[Dict[str, Any]]:
        """Process a single EMBODY3D sequence."""

        subfolder, dataset, local_idx = self._split_datasets[self.current_split][sequence_idx]
        data = dataset[local_idx]

        data['mocap_frame_rate'] = 30 # Checked from the documentation
        
        # Extract parameters
        betas = data[FeatName.SHAPE.value].to(self.device)
        gender = 'neutral'
        
        sample_freq = int(data['mocap_frame_rate'] / self.config.TARGET_FPS)
        trans = data[FeatName.TRANS.value][::sample_freq].numpy()

        if data['missing'].sum().item() != self.config.WINDOW:
            # print(f"Skipping sequence {sequence_idx} due to missing frames")
            return None 
        
        # Concatenate full pose (torch.cat since __getitem__ returns tensors)
        fullpose = torch.cat([
            data[FeatName.ROT.value],
            data[FeatName.BODY.value],
            torch.zeros((self.config.WINDOW, 3)),   # jaw pose
            torch.zeros((self.config.WINDOW, 3)),   # left eye pose
            torch.zeros((self.config.WINDOW, 3)),   # right eye pose
            data[FeatName.LEFT_HAND.value],
            data[FeatName.RIGHT_HAND.value]
        ], dim=-1)[::sample_freq].numpy()
        
        T = len(fullpose)
        
        if T < self.config.WINDOW:
            return None
    
        # Compute rest pose and offsets
        bm_rest_out = self.body_models[gender](betas=betas, gender=gender)
        rest_joint_pos = bm_rest_out.joints.cpu().detach().numpy()[0, :len(SMPLX_JOINT_MIRROR_ARR)]
        root_offset = rest_joint_pos[0]
        
        
        # Compute joint offsets
        pos_offset = [[0, 0, 0]]
        for child, parent in enumerate(self.body_models[gender].parents):
            if parent == -1:
                continue
            pos_offset.append(rest_joint_pos[child] - rest_joint_pos[parent])
        pos_offset = np.vstack(pos_offset)
        
        # Rotate from Vicon to SMPLX coordinate frame
        root_orient, trans = self._rotate_global(fullpose[:, :3].copy(), trans, root_offset)
        fullpose[:, :3] = root_orient
        
        results = []
         
        for augment_flag in [False, True]:
            result = self._process_single_variant(
                fullpose, trans, betas, gender,
                sequence_idx, root_offset, pos_offset, subfolder,
                augment_flag)
            
            if result:
                self.add_to_data_dict(result)
            
        return results if results else None
        

    def _rotate_global(self, global_rot_aa, global_trans, root_offset) -> Tuple[np.ndarray, np.ndarray]:
        """Rotate from Vicon to SMPLX coordinate frame."""

        # Transformation from Vicon to SMPLX coordinate frame
        R_V2S = torch.tensor([[1., 0., 0.],
                            [0., 0., 1.],
                            [0., -1., 0.]]).reshape(3, 3)

        
        origin2root = torch.from_numpy(global_trans + root_offset)
        global_rot_aa = matrix_to_axis_angle(R_V2S.T @ axis_angle_to_matrix(torch.from_numpy(global_rot_aa).float()))
        global_transl = torch.matmul(origin2root, R_V2S) - root_offset
        
        return global_rot_aa.numpy(), global_transl.numpy()
    
    def _process_single_variant(
        self,
        fullpose: np.ndarray,
        trans: np.ndarray,
        betas: torch.Tensor,
        gender: str,
        seq_idx: int,
        root_offset: np.ndarray,
        pos_offset: np.ndarray,
        subfolder: str,
        augment_flag: bool
    ) -> Optional[Dict[str, Any]]:


        """Process a single variant (no augmentation)."""
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
        
        # Split pose into components
        root_orient = fullpose_[..., :3]
        pose_body = fullpose_[..., 3:66]
        pose_jaw = fullpose_[..., 66:69]
        pose_eye = fullpose_[..., 69:75]
        pose_lhand = fullpose_[..., 75:120]
        pose_rhand = fullpose_[..., 120:]
        trans = trans_.copy()
        
        T = len(fullpose_)
        

        # Use non-overlapping chunks
        chunks = SequenceChunker.chunk_sequence(
            T, self.config.WINDOW, overlap=0
        )
        
        # Process only the first chunk (EMBODY3D seems to process one chunk per sequence)
        if len(chunks) == 0:
            return None
        
        start, end = chunks[0]
        chunk_slice = slice(start, end)
        chunk_len = end - start
        
        if chunk_len < self.config.WINDOW:
            return None
        
        # Prepare motion parameters
        motion_params_gt = {
            'betas': betas.float().to(self.device),
            'global_orient': torch.from_numpy(root_orient[chunk_slice]).float().to(self.device),
            'body_pose': torch.from_numpy(pose_body[chunk_slice]).float().to(self.device),
            'left_hand_pose': torch.from_numpy(pose_lhand[chunk_slice]).float().to(self.device),
            'right_hand_pose': torch.from_numpy(pose_rhand[chunk_slice]).float().to(self.device),
            'jaw_pose': torch.from_numpy(pose_jaw[chunk_slice]).float().to(self.device),
            'leye_pose': torch.from_numpy(pose_eye[chunk_slice][:, :3]).float().to(self.device),
            'reye_pose': torch.from_numpy(pose_eye[chunk_slice][:, 3:]).float().to(self.device),
            'transl': torch.from_numpy(trans[chunk_slice]).float().to(self.device),
            'expression': torch.zeros((chunk_len, 10)).float().to(self.device)
        }
        
        # Generate mesh
        bm_out = self.body_models[gender](**motion_params_gt)
        joints = bm_out.joints.cpu().detach().numpy()[:, :len(SMPLX_JOINT_MIRROR_ARR)]
        vertices = bm_out.vertices.cpu().detach().numpy()
        
        # Determine floor height and contacts
        offset_floor_height, contacts, discard_seq = \
            determine_floor_height_and_contacts(joints, self.config.TARGET_FPS)
        

        # Store processed data
        seq_name = f"{subfolder}_{str(seq_idx).zfill(6)}"
 
        if discard_seq:
            self._visualize_sequence(
                motion_params_gt, vertices, seq_name,
                dataset_name=self.dataset_name,
                root_orient=root_orient[chunk_slice],
                trans=trans[chunk_slice],
                augment_flag=augment_flag,
                status='discard')
            print(f"Discarding EMBODY3D sequence {seq_idx}")

            self.total_discarded += 1
            return None
        
        
        # Validate betas are constant
        if betas.std(0).sum() > 0:
            print(f"Warning: Betas are not constant in sequence {seq_name} {betas.std(0).sum()}, skipping sequence")
            return None

   
        chunk_data = {
            'betas': betas[0, :self.config.NUM_BETAS].cpu().numpy(),
            'gender': gender,
            'seq_name': seq_name,
            'motion_no': self.motion_idx,
            'body_dataset_name': self.dataset_name,
            'augment_flag': augment_flag,
            'trans': trans[chunk_slice],
            'pose_jaw': pose_jaw[chunk_slice],
            'pose_eye': pose_eye[chunk_slice],
            'pose_body': pose_body[chunk_slice],
            'pose_lhand': pose_lhand[chunk_slice],
            'pose_rhand': pose_rhand[chunk_slice],
            'root_orient': root_orient[chunk_slice],
            'root_offset': root_offset,
            'pos_offset': pos_offset,
            'contacts_mask': contacts
        }
        
        
        self.motion_idx += 1
        return chunk_data
    
    def _visualize_sequence(
        self,
        motion_params_gt: Dict[str, torch.Tensor],
        vertices: np.ndarray,
        seq_name: str,
        dataset_name: Optional[str] = None,
        root_orient: Optional[np.ndarray] = None,
        trans: Optional[np.ndarray] = None,
        augment_flag: bool = False,
        status: str = 'accept'
    ):
        """Visualize a sequence for debugging."""
        dataset_name = dataset_name or self.dataset_name
        filename = f"fusion_runs/{BRANCH_NAME}/contact_vis_{status}/{dataset_name}/{seq_name}_{augment_flag}"
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
            'coef': 1.9}
        
        self.renderer.render_motion(
            mesh_dict, filename,
            camera_dict=camera_dict,
            color=(255/255, 160/255, 0/255, 1))


if __name__ == '__main__':
    processor = Embody3DProcessor()
    processor.run()
    print(f'Done processing EMBODY3D. Total discarded sequences: {processor.total_discarded}')
