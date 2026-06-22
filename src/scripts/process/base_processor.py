"""
Base processor class for all dataset processing scripts.
Defines the common interface and shared functionality.
"""
import os
import sys
import smplx
import torch
import joblib
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.render.mesh_viz import RendererWrapper
from src.scripts.process.config import ProcessingConfig
from src.utils.transforms import axis_angle_to_quaternion, axis_angle_to_rotation_6d, quat_fk


class BaseDatasetProcessor(ABC):
    """
    Abstract base class for dataset processors.
    All dataset-specific processors should inherit from this class.
    """
    
    # Override in subclasses to enable motion variance filtering.
    # LOWEST_PERCENT removes the bottom X% of sequences by motion std.
    # HIGHEST_PERCENT is used only for histogram visualization.
    LOWEST_PERCENT: Optional[float] = None
    HIGHEST_PERCENT: Optional[float] = None
    
    def __init__(
        self, 
        dataset_name: str,
        dataset_path: str,
        output_path: str,
        config: Optional[ProcessingConfig] = None
    ):
        """
        Initialize the dataset processor.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'amass', 'grab')
            dataset_path: Path to the raw dataset
            output_path: Path where processed data will be saved
            config: Processing configuration (uses default if None)
        """
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.config = config or ProcessingConfig()
        
        # Initialize data storage
        self.data_dict = {}
        self.motion_idx = 0
        
        # Will be initialized by subclasses if needed
        self.device = None
        self.body_models = None
        self.renderer = None
        
    @abstractmethod
    def load_sequences(self) -> List[str]:
        """
        Load and return list of sequence file paths.
        
        Returns:
            List of paths to sequence files
        """
        pass
    
    @abstractmethod
    def filter_sequences(self, sequences: List[str]) -> Dict[str, List[str]]:
        """
        Filter sequences into train/val/test splits.
        
        Args:
            sequences: List of all sequence paths
            
        Returns:
            Dictionary mapping split names to sequence lists
            e.g., {'train': [...], 'val': [...]}
        """
        pass
    
    @abstractmethod
    def process_sequence(self, sequence_path: str) -> Optional[Dict[str, Any]]:
        """
        Process a single sequence file.
        
        Args:
            sequence_path: Path to the sequence file
            
        Returns:
            Dictionary containing processed data, or None if sequence should be skipped
        """
        pass
    
    def cleanup_data(self):
        """
        Optional: Perform any data cleanup before processing.
        Override this method if the dataset needs pre-processing cleanup.
        """
        pass
    
    def validate_sequence(self, seq_data: Dict[str, Any]) -> bool:
        """
        Optional: Validate sequence data before processing.
        Override this method to add dataset-specific validation.
        
        Args:
            seq_data: Sequence data to validate
            
        Returns:
            True if sequence is valid, False otherwise
        """
        return True
    
    def add_to_data_dict(self, processed_data: Dict[str, Any]):
        """
        Add processed sequence data to the main data dictionary.
        
        Args:
            processed_data: Processed data for one sequence
        """
        self.data_dict[self.motion_idx] = processed_data
        self.motion_idx += 1
    
    def save_data(self, split: str = 'train'):
        """
        Save processed data to disk.
        
        Args:
            split: Split name (e.g., 'train', 'val', 'test')
        """
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        # Modify output path to include split name
        base_path = self.output_path.replace('.p', f'_{split}.p')
        
        print(f"Saving {len(self.data_dict)} sequences to {base_path}")
        joblib.dump(self.data_dict, base_path)
    
    def reset_data_dict(self):
        """Reset data dictionary for processing next split."""
        self.data_dict = {}
        self.motion_idx = 0
    
    def run(self):
        """
        Main processing pipeline.
        Executes the full data processing workflow.
        """
        print(f"Starting {self.dataset_name} dataset processing...")

        # Always regenerate processed outputs from scratch on each run.
        self.clear_existing_processed_outputs()
        
        # Step 1: Cleanup (if needed)
        self.cleanup_data()
        
        # Step 2: Load all sequences
        print("Loading sequences...")
        sequences = self.load_sequences()
        print(f"Found {len(sequences)} sequences")
        
        # Step 3: Filter into splits
        print("Filtering sequences into splits...")
        filtered_splits = self.filter_sequences(sequences)
        
        # Step 4: Process each split
        for split_name, split_sequences in filtered_splits.items():
            print(f"\nProcessing {split_name} split ({len(split_sequences)} sequences)...")
            
            self.reset_data_dict()
            
            for seq_path in tqdm(split_sequences, desc=f"Processing {split_name}"):
                try:
                    processed_data = self.process_sequence(seq_path)
                    
                    if processed_data is not None:
                        self.add_to_data_dict(processed_data)
                        
                except Exception as e:
                    print(f"\nError processing {seq_path}: {str(e)}")
                    continue
            
            # # Step 5: Apply motion variance filtering
            self.apply_motion_variance_filter(split_name)

            # Step 6: Save the split
            if len(self.data_dict) > 0:
                self.save_data(split=split_name)
            else:
                print(f"Warning: No valid sequences in {split_name} split")
        
        print(f"\n{self.dataset_name} processing complete!")

    def clear_existing_processed_outputs(self):
        """Delete previous processed output files for this dataset."""
        output_dir = os.path.dirname(self.output_path)
        output_stem = os.path.splitext(os.path.basename(self.output_path))[0]
    
        if not output_dir or not os.path.isdir(output_dir):
            return

        removed_count = 0
        for split_name in ['train', 'val', 'test']:
            split_output_path = os.path.join(output_dir, f"{output_stem}_{split_name}_processed.p")
            if os.path.isfile(split_output_path):
                print(f"Removing existing processed file: {split_output_path}")
                os.remove(split_output_path)
                removed_count += 1

        if os.path.isfile(self.output_path):
            print(f"Removing existing processed file: {self.output_path}")
            os.remove(self.output_path)
            removed_count += 1

        if removed_count > 0:
            print(f"Removed {removed_count} existing processed file(s) for {self.dataset_name}")
    
    def get_device(self) -> torch.device:
        """Get PyTorch device (cached)."""
        if self.device is None:
            self.device = torch.device(self.config.DEVICE if torch.cuda.is_available() else "cpu")
        return self.device

    def get_output_dir(self) -> str:
        """Get output directory for processed data. Override in subclasses."""
        return os.path.dirname(self.output_path)

    def apply_motion_variance_filter(self, split_name: str):
        """Filter data_dict by motion variance.

        Removes sequences with the lowest motion variance (static/near-static poses).
        Uses LOWEST_PERCENT to determine the cutoff threshold.
        Saves a histogram visualization and renders boundary samples to the output directory.
        Subclasses must implement _compute_motion_variance() to define the variance metric.
        """
        if self.LOWEST_PERCENT is None:
            self.LOWEST_PERCENT = 0.0  # No filtering by default
            
        if self.HIGHEST_PERCENT is None:
            self.HIGHEST_PERCENT = 1.0  # No upper threshold by default
        
        if len(self.data_dict) == 0:
            return
        
        n_total = len(self.data_dict)
        data_keys = list(self.data_dict.keys())

        pose_change, mean_travel = self._compute_motion_variance()
        
        sorted_idx = np.argsort(mean_travel)
    
        low_cutoff = int(self.LOWEST_PERCENT * len(sorted_idx))
        high_cutoff = int(self.HIGHEST_PERCENT * len(sorted_idx))
        
        cropped_sorted_idx = sorted_idx[low_cutoff:high_cutoff]
        # cropped_sorted_idx.sort()
          
        # Map positional indices back to actual data_dict keys
        kept_keys = [data_keys[i] for i in cropped_sorted_idx]

        # Save histogram
        first = int(n_total * self.LOWEST_PERCENT)
        last = min(int(n_total * self.HIGHEST_PERCENT), len(mean_travel) - 1)

        hist_path = os.path.join(self.get_output_dir(),
                                f'{self.dataset_name.lower()}_{split_name}_posechange.png')
        os.makedirs(os.path.dirname(hist_path), exist_ok=True)

        plt.hist(pose_change, bins=100)
        plt.vlines(np.sort(pose_change)[first], ymin=0, ymax=n_total // 30, colors='r')
        plt.vlines(np.sort(pose_change)[last], ymin=0, ymax=n_total // 30, colors='r')
        plt.savefig(hist_path)
        plt.close()

        hist_path = os.path.join(self.get_output_dir(),
                                 f'{self.dataset_name.lower()}_{split_name}_travel.png')
        os.makedirs(os.path.dirname(hist_path), exist_ok=True)
        
        plt.hist(mean_travel, bins=100)
        plt.vlines(np.sort(mean_travel)[first], ymin=0, ymax=n_total // 30, colors='r')
        plt.vlines(np.sort(mean_travel)[last], ymin=0, ymax=n_total // 30, colors='r')
        plt.savefig(hist_path)
        plt.close()


        print(f"Saved motion variance histogram to {hist_path}")
        
        # Render boundary samples closest to thresholds
        self._render_threshold_boundary_samples(split_name, sorted_idx, pose_change, low_cutoff, high_cutoff)

        self.data_dict = {key: self.data_dict[key] for key in kept_keys}
        print(f"Motion variance filter: {n_total} -> {len(self.data_dict)} sequences "
              f"(removed bottom {self.LOWEST_PERCENT*100:.0f}%, top {self.HIGHEST_PERCENT*100:.0f}%)")

    def _compute_motion_variance(self) -> np.ndarray:
        """Compute per-sample motion variance. Must be overridden in subclasses.

        Returns:
            1D numpy array of shape (len(self.data_dict),) with variance values per sample.
        """
        raise NotImplementedError(
            "Subclasses must implement _compute_motion_variance() to use motion variance filtering."
        )

    def _render_threshold_boundary_samples(self, split_name, sorted_idx, pose_change, low_cutoff, high_cutoff):
        """Render motion samples closest to the filter thresholds. Override in subclasses."""
        pass


class BodyDatasetProcessor(BaseDatasetProcessor):
    """
    Base class specifically for body motion datasets.
    Provides common functionality for full-body SMPLX processing,
    including motion variance filtering based on joint location std.
    """
    
    N_BODY_JOINTS = 22  # root + 21 body joints
    
    def __init__(self, dataset_name: str, dataset_path: str, output_path: str, 
                 config: Optional[ProcessingConfig] = None):
        super().__init__(dataset_name, dataset_path, output_path, config)
        
    def get_output_dir(self) -> str:
        """Get output directory for body datasets."""
        return self.config.BODY_OUTPUT_DIR

    def _get_body_parents(self) -> np.ndarray:
        """Get the SMPLX kinematic parent chain for body joints."""
        if self.body_models is not None:
            parents = self.body_models['neutral'].parents.cpu().numpy()
        else:
            neutral_bm = smplx.create(
                model_path=self.config.SMPLX_NEUTRAL_PATH,
                gender='neutral', batch_size=1,
                # num_betas=10,
            )
            parents = neutral_bm.parents.cpu().numpy()
        return parents[:self.N_BODY_JOINTS]

    def _compute_motion_variance(self) -> np.ndarray:
        """Compute per-sample motion variance based on joint location std.

        Uses forward kinematics (quat_fk) to compute joint positions from
        pose parameters and bone offsets, then measures both the temporal std
        and average total travel distance of joint positions.

        Returns:
            variance: 1D numpy array of temporal std values, one per sample.
            mean_travel: 1D numpy array of mean total joint travel distance, one per sample.
        """
        parents = self._get_body_parents()
        J = self.N_BODY_JOINTS
        T = self.config.WINDOW
        N = len(self.data_dict)

        all_pose_aa = np.zeros((N, T, J, 3), dtype=np.float32)
        all_lpos = np.zeros((N, T, J, 3), dtype=np.float32)
        all_trans = np.zeros((N, T, 3), dtype=np.float32)

        for i, idx in enumerate(self.data_dict):
            v = self.data_dict[idx]
            # root_orient (T,3) + pose_body (T,63) -> (T, 22, 3)
            full_pose = np.concatenate([v['root_orient'], v['pose_body']], axis=-1)
            all_pose_aa[i] = full_pose.reshape(T, J, 3)

            all_lpos[i, :] = v['pos_offset'][:J][None]
            all_lpos[i, :, 0] = v['root_offset']
            all_trans[i] = v['trans']

        lrot = axis_angle_to_quaternion(
            torch.from_numpy(all_pose_aa).float()
        )
        lpos_t = torch.from_numpy(all_lpos).float()

        _, joint_pos = quat_fk(lrot, lpos_t, parents)
        joint_pos = joint_pos + torch.from_numpy(all_trans).float().unsqueeze(2)
        joint_pos = joint_pos.numpy()

        # std across time, mean across joints and xyz
        variance = joint_pos.std(axis=1).mean(axis=1).mean(axis=1)

        # sum of per-step joint displacements, mean across joints -> (N,)
        step_dist = np.linalg.norm(np.diff(joint_pos, axis=1), axis=-1)  # (N, T-1, J)
        mean_travel = step_dist.sum(axis=1).mean(axis=1)  # (N,)

        return variance, mean_travel

    def _render_threshold_boundary_samples(self, split_name, sorted_idx, pose_change, low_cutoff, high_cutoff):
        """Render body motion samples closest to the filter thresholds."""

        output_dir = os.path.join(self.get_output_dir(),
            f'{self.dataset_name.lower()}_{split_name}_threshold_samples')

        if os.path.exists(output_dir):
            for f in os.listdir(output_dir):
                os.remove(os.path.join(output_dir, f))
        else:
            os.makedirs(output_dir, exist_ok=False)

        device = self.get_device()

        if self.renderer is None:
            self.renderer = RendererWrapper(path2body_models='data/body_models')

        color = (100 / 255, 149 / 255, 237 / 255, 1.0)  # cornflower blue

        samples_to_render = []

        # Boundary at lowest threshold
        if low_cutoff > 0:
            samples_to_render.append((
                sorted_idx[low_cutoff - 1],
                f'lowest_discarded_std {pose_change[sorted_idx[low_cutoff - 1]]:.4f}'
            ))
        if low_cutoff < len(sorted_idx):
            samples_to_render.append((
                sorted_idx[low_cutoff],
                f'lowest_kept_std {pose_change[sorted_idx[low_cutoff]]:.4f}'
            ))

        # Boundary at highest threshold
        if high_cutoff < len(sorted_idx):
            samples_to_render.append((
                sorted_idx[high_cutoff],
                f'highest_kept_std {pose_change[sorted_idx[high_cutoff]]:.4f}'
            ))
        if high_cutoff + 1 < len(sorted_idx):
            samples_to_render.append((
                sorted_idx[high_cutoff + 1],
                f'highest_discarded_std {pose_change[sorted_idx[high_cutoff + 1]]:.4f}'
            ))
                
        for sample_idx, label in samples_to_render:
            
            v = self.data_dict[list(self.data_dict.keys())[sample_idx]]
            gender = v.get('gender', 'neutral')

            if self.body_models is None or gender not in self.body_models:
                print(f"Skipping render for {label}: body model not available")
                continue

            bm = self.body_models[gender]
            betas = torch.tensor(v['betas']).float().to(device)
            if betas.ndim == 1:
                betas = betas.unsqueeze(0).expand(self.config.WINDOW, -1)

            body_params = {'global_orient': torch.tensor(v['root_orient']).float().to(device),
                            'body_pose': torch.tensor(v['pose_body']).float().to(device),
                            'transl': torch.tensor(v['trans']).float().to(device),
                            'right_hand_pose': torch.tensor(v['pose_rhand']).float().to(device),
                            'left_hand_pose': torch.tensor(v['pose_lhand']).float().to(device),
                            'betas': betas}
        
            output = bm(**body_params)
            vertices = output.vertices.cpu().detach().numpy()

            mesh_dict = {'vertices': vertices, 'faces': bm.faces}

            root_orient = torch.tensor(v['root_orient']).float()
            root_orient[:, 1] = 0
            root_orient[:, 2] = 0

            camera_dict = {
                'camera_transl': torch.from_numpy(
                    np.array(v['trans'], dtype=np.float32)
                ).float(),
                'camera_rot': root_orient,
                'coef': 2.0,
            }

            filename = os.path.join(output_dir, f'{label}')
            self.renderer.render_motion(
                [mesh_dict], filename=filename,
                camera_dict=camera_dict, color=[color],
                rendering_scale=2.0,
            )
            print(f"Rendered boundary sample: {label}")

       


class HandDatasetProcessor(BaseDatasetProcessor):
    """
    Base class specifically for hand motion datasets.
    Provides common functionality for MANO hand processing, including:
    - Pose augmentation (left-right hand mirroring)
    - Time augmentation (temporal reversal)
    - Motion variance filtering based on 6D rotation std
    """
    
    def __init__(self, dataset_name: str, dataset_path: str, output_path: str,
                 config: Optional[ProcessingConfig] = None):
        super().__init__(dataset_name, dataset_path, output_path, config)
        
    def get_output_dir(self) -> str:
        """Get output directory for hand datasets."""
        return self.config.HAND_OUTPUT_DIR
    
    @staticmethod
    def mirror_left_to_right(
        hand_trans, hand_pose, hand_orient, **extra_arrays
    ):
        """Mirror left hand data to right hand convention.
        
        Flips x-axis of translation and y,z components of axis-angle rotations.
        Works with both numpy arrays and torch tensors.
        
        Args:
            hand_trans: (T, 3) hand translation
            hand_pose: (T, 45) hand pose in axis-angle
            hand_orient: (T, 3) hand orientation in axis-angle
            **extra_arrays: Additional arrays to mirror (e.g. relative_wrist_orient)
            
        Returns:
            (trans, pose, orient, mirrored_extra_dict)
        """
        is_numpy = isinstance(hand_trans, np.ndarray)
        
        trans = hand_trans.copy() if is_numpy else hand_trans.clone()
        trans[:, 0] *= -1
        
        if is_numpy:
            pose = hand_pose.copy().reshape(-1, 15, 3)
        else:
            pose = hand_pose.clone().reshape(-1, 15, 3)
        pose[..., 1:] *= -1
        pose = pose.reshape(-1, 45)
        
        orient = hand_orient.copy() if is_numpy else hand_orient.clone()
        orient[..., 1:] *= -1
        
        mirrored_extra = {}
        for key, arr in extra_arrays.items():
            m = arr.copy() if isinstance(arr, np.ndarray) else arr.clone()
            m[..., 1:] *= -1
            mirrored_extra[key] = m
        
        return trans, pose, orient, mirrored_extra
    
    @staticmethod
    def flip_time(*arrays):
        """Reverse temporal order of arrays (numpy or torch)."""
        results = []
        for arr in arrays:
            if isinstance(arr, torch.Tensor):
                results.append(torch.flip(arr, dims=(0,)))
            elif isinstance(arr, np.ndarray):
                results.append(np.flip(arr, axis=0))
            else:
                results.append(arr)
        return tuple(results) if len(results) > 1 else results[0]
    
    def _compute_motion_variance(self) -> np.ndarray:
        """Compute per-sample motion variance using 6D rotation std of hand pose.

        Returns:
            1D numpy array of variance values, one per sample in data_dict.
        """
        pose_rhand_cat = np.concatenate(
            [self.data_dict[idx]['pose_rhand'][None] for idx in self.data_dict.keys()], axis=0
        )
        pose_rhand_6d = axis_angle_to_rotation_6d(
            torch.from_numpy(pose_rhand_cat).reshape(-1, self.config.WINDOW, 15, 3)
        ).numpy()
        return pose_rhand_6d.std(1).mean(1).mean(1), None

    def _render_threshold_boundary_samples(self, split_name, sorted_idx, pose_change, low_cutoff, high_cutoff):
        """Render motion samples closest to the lowest and highest filter thresholds.
        
        For each threshold, renders the last-discarded and first-kept samples as videos.
        
        Args:
            sorted_idx: Indices into data_dict sorted by ascending pose_change
            pose_change: Array of motion variance values per sample
            low_cutoff: Index into sorted_idx where the lowest threshold cuts
        """
    
        output_dir = os.path.join(self.config.HAND_OUTPUT_DIR,
            f'{self.dataset_name.lower()}_{split_name}_threshold_samples')
        
        # clear existing samples
        if os.path.exists(output_dir):
            for f in os.listdir(output_dir):
                os.remove(os.path.join(output_dir, f))
        else:
            os.makedirs(output_dir, exist_ok=False)
 
        device = self.get_device()
        
        mano_model = smplx.create(
            model_path=self.config.RIGHT_MANO_PATH,
            model_type='mano', is_rhand=True, use_pca=False,
            flat_hand_mean=True, batch_size=self.config.WINDOW,
            num_betas=10
        ).to(device)
        
        renderer = RendererWrapper(self.config.RIGHT_MANO_PATH)
        color = (160 / 255, 160 / 255, 0 / 255, 1.0)
        
        samples_to_render = []
        
        # Boundary at lowest threshold
        if low_cutoff > 0:
            # Last discarded sample (just below threshold)
            samples_to_render.append((
                sorted_idx[low_cutoff - 1],
                f'lowest_discarded_std{pose_change[sorted_idx[low_cutoff - 1]]:.4f}'
            ))
        if low_cutoff < len(sorted_idx):
            # First kept sample (just above threshold)
            samples_to_render.append((
                sorted_idx[low_cutoff],
                f'lowest_kept_std{pose_change[sorted_idx[low_cutoff]]:.4f}'
            ))
        
                
        if high_cutoff < len(sorted_idx):
            # Last kept sample (just below highest threshold)
            samples_to_render.append((
                sorted_idx[high_cutoff],
                f'highest_kept_std {pose_change[sorted_idx[high_cutoff]]:.4f}'
            ))
        if high_cutoff + 1 < len(sorted_idx):
            # First discarded sample (just above highest threshold)
            samples_to_render.append((
                sorted_idx[high_cutoff + 1],
                f'highest_discarded_std {pose_change[sorted_idx[high_cutoff + 1]]:.4f}'
            ))
    
        for sample_idx, label in samples_to_render:
            try:
                v = self.data_dict[sample_idx]
                
                motion_params = {
                    'hand_pose': torch.tensor(v['pose_rhand']).float().to(device),
                }
                vertices = mano_model(**motion_params).vertices 
                
                mesh_dict = {
                    'vertices': vertices.cpu().detach().numpy() + np.array([0, 0, 0.2]),  # elevate for better visualization
                    'faces': mano_model.faces,
                }
                
                root_orient = torch.tensor(v['root_orient']).float()
                root_orient[:, 1] = 0
                root_orient[:, 2] = 0
                
                camera_dict = {
                    'camera_transl': torch.from_numpy(
                        np.array(v['trans'], dtype=np.float32)
                    ).float(),
                    'camera_rot': root_orient,
                    'coef': 0.1,
                }
                
                filename = os.path.join(output_dir, f'{label}')
                renderer.render_motion(
                    [mesh_dict], filename=filename,
                    camera_dict=camera_dict, color=[color], 
                    rendering_scale=0.2, 
                )
                print(f"Rendered boundary sample: {label}")
                
            except Exception as e:
                print(f"Failed to render boundary sample {label}: {e}")
                continue
    
    def run(self):
        """Main processing pipeline with motion variance filtering."""
        print(f"Starting {self.dataset_name} hand dataset processing...")
        
        # Cleanup
        self.cleanup_data()
        
        # Remove existing output files
        for path in [self.output_path, self.output_path.replace('.p', '_processed.p')]:
            if os.path.exists(path):
                os.remove(path)
        
        # Load sequences
        print("Loading sequences...")
        sequences = self.load_sequences()
        print(f"Found {len(sequences)} sequences/items")
        
        # Filter into splits
        filtered_splits = self.filter_sequences(sequences)
        
        # Process each split
        for split_name, split_sequences in filtered_splits.items():
            print(f"\nProcessing {split_name} split ({len(split_sequences)} sequences)...")
            self.reset_data_dict()
            
            for seq in tqdm(split_sequences, desc=f"Processing {split_name}"):
                try:
                    result = self.process_sequence(seq)
                    if result is not None:
                        self.add_to_data_dict(result)
                except Exception as e:
                    print(f"\nError processing sequence: {str(e)}")
                    continue
            
            # Apply motion variance filtering
            self.apply_motion_variance_filter()
            
            # Save
            if len(self.data_dict) > 0:
                os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
                print(f"Saving {len(self.data_dict)} sequences to {self.output_path}")
                joblib.dump(self.data_dict, self.output_path)
            else:
                print(f"Warning: No valid sequences in {split_name} split")
        
        print(f"\n{self.dataset_name} hand processing complete!")
    
    def add_to_data_dict(self, processed_data):
        """Handle lists of chunks or single chunks."""
        if isinstance(processed_data, list):
            for chunk_data in processed_data:
                chunk_data['motion_no'] = self.motion_idx
                super().add_to_data_dict(chunk_data)
        else:
            processed_data['motion_no'] = self.motion_idx
            super().add_to_data_dict(processed_data)
