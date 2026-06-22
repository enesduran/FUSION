"""
OMOMO Dataset Processor - Refactored version using new architecture.
OMOMO contains body-object manipulation sequences with articulated objects.
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
from src.scripts.process.config import (
    ProcessingConfig, 
    SMPLX_JOINT_MIRROR_ARR, 
    DATASET_PATHS
)
from src.scripts.process.processor_utils import ModelLoader, SequenceChunker
from src.utils.process_utils import BRANCH_NAME, determine_floor_height_and_contacts
from src.utils.transforms3d import transform_body_pose
from src.utils.transforms import axis_angle_to_rotation_6d
from src.utils.data_utils import apply_transformation_to_obj_geometry


# Objects with two parts (top and bottom)
TWO_PART_OBJECTS = ["mop", "vacuum"]


class OmomoProcessor(BodyDatasetProcessor):
    """Processor for OMOMO object manipulation dataset."""
    
    def __init__(self):
        dataset_path = DATASET_PATHS.get('omomo', 'data/motion/Body_Raw/OMOMO')
        output_path = 'data/motion/Body_Processed/omomo.p'
        
        super().__init__(
            dataset_name='OMOMO',
            dataset_path=dataset_path,
            output_path=output_path
        )
        
        self.LOWEST_PERCENT = 0.10  
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
        
        # Cache for shifted object centers
        self.shifted_object_dicts = {}
    
    def cleanup_data(self):
        """OMOMO doesn't require specific cleanup."""
        print('No cleanup needed for OMOMO')
    
    def load_sequences(self) -> List[str]:
        """
        Load OMOMO sequences from preprocessed pickle files.
        OMOMO data comes as preprocessed .p files rather than individual sequences.
        """
        # OMOMO sequences are stored in preprocessed pickle files
        sequences = []
        
        # Train split (commented out in original, keeping for consistency)
        train_path = os.path.join(self.dataset_path, 'train_diffusion_manip_seq_joints24.p')
        if os.path.exists(train_path):
            sequences.append(('train', train_path))
        
        # Val/test split
        val_path = os.path.join(self.dataset_path, 'test_diffusion_manip_seq_joints24.p')
        if os.path.exists(val_path):
            sequences.append(('val', val_path))
        
        return sequences
    
    def filter_sequences(self, sequences: List[str]) -> Dict[str, List[str]]:
        """
        OMOMO sequences are already split by file.
        Return them organized by split name.
        """
        filtered = {}
        for split_name, file_path in sequences:
            if split_name not in filtered:
                filtered[split_name] = []
            filtered[split_name].append(file_path)
        
        return filtered if filtered else {'val': sequences}
    
    def process_sequence(self, sequence_path: str) -> Optional[List[Dict[str, Any]]]:
        """
        Process OMOMO sequences from pickle file.
        Each pickle file contains multiple sequences as a dictionary.
        """
   
        import joblib
        
        # Load the pickle file containing multiple sequences
        datum = joblib.load(sequence_path)
        
        all_results = []
         
        # Process each sequence in the file
        for idx in datum.keys():
            data = datum[idx]
            
            results = self._process_single_sequence(data)
            if results:
                all_results.extend(results)
        
        # Post-process: filter by motion variance (OMOMO-specific)
        all_results = self._filter_by_motion_variance(all_results)
        
        return all_results if all_results else None
            
  
    
    def _process_single_sequence(self, data: Dict) -> Optional[List[Dict[str, Any]]]:
        """Process a single OMOMO sequence."""
        sample_freq = 1
        seq_name = data["seq_name"]
        object_name = seq_name.split('_')[1]
        
        # Extract basic info        
        betas = np.concatenate([data['betas'][0], np.zeros(self.config.NUM_BETAS - len(data['betas'][0]))])
        gender = data['gender'].item()
        
        # Extract motion data
        trans = data['trans'][::sample_freq]
        pose_body = data['pose_body'][::sample_freq]
        root_orient = data['root_orient'][::sample_freq]
        
        # Extract object data
        obj_trans = data['obj_trans'][::sample_freq].squeeze(-1)
        obj_scale = data['obj_scale'][::sample_freq]
        obj_rot = transform_body_pose(data['obj_rot'][::sample_freq], 'rot->aa').numpy()
        obj_arti = np.zeros((obj_trans.shape[0]))
        
        T = len(pose_body)
        
        # Skip short sequences
        if T < self.config.WINDOW:
            return None
        
        # Check if object has two parts
        two_parts_object = object_name in TWO_PART_OBJECTS
        
        if two_parts_object:
            obj_bottom_trans = data['obj_bottom_trans'][:, :, 0][::sample_freq]
            obj_bottom_rot = transform_body_pose(data['obj_bottom_rot'][::sample_freq], 'rot->aa').numpy()
            obj_bottom_scale = data['obj_bottom_scale'][::sample_freq]
        else:
            obj_bottom_trans = None
            obj_bottom_rot = None
            obj_bottom_scale = None
        
        # Create full pose (add zero poses for jaw, eye, hands)
        pose_jaw = np.zeros((T, 3))
        pose_eye = np.zeros((T, 6))
        pose_rhand = pose_lhand = np.zeros((T, 45))
        
        fullpose = np.concatenate([
            root_orient, pose_body, pose_jaw, pose_eye, pose_lhand, pose_rhand
        ], axis=1)
        
        # Compute rest pose offsets
        bm_rest_out = self.body_models[gender](
            betas=torch.from_numpy(betas[None, :]).float().to(self.device)
        )
        rest_joint_pos = bm_rest_out.joints.cpu().detach().numpy()[0, :len(SMPLX_JOINT_MIRROR_ARR)]
        root_offset = rest_joint_pos[0]
        
        pos_offset = [[0, 0, 0]]
        for child, parent in enumerate(self.body_models[gender].parents):
            if parent == -1:
                continue
            pos_offset.append(rest_joint_pos[child] - rest_joint_pos[parent])
        pos_offset = np.vstack(pos_offset)
        
        # Get object paths
        object_mesh_path, object_simplified_mesh_path = self._get_object_paths(
            object_name, two_parts_object
        )
        
        # Shift object to center (with caching)
        shift_value = self._get_or_compute_object_shift(
            object_name, object_mesh_path, object_simplified_mesh_path, two_parts_object
        )
        
        # Process both original and augmented versions
        results = []
        for augment_flag in [False, True]:
            result = self._process_single_variant(
                fullpose, trans, obj_trans, obj_rot, obj_scale, obj_arti,
                obj_bottom_trans, obj_bottom_rot, obj_bottom_scale,
                betas, gender, seq_name, object_name,
                object_mesh_path, object_simplified_mesh_path,
                shift_value, two_parts_object, root_offset, pos_offset,
                augment_flag, T
            )
            if result:
                results.extend(result)
        
        return results if results else None
    
    def _get_object_paths(self, object_name: str, two_parts: bool) -> tuple:
        """Get paths to object mesh files."""
        if two_parts:
            mesh_path = f'{self.dataset_path}/captured_objects/{object_name}_cleaned_simplified_top.obj'
            simplified_path = f'{self.dataset_path}/captured_objects_simplified/{object_name}_cleaned_simplified_top.obj'
        else:
            mesh_path = f'{self.dataset_path}/captured_objects/{object_name}_cleaned_simplified.obj'
            simplified_path = f'{self.dataset_path}/captured_objects_simplified/{object_name}_cleaned_simplified.obj'
        
        return mesh_path, simplified_path
    
    def _get_or_compute_object_shift(
        self, 
        object_name: str,
        object_mesh_path: str,
        simplified_mesh_path: str,
        two_parts: bool
    ) -> torch.Tensor:
        """Get or compute object shift to center."""
        if object_name in self.shifted_object_dicts:
            return self.shifted_object_dicts[object_name]
        
        print(f'Processing {object_name}')
        
        # Load object mesh
        obj = trimesh.load(object_mesh_path, process=False)
        shift_value = obj.vertices.mean(0)
        
        # Save shifted object
        os.makedirs(os.path.dirname(simplified_mesh_path), exist_ok=True)
        trimesh.Trimesh(
            vertices=obj.vertices - shift_value,
            faces=obj.faces
        ).export(simplified_mesh_path)
        
        # Handle two-part objects
        if two_parts:
            object_bottom_path = object_mesh_path.replace('top', 'bottom')
            save_bottom_path = simplified_mesh_path.replace('top', 'bottom')
            
            obj_bottom = trimesh.load(object_bottom_path, process=False)
            trimesh.Trimesh(
                vertices=obj_bottom.vertices - shift_value,
                faces=obj_bottom.faces
            ).export(save_bottom_path)
        
        shift_tensor = torch.tensor(shift_value).float()
        self.shifted_object_dicts[object_name] = shift_tensor
        
        return shift_tensor
    
    def _process_single_variant(
        self,
        fullpose: np.ndarray,
        trans: np.ndarray,
        obj_trans: np.ndarray,
        obj_rot: np.ndarray,
        obj_scale: np.ndarray,
        obj_arti: np.ndarray,
        obj_bottom_trans: Optional[np.ndarray],
        obj_bottom_rot: Optional[np.ndarray],
        obj_bottom_scale: Optional[np.ndarray],
        betas: np.ndarray,
        gender: str,
        seq_name: str,
        object_name: str,
        object_mesh_path: str,
        object_simplified_mesh_path: str,
        shift_value: torch.Tensor,
        two_parts_object: bool,
        root_offset: np.ndarray,
        pos_offset: np.ndarray,
        augment_flag: bool,
        T: int
    ) -> Optional[List[Dict[str, Any]]]:
        """Process original or augmented version of a sequence."""

        # --- Apply object shift transformation BEFORE augmentation ---
        # The shift converts from original-mesh coords to simplified-mesh coords:
        #   new_trans = obj_trans + scale * R @ shift_value
        # This must happen before augmentation so the YZ-plane reflection
        # of obj_trans correctly reflects the shifted (world-space COM) position.
        obj_rot_mat = transform_body_pose(torch.tensor(obj_rot), "aa->rot")
        delta_trans = torch.tensor(obj_scale[:, None]) * (obj_rot_mat @ shift_value)

        # Verify: original mesh with original trans == simplified mesh with shifted trans
        aa, _ = apply_transformation_to_obj_geometry(
            object_mesh_path,
            torch.tensor(obj_scale),
            obj_rot_mat,
            torch.tensor(obj_trans)
        )
        obj_trans_shifted = obj_trans + delta_trans.numpy()
        bb, _ = apply_transformation_to_obj_geometry(
            object_simplified_mesh_path,
            torch.tensor(obj_scale),
            obj_rot_mat,
            torch.tensor(obj_trans_shifted)
        )
        assert np.allclose(aa, bb, atol=1e-5), "Object transformation mismatch"

        # Shift bottom part too
        if two_parts_object:
            obj_bottom_rot_mat = transform_body_pose(torch.tensor(obj_bottom_rot), "aa->rot")
            delta_trans_bottom = torch.tensor(obj_bottom_scale[:, None]) * (obj_bottom_rot_mat @ shift_value)
            obj_bottom_trans_shifted = obj_bottom_trans + delta_trans_bottom.numpy()

        # --- Apply augmentation (YZ-plane reflection) ---
        if augment_flag:
            trans_ = trans.copy()
            trans_[:, 0] *= -1

            fullpose_ = fullpose.copy().reshape(-1, len(SMPLX_JOINT_MIRROR_ARR), 3)
            fullpose_ = fullpose_[:, SMPLX_JOINT_MIRROR_ARR]
            fullpose_[..., 1:] *= -1
            fullpose_ = fullpose_.reshape(-1, len(SMPLX_JOINT_MIRROR_ARR) * 3)

            obj_trans_ = obj_trans_shifted.copy()
            obj_trans_[:, 0] *= -1

            obj_rot_ = obj_rot.copy()
            obj_rot_[..., 1:] *= -1

            if two_parts_object:
                obj_bottom_trans_ = obj_bottom_trans_shifted.copy()
                obj_bottom_trans_[:, 0] *= -1

                obj_bottom_rot_ = obj_bottom_rot.copy()
                obj_bottom_rot_[..., 1:] *= -1
        else:
            trans_ = trans.copy()
            fullpose_ = fullpose.copy()
            obj_trans_ = obj_trans_shifted.copy()
            obj_rot_ = obj_rot.copy()

            if two_parts_object:
                obj_bottom_trans_ = obj_bottom_trans_shifted.copy()
                obj_bottom_rot_ = obj_bottom_rot.copy()
        
        # Split into body parts
        root_orient_ = fullpose_[..., :3]
        pose_body_ = fullpose_[..., 3:66]
        pose_jaw_ = fullpose_[..., 66:69]
        pose_eye_ = fullpose_[..., 69:75]
        pose_lhand_ = fullpose_[..., 75:120]
        pose_rhand_ = fullpose_[..., 120:]
        
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
            
            # Generate mesh
            bm_out = self.body_models[gender](**motion_params_gt)
            joints = bm_out.joints.cpu().detach().numpy()[:, :len(SMPLX_JOINT_MIRROR_ARR)]
            
            # Determine contacts
            offset_floor_height, contacts, discard_seq = \
                determine_floor_height_and_contacts(joints, self.config.TARGET_FPS)
            
            if discard_seq:
                # Optionally visualize discarded sequences
                continue
            
            # Store processed data
            chunk_data = {
                'betas': betas,
                'gender': gender,
                'seq_name': seq_name,
                'body_dataset_name': 'OMOMO',
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
                'obj_arti': obj_arti[chunk_slice],
                'obj_scale': obj_scale[chunk_slice]
            }
            
            # Add two-part object data if applicable
            if two_parts_object:
                chunk_data['obj_bottom_scale'] = obj_bottom_scale[chunk_slice]
                chunk_data['obj_bottom_trans'] = obj_bottom_trans_[chunk_slice]
                chunk_data['obj_bottom_orient'] = obj_bottom_rot_[chunk_slice]
            
            results.append(chunk_data)
            self.motion_idx += 1
        
        return results if results else None
    
    def _filter_by_motion_variance(
        self, 
        results: List[Dict[str, Any]],
        lowest_percent: float = 0.10
    ) -> List[Dict[str, Any]]:
        """
        Filter out sequences with low motion variance (OMOMO-specific).
        Removes the lowest 10% of sequences by pose variation.
        """
        if not results:
            return results
        
        # Compute motion variance for each sequence
        pose_body_list = [r['pose_body'] for r in results]
        pose_body = np.array(pose_body_list)  # (N, T, 63)
        
        # Convert to 6D rotation representation
        pose_body_6d = axis_angle_to_rotation_6d(
            torch.from_numpy(pose_body).reshape(-1, self.config.WINDOW, 21, 3)
        ).numpy()
        
        # Compute standard deviation across time, then average across joints and dims
        pose_change = pose_body_6d.std(1).mean(1).mean(1)
        
        # Sort by variance
        sorted_idx = np.argsort(pose_change)
        
        # Crop lower percentage
        cropped_sorted_idx = sorted_idx[int(lowest_percent * len(sorted_idx)):]
        cropped_sorted_idx.sort()
        
        # Filter results
        filtered_results = [results[idx] for idx in cropped_sorted_idx]
        
        print(f"Filtered from {len(results)} to {len(filtered_results)} sequences "
              f"(removed lowest {lowest_percent*100:.0f}% by motion variance)")
        
        return filtered_results
    
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
    
    def simplify_meshes(self):
        """
        Simplify all object meshes in the captured_objects_simplified directory.
        This is run after processing to reduce mesh complexity.
        """
        print("Simplifying object meshes...")
        
        mesh_pattern = os.path.join(
            self.dataset_path, 
            'captured_objects_simplified/*_cleaned_simplified.obj'
        )
        
        for obj_path in glob.glob(mesh_pattern):
            try:
                raw_obj_mesh = trimesh.load(obj_path, process=False)
                num_faces = max(2000, raw_obj_mesh.faces.shape[0] // 10)
                mesh_simplified = raw_obj_mesh.simplify_quadric_decimation(num_faces)
                mesh_simplified.export(obj_path)
                print(f"Simplified: {os.path.basename(obj_path)}")
            except Exception as e:
                print(f"Error simplifying {obj_path}: {e}")
        
        print("Done simplifying meshes")
    
    def add_to_data_dict(self, processed_data: Any):
        """Override to handle list of chunks."""
        if isinstance(processed_data, list):
            for chunk_data in processed_data:
                super().add_to_data_dict(chunk_data)
        else:
            super().add_to_data_dict(processed_data)


if __name__ == '__main__':
    processor = OmomoProcessor()
    processor.run()
    
    # Simplify meshes after processing
    processor.simplify_meshes()
    
    print('Done processing OMOMO')
