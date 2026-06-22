"""
Shared utility functions and classes for dataset processing.
Contains model loaders, validators, and common helper functions.
"""
import os
import sys
import torch
import smplx
import trimesh
import numpy as np
from typing import Dict, Optional, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.scripts.process.config import ProcessingConfig
from src.render.mesh_viz import RendererWrapper
from src.utils.transforms3d import loc2vel
from src.utils.transforms import axis_angle_to_matrix


class ModelLoader:
    """Centralized model loading utilities."""
    
    @staticmethod
    def load_smplx_models(
        batch_size: int,
        num_betas: int,
        device: torch.device,
        config: Optional[ProcessingConfig] = None
    ) -> Dict[str, smplx.SMPLX]:
        """
        Load SMPLX body models for all genders.
        
        Args:
            batch_size: Batch size for the models
            num_betas: Number of shape parameters
            device: Device to load models on
            config: Configuration object
            
        Returns:
            Dictionary mapping gender to SMPLX model
        """
        config = config or ProcessingConfig()
        
        male_bm = smplx.create(
            model_path=config.SMPLX_MALE_PATH,
            gender='male',
            num_betas=num_betas,
            batch_size=batch_size,
            flat_hand_mean=True,
            use_pca=False
        ).to(device)
        
        female_bm = smplx.create(
            model_path=config.SMPLX_FEMALE_PATH,
            gender='female',
            num_betas=num_betas,
            batch_size=batch_size,
            flat_hand_mean=True,
            use_pca=False
        ).to(device)
        
        neutral_bm = smplx.create(
            model_path=config.SMPLX_NEUTRAL_PATH,
            gender='neutral',
            num_betas=num_betas,
            batch_size=batch_size,
            flat_hand_mean=True,
            use_pca=False
        ).to(device)
        
        # Freeze parameters
        for model in [male_bm, female_bm, neutral_bm]:
            for p in model.parameters():
                p.requires_grad = False
        
        return {
            'male': male_bm,
            'female': female_bm,
            'neutral': neutral_bm
        }
    
    @staticmethod
    def get_default_vtemplates(
        body_models: Dict[str, smplx.SMPLX]
    ) -> Dict[str, torch.Tensor]:
        """
        Extract default vertex templates from body models.
        
        Args:
            body_models: Dictionary of SMPLX models
            
        Returns:
            Dictionary mapping gender to vertex template
        """
        return {
            gender: model.v_template 
            for gender, model in body_models.items()
        }
    
    @staticmethod
    def load_mano_mean_poses(
        config: Optional[ProcessingConfig] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load MANO mean hand poses.
        
        Args:
            config: Configuration object
            
        Returns:
            Tuple of (right_hand_mean_pose, left_hand_mean_pose)
        """
        config = config or ProcessingConfig()
        
        r_mean_pose = smplx.create(
            model_path=config.RIGHT_MANO_PATH,
            model_type='mano',
            is_rhand=True,
            use_pca=False,
            ext='pkl'
        ).hand_mean.detach().cpu().numpy()
        
        l_mean_pose = smplx.create(
            model_path=config.LEFT_MANO_PATH,
            model_type='mano',
            is_rhand=False,
            use_pca=False,
            ext='pkl'
        ).hand_mean.detach().cpu().numpy()
        
        return r_mean_pose, l_mean_pose
    
    @staticmethod
    def create_renderer(
        config: Optional[ProcessingConfig] = None
    ) -> RendererWrapper:
        """
        Create a renderer for visualization.
        
        Args:
            config: Configuration object
            
        Returns:
            Initialized RendererWrapper
        """
        config = config or ProcessingConfig()
        return RendererWrapper(path2body_models='data/body_models')


class DataValidator:
    """Common validation and filtering functions."""
    
    @staticmethod
    def filter_grasp_frames_by_velocity(
        body_pose: np.ndarray,
        fps: float = 120.0,
        velocity_threshold: float = 0.6,
        target_fps: int = 30
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Filter frames based on hand angular velocity.
        
        Args:
            body_pose: Body pose parameters (T, D)
            fps: Original framerate
            velocity_threshold: Threshold for start detection
            target_fps: Target framerate for downsampling
            
        Returns:
            Tuple of (grasp_motion_frames, obj_moving_frames) or (None, frames)
        """
        # Extract right wrist pose (joint 17)
        hand = body_pose[:, 17 * 3:18 * 3]
        hand_rotmat = axis_angle_to_matrix(torch.tensor(hand))
        hand_ang_vel = loc2vel(hand_rotmat, fps=fps).abs().norm(dim=-1).norm(dim=-1).squeeze()
        
        idxs = np.arange(body_pose.shape[0])
        
        # Find start frame based on velocity
        start_fil = hand_ang_vel > velocity_threshold
        if start_fil.sum() < 2:
            return None, np.zeros(body_pose.shape[0], dtype=bool)
        
        start_frame = idxs[start_fil][1]
        start_fil = idxs > start_frame
        
        # Downsample to target FPS
        skip_frame = int(fps // target_fps)
        fps_fil = (idxs % skip_frame) == 0
        
        grasp_motion_frames = fps_fil * start_fil
        obj_moving_frames = fps_fil
        
        return grasp_motion_frames, obj_moving_frames
    
    @staticmethod
    def filter_grasp_frames_by_contact(
        contact_array: np.ndarray,
        obj_height: np.ndarray,
        table_height: float,
        target_fps: int = 30,
        original_fps: float = 120.0,
        height_threshold: float = 0.004
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Filter frames based on object contact or height.
        
        Args:
            contact_array: Contact indicator array (T, N)
            obj_height: Object height over time (T,)
            table_height: Height of the table surface
            target_fps: Target framerate
            original_fps: Original framerate
            height_threshold: Height threshold above table
            
        Returns:
            Tuple of (grasp_motion_frames, obj_moving_frames)
        """
        idxs = np.arange(obj_height.shape[0])
        
        # Filter by height or contact
        if contact_array is not None and contact_array.mean() > 0:
            fil = (contact_array.mean(axis=1) > 0)
        else:
            fil = obj_height > (table_height + height_threshold)
        
        if fil.sum() < 1:
            return None, fil
        
        # Downsample
        skip_frame = int(original_fps // target_fps)
        fps_fil = (idxs % skip_frame) == 0
        
        grasp_frame = idxs[fil][0]
        grasp_frames = idxs > grasp_frame
        
        grasp_motion_frames = grasp_frames * fps_fil
        obj_moving_frames = fil * fps_fil
        
        return grasp_motion_frames, obj_moving_frames
    
    @staticmethod
    def validate_sequence_length(
        seq_length: int,
        min_length: int = 10,
        max_length: Optional[int] = None
    ) -> bool:
        """
        Validate sequence length.
        
        Args:
            seq_length: Length of the sequence
            min_length: Minimum acceptable length
            max_length: Maximum acceptable length (None = no limit)
            
        Returns:
            True if sequence length is valid
        """
        if seq_length < min_length:
            return False
        if max_length is not None and seq_length > max_length:
            return False
        return True


class ObjectLoader:
    """Utilities for loading and processing object meshes."""
    
    def __init__(self):
        self.obj_cache = {}
    
    def load_and_simplify_mesh(
        self,
        mesh_path: str,
        n_verts_sample: int = 2048,
        scale: float = 1.0,
        cache_key: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Load and simplify an object mesh.
        
        Args:
            mesh_path: Path to the mesh file
            n_verts_sample: Target number of vertices for simplified mesh
            scale: Scale factor for vertices
            cache_key: Key for caching (e.g., object name)
            
        Returns:
            Dictionary with mesh data
        """
        if cache_key and cache_key in self.obj_cache:
            return self.obj_cache[cache_key]
        
        # Load mesh
        obj_mesh = trimesh.load(file_obj=mesh_path, process=False)
        verts_obj = np.array(obj_mesh.vertices) * scale
        faces_obj = np.array(obj_mesh.faces)
        
        # Simplify mesh
        n_faces = max(faces_obj.shape[0] // 10, n_verts_sample)
        try:
            mesh_simplified = obj_mesh.simplify_quadric_decimation(n_faces)
            verts_sample = np.array(mesh_simplified.vertices) * scale
            faces_sample = np.array(mesh_simplified.faces)
        except:
            # If simplification fails, use original
            verts_sample = verts_obj
            faces_sample = faces_obj
        
        # Compute center of mass offset
        obj_com_offset = verts_obj.mean(axis=0)
        
        result = {
            'verts': verts_obj,
            'faces': faces_obj,
            'verts_sample': verts_sample,
            'faces_sample': faces_sample,
            'obj_com_offset': obj_com_offset,
            'obj_mesh_file': mesh_path
        }
        
        # Cache if key provided
        if cache_key:
            self.obj_cache[cache_key] = result
        
        return result


class SequenceChunker:
    """Utility for chunking long sequences into windows."""
    
    @staticmethod
    def chunk_sequence(
        seq_length: int,
        window_size: int,
        overlap: int = 0
    ) -> list:
        """
        Generate chunk indices for a sequence.
        
        Args:
            seq_length: Total sequence length
            window_size: Size of each chunk
            overlap: Overlap between consecutive chunks
            
        Returns:
            List of (start, end) tuples for each chunk
        """
        chunks = []
        stride = window_size - overlap
        
        for start in range(0, seq_length - window_size + 1, stride):
            end = start + window_size
            chunks.append((start, end))
        
        # Add final chunk if needed
        if chunks and chunks[-1][1] < seq_length:
            chunks.append((seq_length - window_size, seq_length))
        elif not chunks and seq_length > 0:
            # Sequence shorter than window
            chunks.append((0, seq_length))
        
        return chunks


class PathUtils:
    """Utilities for file path operations."""
    
    @staticmethod
    def ensure_dir(path: str):
        """Create directory if it doesn't exist."""
        os.makedirs(path, exist_ok=True)
    
    @staticmethod
    def get_sequence_name(file_path: str) -> str:
        """Extract sequence name from file path."""
        return os.path.splitext(os.path.basename(file_path))[0]
    
    @staticmethod
    def change_extension(file_path: str, new_ext: str) -> str:
        """Change file extension."""
        base = os.path.splitext(file_path)[0]
        return f"{base}.{new_ext.lstrip('.')}"
