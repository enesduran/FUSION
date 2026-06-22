"""
Shared configuration for all dataset processing scripts.
Contains constants, paths, and common mappings used across different datasets.
"""
import numpy as np
from dataclasses import dataclass
from typing import Dict


@dataclass
class ProcessingConfig:
    """Common configuration parameters for dataset processing."""
    
    # Processing parameters
    TARGET_FPS: int = 30
    WINDOW: int = 120
    NUM_BETAS: int = 300
    
    # Device configuration
    DEVICE: str = "cuda:0"
    
    # Model paths
    SMPLX_MALE_PATH: str = "data/body_models/smplx/SMPLX_MALE.npz"
    SMPLX_FEMALE_PATH: str = "data/body_models/smplx/SMPLX_FEMALE.npz"
    SMPLX_NEUTRAL_PATH: str = "data/body_models/smplx/SMPLX_NEUTRAL.npz"
    RIGHT_MANO_PATH: str = "data/body_models/mano/MANO_RIGHT.pkl"
    LEFT_MANO_PATH: str = "data/body_models/mano/MANO_LEFT.pkl"
    
    # Output paths
    BODY_OUTPUT_DIR: str = "data/motion/Body_Processed"
    HAND_OUTPUT_DIR: str = "data/motion/Hand_Processed"


# SMPLX joint mirroring dictionary for left-right symmetry
SMPLX_JOINT_MIRROR_DICT: Dict[int, int] = {
    0: 0, 1: 2, 2: 1, 3: 3, 4: 5, 5: 4, 6: 6, 7: 8, 8: 7, 9: 9, 10: 11, 11: 10, 
    12: 12, 13: 14, 14: 13, 15: 15, 16: 17, 17: 16, 18: 19, 19: 18, 20: 21, 21: 20, 
    22: 22, 23: 24, 24: 23, 25: 40, 26: 41, 27: 42, 28: 43, 29: 44, 30: 45, 31: 46, 
    32: 47, 33: 48, 34: 49, 35: 50, 36: 51, 37: 52, 38: 53, 39: 54, 40: 25, 41: 26,
    42: 27, 43: 28, 44: 29, 45: 30, 46: 31, 47: 32, 48: 33, 49: 34, 50: 35, 51: 36, 
    52: 37, 53: 38, 54: 39
}

# Array version for efficient indexing
SMPLX_JOINT_MIRROR_ARR = np.array(list(SMPLX_JOINT_MIRROR_DICT.values()), dtype=np.int32)


# Contact indices for body contact detection
CONTACT_INDICES = [7, 8, 10, 11]  # Feet joint indices


# Dataset-specific paths (can be overridden in individual processors)
DATASET_PATHS = {
    'amass': '/is/cluster/fast/eduran2/omomo_fullbody/data/AMASS/',
    'grab': 'data/motion/Hand_Raw/GRAB/grab',
    'arctic': 'data/motion/Hand_Raw/ARCTIC',
    'beat2': 'data/motion/Body_Raw/BEAT2',
    'omomo': 'data/motion/Body_Raw/OMOMO',
    'embody3d': 'external/embody-3d/data',
    'interx': 'external/interx',
    'mamma': 'external/Mamma/data',
    'motionx': 'external/Motion-X/smplx322',
    'samp': 'data/motion/Body_Raw/SAMP',
    'moyo': 'data/motion/Hand_Raw/MOYO',
    'hot3d': 'data/motion/Hand_Raw/HOT3D',
    'interhands': 'data/motion/Hand_Raw/InterHands',
    'reinterhands': 'data/motion/Hand_Raw/ReInterHands',
}
