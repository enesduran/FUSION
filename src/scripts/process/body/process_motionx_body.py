"""
Motion-X Dataset Processor.

Processes the Motion-X smplx_322 release (animation, haa500, humman, idea400,
kungfu, music, perform subsets). Each clip is stored as a single
`(T, 322)` float32 numpy array packing the SMPL-X parameters in the canonical
Motion-X layout:

    [  0:  3]  root_orient   (axis-angle)
    [  3: 66]  pose_body     (21 joints x 3, axis-angle)
    [ 66:111]  pose_lhand    (15 joints x 3, axis-angle)
    [111:156]  pose_rhand    (15 joints x 3, axis-angle)
    [156:159]  jaw_pose      (axis-angle)
    [159:209]  expression    (50 SMPL-X expression coeffs)
    [209:309]  face_shape    (100 -- stored as zeros)
    [309:312]  trans         (global translation, meters)
    [312:322]  betas         (10 -- stored as zeros)

Notes
-----
* FPS is not embedded in the files. Motion-X is documented at 30 fps and the
  observed clip lengths are consistent with that, so we treat the data as
  already at TARGET_FPS (sample_freq = 1).
* `face_shape` and `betas` are zero in every clip -- we use the SMPL-X
  neutral mean shape and pad betas to NUM_BETAS with zeros.
* No eye-pose channel exists in the 322 layout -- we substitute zeros.
* Gender is not annotated -- everything is processed with the neutral model.
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
from src.utils.transforms import axis_angle_to_matrix, matrix_to_axis_angle


# Rotation that takes Motion-X's world frame into the codebase's expected
# SMPL-X frame (identical to the matrix used by the BEAT2 processor).
R_V2S = torch.tensor([[1., 0., 0.],
                      [0., 0., 1.],
                      [0., -1., 0.]]).reshape(3, 3)


def rotate_global(global_rot_aa, global_trans, root_offset):
    """Apply R_V2S to root orientation and translation. See BEAT2 processor."""
    origin2root = torch.from_numpy(global_trans + root_offset)
    global_rot_aa = matrix_to_axis_angle(
        R_V2S.T @ axis_angle_to_matrix(torch.from_numpy(global_rot_aa).float())
    )
    global_transl = torch.matmul(origin2root, R_V2S) - root_offset
    return global_rot_aa.numpy(), global_transl.numpy()


# Subset name -> directory relative to dataset_path.
# idea400 was packaged with deeper nesting than the others.
MOTIONX_SUBSETS = {
    # 'animation': 'animation',   #bad quality 
    # 'haa500':    'haa500',      #bad quality
    # 'humman':    'humman',      #bad quality
    # 'idea400':   'motion_generation/smplx_322/idea400', #bad quality
    # 'music':     'music',        #bad quality
    # 'perform':   'perform',        #bad quality
    'kungfu':    'kungfu',     # maybe
}

# Canonical Motion-X 322 layout slices
SLICE_ROOT   = slice(0, 3)
SLICE_BODY   = slice(3, 66)
SLICE_LHAND  = slice(66, 111)
SLICE_RHAND  = slice(111, 156)
SLICE_JAW    = slice(156, 159)
SLICE_TRANS  = slice(309, 312)

# Motion-X is documented at 30 fps for all smplx_322 subsets.
MOTIONX_FPS = 30


class MotionXProcessor(BodyDatasetProcessor):
    """Processor for the Motion-X smplx_322 body motion dataset."""

    def __init__(self):
        dataset_path = DATASET_PATHS.get('motionx', 'external/Motion-X/smplx322')
        output_path = 'data/motion/Body_Processed/motionx.p'

        self.LOWEST_PERCENT = 0.10
        self.HIGHEST_PERCENT = 1.00

        super().__init__(
            dataset_name='MOTIONX',
            dataset_path=dataset_path,
            output_path=output_path,
        )

        self.device = self.get_device()
        self.body_models = ModelLoader.load_smplx_models(
            batch_size=self.config.WINDOW,
            num_betas=self.config.NUM_BETAS,
            device=self.device,
            config=self.config,
        )
        self.renderer = ModelLoader.create_renderer(self.config)

        self.sample_freq = max(1, int(round(MOTIONX_FPS / self.config.TARGET_FPS)))

    def cleanup_data(self):
        print('No cleanup needed for Motion-X')

    def load_sequences(self) -> List[str]:
        """Collect every .npy clip across the seven Motion-X subsets."""
        sequences = []
        for subset_name, rel_dir in MOTIONX_SUBSETS.items():
            subset_dir = os.path.join(self.dataset_path, rel_dir)
            if not os.path.isdir(subset_dir):
                print(f'Motion-X subset not found, skipping: {subset_dir}')
                continue
            subset_files = sorted(glob.glob(os.path.join(subset_dir, '*.npy')))
            print(f'  {subset_name:<10}  {len(subset_files):>6} clips')
            sequences.extend(subset_files)
        return sequences

    def filter_sequences(self, sequences: List[str]) -> Dict[str, List[str]]:
        """All Motion-X clips land in the train split."""
        return {'train': sequences}

    def _subset_and_clip_name(self, sequence_path: str) -> (str, str):
        """Recover (subset_name, clip_stem) from a sequence path."""
        clip_stem = os.path.splitext(os.path.basename(sequence_path))[0]
        rel = os.path.relpath(sequence_path, self.dataset_path)
        rel_parts = rel.split(os.sep)
        # Match against MOTIONX_SUBSETS for robust subset identification
        for subset_name, rel_dir in MOTIONX_SUBSETS.items():
            rel_dir_parts = rel_dir.split(os.sep)
            if rel_parts[:len(rel_dir_parts)] == rel_dir_parts:
                return subset_name, clip_stem
        # Fallback: use top-level directory
        return rel_parts[0], clip_stem

    def process_sequence(self, sequence_path: str) -> Optional[List[Dict[str, Any]]]:
        """Process a single Motion-X clip."""
        motion = np.load(sequence_path).astype(np.float32)
        if motion.ndim != 2 or motion.shape[1] != 322:
            print(f'  Skipping malformed clip {sequence_path}: shape={motion.shape}')
            return None

        subset_name, clip_stem = self._subset_and_clip_name(sequence_path)

        # Downsample to TARGET_FPS (no-op when MOTIONX_FPS == TARGET_FPS).
        motion = motion[::self.sample_freq]
        T = motion.shape[0]
        if T < self.config.WINDOW:
            return None

        # Split into the named SMPL-X parts.
        root_orient = motion[:, SLICE_ROOT]
        pose_body   = motion[:, SLICE_BODY]
        pose_lhand  = motion[:, SLICE_LHAND]
        pose_rhand  = motion[:, SLICE_RHAND]
        pose_jaw    = motion[:, SLICE_JAW]
        trans       = motion[:, SLICE_TRANS]

        # No eye-pose channel in the 322 layout -> zeros.
        pose_eye = np.zeros((T, 6), dtype=np.float32)

        # No subject identity: zero betas padded to NUM_BETAS, neutral gender.
        betas = np.zeros(self.config.NUM_BETAS, dtype=np.float32)
        gender = 'neutral'

        # Rest pose offsets for FK and motion-variance filtering. Computed
        # once here because rotate_global() needs root_offset.
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

        # Rotate Motion-X's world frame into the codebase's SMPL-X frame.
        root_orient, trans = rotate_global(root_orient.copy(), trans.copy(), root_offset)

        seq_name = f"{subset_name}/{clip_stem}"

        results = []
        for augment_flag in [False, True]:
            variant = self._process_single_variant(
                root_orient, pose_body, pose_lhand, pose_rhand,
                pose_jaw, pose_eye, trans, betas, gender,
                subset_name, seq_name, augment_flag,
                pos_offset, root_offset, T,
            )
            if variant:
                results.extend(variant)

        return results if results else None

    def _process_single_variant(
        self,
        root_orient: np.ndarray,
        pose_body: np.ndarray,
        pose_lhand: np.ndarray,
        pose_rhand: np.ndarray,
        pose_jaw: np.ndarray,
        pose_eye: np.ndarray,
        trans: np.ndarray,
        betas: np.ndarray,
        gender: str,
        subset_name: str,
        seq_name: str,
        augment_flag: bool,
        pos_offset: np.ndarray,
        root_offset: np.ndarray,
        T: int,
    ) -> Optional[List[Dict[str, Any]]]:
        """Original or YZ-mirrored variant of a Motion-X clip."""
        if augment_flag:
            # Assemble an AMASS-style 165-dim full pose so we can reuse the
            # shared SMPLX_JOINT_MIRROR_ARR (root, body, jaw, leye, reye,
            # lhand, rhand) for the left/right swap.
            fullpose = np.concatenate(
                [root_orient, pose_body, pose_jaw, pose_eye, pose_lhand, pose_rhand],
                axis=-1,
            )
            n_joints = len(SMPLX_JOINT_MIRROR_ARR)  # 55
            fullpose = fullpose.reshape(-1, n_joints, 3)
            fullpose = fullpose[:, SMPLX_JOINT_MIRROR_ARR]
            fullpose[..., 1:] *= -1
            fullpose = fullpose.reshape(-1, n_joints * 3)

            root_orient_ = fullpose[..., :3]
            pose_body_   = fullpose[..., 3:66]
            pose_jaw_    = fullpose[..., 66:69]
            pose_eye_    = fullpose[..., 69:75]
            pose_lhand_  = fullpose[..., 75:120]
            pose_rhand_  = fullpose[..., 120:165]

            trans_ = trans.copy()
            trans_[:, 0] *= -1
        else:
            root_orient_ = root_orient.copy()
            pose_body_   = pose_body.copy()
            pose_jaw_    = pose_jaw.copy()
            pose_eye_    = pose_eye.copy()
            pose_lhand_  = pose_lhand.copy()
            pose_rhand_  = pose_rhand.copy()
            trans_       = trans.copy()

        chunks = SequenceChunker.chunk_sequence(T, self.config.WINDOW, overlap=0)

        results = []
        for start, end in chunks:
            chunk_slice = slice(start, end)
            chunk_len = end - start
            if chunk_len < self.config.WINDOW:
                continue

            motion_params_gt = {
                'betas':           torch.from_numpy(betas[None, :]).float().to(self.device),
                'global_orient':   torch.from_numpy(root_orient_[chunk_slice]).float().to(self.device),
                'body_pose':       torch.from_numpy(pose_body_[chunk_slice]).float().to(self.device),
                'left_hand_pose':  torch.from_numpy(pose_lhand_[chunk_slice]).float().to(self.device),
                'right_hand_pose': torch.from_numpy(pose_rhand_[chunk_slice]).float().to(self.device),
                'jaw_pose':        torch.from_numpy(pose_jaw_[chunk_slice]).float().to(self.device),
                'leye_pose':       torch.from_numpy(pose_eye_[chunk_slice][:, :3]).float().to(self.device),
                'reye_pose':       torch.from_numpy(pose_eye_[chunk_slice][:, 3:]).float().to(self.device),
                'transl':          torch.from_numpy(trans_[chunk_slice]).float().to(self.device),
                'expression':      torch.zeros((chunk_len, 10)).float().to(self.device),
            }

            bm_out = self.body_models[gender](**motion_params_gt)
            joints = bm_out.joints.cpu().detach().numpy()[:, :len(SMPLX_JOINT_MIRROR_ARR)]
            vertices = bm_out.vertices.cpu().detach().numpy()

            offset_floor_height, contacts, discard_seq = \
                determine_floor_height_and_contacts(joints, self.config.TARGET_FPS)

            if discard_seq:
                self._visualize_sequence(
                    motion_params_gt, vertices, seq_name,
                    dataset_name=subset_name,
                    root_orient=root_orient_[chunk_slice],
                    trans=trans_[chunk_slice],
                )
                continue

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
                'contacts_mask': contacts,
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
    ):
        dataset_name = dataset_name or self.dataset_name
        filename = f"fusion_runs/{BRANCH_NAME}/body_dataset_vis/{dataset_name}/{seq_name}"
        if filename.endswith('.npz') or filename.endswith('.pkl'):
            filename = filename[:-4]
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        mesh_dict = {
            'transl': motion_params_gt['transl'].cpu(),
            'global_orient': motion_params_gt['global_orient'].cpu(),
            'faces': self.body_models['neutral'].faces,
            'vertices': vertices,
        }

        camera_dict = {
            'camera_rot': torch.from_numpy(root_orient).float() if root_orient is not None else motion_params_gt['global_orient'].cpu(),
            'camera_transl': torch.from_numpy(trans).float() if trans is not None else motion_params_gt['transl'].cpu(),
            'coef': 1.9,
        }

        self.renderer.render_motion(
            mesh_dict, filename,
            camera_dict=camera_dict,
            color=(255 / 255, 160 / 255, 0 / 255, 1),
        )

    def add_to_data_dict(self, processed_data: Any):
        if isinstance(processed_data, list):
            for chunk_data in processed_data:
                super().add_to_data_dict(chunk_data)
        else:
            super().add_to_data_dict(processed_data)


if __name__ == '__main__':
    processor = MotionXProcessor()
    processor.run()
    print('Done processing Motion-X')
