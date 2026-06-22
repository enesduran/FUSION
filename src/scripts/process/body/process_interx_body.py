"""
Inter-X Dataset Processor - body motion processing.

Inter-X is a two-person interaction dataset captured at 120 fps. Each interaction
directory holds two per-person SMPL-X motion files (P1.npz, P2.npz). The fusion
body pipeline operates on single bodies, so every person is processed as an
independent body sequence (seq_name = "<interaction_id>_P1" / "_P2").

Raw motion files (motions/<id>/P{1,2}.npz) store:
    pose_body   (T, 21, 3)   body joint rotations (axis-angle)
    pose_lhand  (T, 15, 3)   left-hand joint rotations
    pose_rhand  (T, 15, 3)   right-hand joint rotations
    root_orient (T, 3)       global orientation
    trans       (T, 3)       global translation
    betas       (1, 10)      shape parameters
    gender      ()           'male' / 'female'

Inter-X bodies are stored Y-up; the fusion pipeline is Z-up (contact detection
uses z_ax_number=2). We therefore rotate the global pose/translation from Y-up to
Z-up with the same convention used by the BEAT2 processor.
"""
import os
import sys
import torch
import numpy as np
from tqdm import tqdm 
from typing import Dict, List, Optional, Any, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

from src.scripts.process.base_processor import BodyDatasetProcessor
from src.scripts.process.config import ProcessingConfig, SMPLX_JOINT_MIRROR_ARR, DATASET_PATHS
from src.scripts.process.processor_utils import ModelLoader
from src.utils.process_utils import BRANCH_NAME, CONTACT_INDICES, determine_floor_height_and_contacts
from src.utils.transforms import axis_angle_to_matrix, matrix_to_axis_angle


# Inter-X is captured at 120 fps (see external/Inter-X/README.md).
INTERX_FPS = 120

# Rotation that maps the Y-up Inter-X frame to the Z-up frame the fusion pipeline
# expects. Same convention as the BEAT2 processor (R^T applied to global_orient,
# row-vector multiply for translation).
R_Y2Z = torch.tensor([[1., 0., 0.],
                      [0., 0., 1.],
                      [0., -1., 0.]]).reshape(3, 3)


def rotate_global(global_rot_aa, global_trans, root_offset):
    """Rotate global orientation and translation from Y-up to Z-up.

    SMPL-X rotates the body about the pelvis rest location (root_offset) and then
    adds transl, so a world-frame rotation R has to be folded into both the global
    orientation and the translation that places the pelvis in the world.
    """
    origin2root = torch.from_numpy(global_trans + root_offset)
    global_rot_aa = matrix_to_axis_angle(R_Y2Z.T @ axis_angle_to_matrix(torch.from_numpy(global_rot_aa).float()))
    global_transl = torch.matmul(origin2root, R_Y2Z) - root_offset

    return global_rot_aa.numpy(), global_transl.numpy()


class InterXProcessor(BodyDatasetProcessor):
    """Processor for the Inter-X two-person interaction dataset (body only)."""

    def __init__(self):
        dataset_path = DATASET_PATHS.get('interx', 'external/interx')
        output_path = 'data/motion/Body_Processed/interx.p'

        super().__init__(
            dataset_name='INTERX',
            dataset_path=dataset_path,
            output_path=output_path
        )

        # Trim the least-dynamic 10% of windows (consistent with AMASS).
        self.LOWEST_PERCENT = 0.20
        self.HIGHEST_PERCENT = 0.99

        # Downsample factor from native 120 fps to TARGET_FPS.
        self.sample_freq = max(1, int(round(INTERX_FPS / self.config.TARGET_FPS)))

        # Initialize models
        self.device = self.get_device()
        self.body_models = ModelLoader.load_smplx_models(
            batch_size=self.config.WINDOW,
            num_betas=self.config.NUM_BETAS,
            device=self.device,
            config=self.config
        )
        self.renderer = ModelLoader.create_renderer(self.config)

    def cleanup_data(self):
        """Inter-X requires no destructive cleanup (dataset is read-only)."""
        print('No cleanup needed for Inter-X')

    def load_sequences(self) -> List[str]:
        """Return every per-person motion file path (P1.npz and P2.npz)."""
        motions_dir = os.path.join(self.dataset_path, 'motions')
        all_sequences = []
 
        
        for interaction_id in tqdm(sorted(os.listdir(motions_dir))[:], desc="Loading sequences"):
            interaction_dir = os.path.join(motions_dir, interaction_id)
            if not os.path.isdir(interaction_dir):
                continue
            for person_file in ('P1.npz', 'P2.npz'):
                path = os.path.join(interaction_dir, person_file)
                if os.path.isfile(path):
                    all_sequences.append(path)
        return all_sequences

    def _load_split_ids(self, split_name: str) -> set:
        """Read the interaction ids belonging to a split."""
        split_file = os.path.join(self.dataset_path, 'splits', f'{split_name}.txt')
        if not os.path.isfile(split_file):
            return set()
        with open(split_file, 'r') as f:
            return {line.strip() for line in f if line.strip()}

    def filter_sequences(self, sequences: List[str]) -> Dict[str, List[str]]:
        """Bucket per-person files into train/val only.

        The dataset's test ids are folded into train, so the processor emits only
        'train' and 'val'. Any id not listed in val falls back to train.
        """
        val_ids = self._load_split_ids('val')

        filtered: Dict[str, List[str]] = {'train': [], 'val': []}
        for path in sequences:
            interaction_id = os.path.basename(os.path.dirname(path))
            split_name = 'val' if interaction_id in val_ids else 'train'
            filtered[split_name].append(path)

        return {name: paths for name, paths in filtered.items() if paths}

    def process_sequence(self, sequence_path: str) -> Optional[List[Dict[str, Any]]]:
        """Process a single person's motion file."""
        data = np.load(sequence_path, allow_pickle=True)

        interaction_id = os.path.basename(os.path.dirname(sequence_path))
        person = os.path.splitext(os.path.basename(sequence_path))[0]  # 'P1' / 'P2'
        seq_name = f'{interaction_id}_{person}'

        gender = str(data['gender'])
        if gender not in self.body_models:
            gender = 'neutral'

        # Pad shape to the model's beta count (SMPL-X does not auto-pad).
        betas = data['betas'].reshape(-1).astype(np.float32)
        if len(betas) < self.config.NUM_BETAS:
            betas = np.concatenate([betas, np.zeros(self.config.NUM_BETAS - len(betas), dtype=np.float32)])
        else:
            betas = betas[:self.config.NUM_BETAS]

        # Downsample 120 -> TARGET_FPS and flatten per-joint axis-angle.
        sf = self.sample_freq
        root_orient = data['root_orient'][::sf].astype(np.float32)            # (T, 3)
        trans = data['trans'][::sf].astype(np.float32)                        # (T, 3)
        pose_body = data['pose_body'][::sf].reshape(-1, 63).astype(np.float32)
        pose_lhand = data['pose_lhand'][::sf].reshape(-1, 45).astype(np.float32)
        pose_rhand = data['pose_rhand'][::sf].reshape(-1, 45).astype(np.float32)
        T = len(root_orient)
        pose_jaw = np.zeros((T, 3), dtype=np.float32)
        pose_eye = np.zeros((T, 6), dtype=np.float32)

        # Skip sequences shorter than one window.
        if T < self.config.WINDOW:
            return None

        # Compute rest-pose joint offsets for this subject's shape.
        bm_rest_out = self.body_models[gender](
            betas=torch.from_numpy(betas[None, :]).float().to(self.device))
        rest_joint_pos = bm_rest_out.joints.cpu().detach().numpy()[0, :len(SMPLX_JOINT_MIRROR_ARR)]
        root_offset = rest_joint_pos[0]

        pos_offset = [[0, 0, 0]]
        for child, parent in enumerate(self.body_models[gender].parents):
            if parent == -1:
                continue
            pos_offset.append(rest_joint_pos[child] - rest_joint_pos[parent])
        pos_offset = np.vstack(pos_offset)

        # Rotate the global trajectory from Y-up to Z-up.
        root_orient, trans = rotate_global(root_orient.copy(), trans.copy(), root_offset)

        # Process both original and left-right mirrored variants.
        results = []
        for augment_flag in [False, True]:
            result = self._process_single_variant(
                root_orient, pose_body, pose_jaw, pose_eye,
                pose_lhand, pose_rhand, trans, betas, gender,
                seq_name, augment_flag, pos_offset, root_offset, T
            )
            if result:
                results.extend(result)

        return results if results else None

    def _chunk_indices(self, T: int, min_tail: int = 110) -> List[tuple]:
        """Window indices: non-overlapping full windows + a tail only if useful.

        Unlike the shared SequenceChunker (which always appends a (T-W, T) tail
        whenever T isn't a multiple of W), the tail window is emitted only when
        the uncovered remainder is at least 100 frames. This caps the overlap
        between consecutive windows at <=20 frames (~17% of the window) and avoids
        near-duplicate windows for Inter-X's short clips. Sequences shorter than
        one window are filtered out earlier in process_sequence.
        """
        W = self.config.WINDOW

        chunks = [(s, s + W) for s in range(0, T - W + 1, W)]
        if chunks and (T - chunks[-1][1]) >= min_tail:
            chunks.append((T - W, T))
        return chunks

    # Thresholds calibrated to dataset p99 (see arm-pose analysis on a 446-window sample).
    # The named bad example G001T001A005R003_P2 trips all three signals simultaneously.
    ARM_SHOULDER_MAX = 1.30   # rad, max axis-angle norm of L/R shoulder (~75 deg)
    ARM_WRIST_MAX = 1.30      # rad, max axis-angle norm of L/R wrist (~75 deg)
    ARM_JERK_MAX = 0.15       # m/frame, max wrist position delta (~4.5 m/s @30fps)

    def _arm_pose_plausible(
        self, joints: np.ndarray, body_pose: np.ndarray
    ) -> Tuple[bool, str]:
        """Detect implausible arm poses (over-twisted shoulder/wrist, wrist jerk).

        Inter-X SMPL-X fits occasionally have severe shoulder/wrist over-rotation
        that makes the elbow position look broken even though elbow flexion itself
        stays in range. This catches that phenotype with three per-window signals.

        Args:
            joints: (T, J>=22, 3) world joint positions from the SMPL-X forward pass.
            body_pose: (T, 63) axis-angle body pose (21 joints starting at left_hip).

        Returns:
            (is_plausible, reason). reason is empty when plausible, otherwise a
            short string naming which signals fired (used in render filenames).
        """
        def jnorm(k: int) -> float:
            # SMPL-X joint k -> body_pose slice [(k-1)*3 : k*3]; max norm over frames.
            return float(np.linalg.norm(body_pose[:, (k - 1) * 3:k * 3], axis=-1).max())

        l_sh, r_sh = jnorm(16), jnorm(17)
        l_wr, r_wr = jnorm(20), jnorm(21)
        j_lw = float(np.linalg.norm(np.diff(joints[:, 20], axis=0), axis=-1).max())
        j_rw = float(np.linalg.norm(np.diff(joints[:, 21], axis=0), axis=-1).max())

        fails = []
        if l_sh > self.ARM_SHOULDER_MAX or r_sh > self.ARM_SHOULDER_MAX:
            fails.append(f'sh{l_sh:.2f}/{r_sh:.2f}')
        if l_wr > self.ARM_WRIST_MAX or r_wr > self.ARM_WRIST_MAX:
            fails.append(f'wr{l_wr:.2f}/{r_wr:.2f}')
        if j_lw > self.ARM_JERK_MAX or j_rw > self.ARM_JERK_MAX:
            fails.append(f'jk{j_lw:.2f}/{j_rw:.2f}')

        return (len(fails) == 0), ','.join(fails)

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
        seq_name: str,
        augment_flag: bool,
        pos_offset: np.ndarray,
        root_offset: np.ndarray,
        T: int
    ) -> Optional[List[Dict[str, Any]]]:
        """Process the original or left-right mirrored version of a sequence."""

        if augment_flag:
            trans_ = trans.copy()
            trans_[:, 0] *= -1

            fullpose = np.hstack([root_orient, pose_body, pose_jaw, pose_eye, pose_lhand, pose_rhand])
            fullpose_ = fullpose.copy().reshape(-1, len(SMPLX_JOINT_MIRROR_ARR), 3)
            fullpose_ = fullpose_[:, SMPLX_JOINT_MIRROR_ARR]
            fullpose_[..., 1:] *= -1
            fullpose_ = fullpose_.reshape(-1, len(SMPLX_JOINT_MIRROR_ARR) * 3)

            root_orient_ = fullpose_[..., :3]
            pose_body_ = fullpose_[..., 3:66]
            pose_jaw_ = fullpose_[..., 66:69]
            pose_eye_ = fullpose_[..., 69:75]
            pose_lhand_ = fullpose_[..., 75:120]
            pose_rhand_ = fullpose_[..., 120:]
        else:
            trans_ = trans.copy()
            root_orient_ = root_orient.copy()
            pose_body_ = pose_body.copy()
            pose_jaw_ = pose_jaw.copy()
            pose_eye_ = pose_eye.copy()
            pose_lhand_ = pose_lhand.copy()
            pose_rhand_ = pose_rhand.copy()

        # Chunk into non-overlapping windows, plus a tail window only when it
        # adds enough new frames (see _chunk_indices). Inter-X clips are short
        # (median ~153 frames vs a 120-frame window), so the generic chunker's
        # unconditional tail window would otherwise produce near-duplicate
        # consecutive windows (~70% overlap) for most sequences.
        chunks = self._chunk_indices(T)

        results = []
        for start, end in chunks:
            chunk_slice = slice(start, end)
            chunk_len = end - start

            if chunk_len < self.config.WINDOW:
                continue

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

            bm_out = self.body_models[gender](**motion_params_gt)
            joints = bm_out.joints.cpu().detach().numpy()[:, :len(SMPLX_JOINT_MIRROR_ARR)]
            vertices = bm_out.vertices.cpu().detach().numpy()

            # Compute foot contacts and the floor/terrain discard heuristic.
            offset_floor_height, contacts, discard_seq = \
                determine_floor_height_and_contacts(joints, self.config.TARGET_FPS)

            # Arm-pose plausibility: catches over-twisted shoulder/wrist + jerky
            # wrist motion that the contact heuristic misses (a separate failure
            # mode of Inter-X SMPL-X fits).
            arm_ok, arm_reason = self._arm_pose_plausible(
                joints, pose_body_[chunk_slice]
            )

            discard_reason = None
            if discard_seq:
                discard_reason = 'contact'
            elif not arm_ok:
                discard_reason = f'arm-{arm_reason}'

            # Drop flagged windows, but render each one first so the discarded
            # poses can be inspected and the thresholds calibrated.
            if discard_reason is not None:
                tag = f"{'mir' if augment_flag else 'orig'}_{start:04d}"
                # self._visualize_sequence(
                #     motion_params_gt, vertices,
                #     f'{seq_name}_{tag}_{discard_reason}',
                #     joints=joints, contacts=contacts, status='discard',
                # )
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
        joints: Optional[np.ndarray] = None,
        contacts: Optional[np.ndarray] = None,
        status: str = 'discard'
    ):
        """Render a window to video for debugging."""
        filename = f"fusion_runs/{BRANCH_NAME}/contact_vis_{status}/{self.dataset_name}/{seq_name}_{self.motion_idx}"
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        mesh_dict = {
            'transl': motion_params_gt['transl'].cpu(),
            'global_orient': motion_params_gt['global_orient'].cpu(),
            'faces': self.body_models['neutral'].faces,
            'vertices': vertices
        }
        camera_dict = {
            'camera_rot': motion_params_gt['global_orient'].cpu(),
            'camera_transl': motion_params_gt['transl'].cpu(),
            'coef': 2.0
        }

        if joints is not None and contacts is not None:
            skeleton_dict = {
                'positions': joints,
                'contact_masks': contacts[:, CONTACT_INDICES],
                'color': (16/255, 60/255, 160/255, 0.9)
            }
            self.renderer.render_motion(
                mesh_dict, filename, skeleton_dict=skeleton_dict,
                camera_dict=camera_dict, color=(255/255, 160/255, 0/255, 1)
            )
        else:
            self.renderer.render_motion(
                mesh_dict, filename, camera_dict=camera_dict,
                color=(255/255, 160/255, 0/255, 1)
            )

    def add_to_data_dict(self, processed_data: Any):
        """Handle the list of windows returned per sequence."""
        if isinstance(processed_data, list):
            for chunk_data in processed_data:
                super().add_to_data_dict(chunk_data)
        else:
            super().add_to_data_dict(processed_data)


if __name__ == '__main__':
    processor = InterXProcessor()
    processor.run()
    print('Done processing Inter-X')
