"""
Mamma Dataset Processor.

Processes Bedlam-Lab "Mamma" evaluation captures (eval_extra / eval_singles / eval_dance).
Each sequence stores ground-truth SMPL-X pose, translation and shape per person in
`<seq>/gt/global.npz`. Multi-person clips (dance) are split into one entry per person.
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


MAMMA_SUBSETS = ['mamma_eval_extra', 'mamma_eval_singles', 'mamma_eval_dance']


class MammaProcessor(BodyDatasetProcessor):
    """Processor for the Mamma evaluation body motion dataset."""

    def __init__(self):
        dataset_path = DATASET_PATHS.get('mamma', 'external/Mamma/data')
        output_path = 'data/motion/Body_Processed/mamma.p'

        self.LOWEST_PERCENT = 0.13
        self.HIGHEST_PERCENT = 1.00

        super().__init__(
            dataset_name='MAMMA',
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

    def cleanup_data(self):
        print('No cleanup needed for MAMMA')

    def load_sequences(self) -> List[str]:
        """Collect every gt/global.npz across the three Mamma eval subsets."""
        sequences = []
        for subset in MAMMA_SUBSETS:
            subset_dir = os.path.join(self.dataset_path, subset)
            if not os.path.isdir(subset_dir):
                print(f'Mamma subset not found, skipping: {subset_dir}')
                continue
            for seq_dir in sorted(glob.glob(os.path.join(subset_dir, '*'))):
                gt_path = os.path.join(seq_dir, 'gt', 'global.npz')
                if os.path.isfile(gt_path):
                    sequences.append(gt_path)
        return sequences

    def filter_sequences(self, sequences: List[str]) -> Dict[str, List[str]]:
        """All Mamma sequences land in the train split (per user spec)."""
        return {'train': sequences}

    def process_sequence(self, sequence_path: str) -> Optional[List[Dict[str, Any]]]:
        """Process a single global.npz, emitting one entry per person."""
        data = np.load(sequence_path, allow_pickle=True)

        fps = int(data['fps'].item())
        sample_freq = max(1, int(round(fps / self.config.TARGET_FPS)))

        pose_world = data['pose_world']           # (T, P, 165)
        trans_world = data['pose_trans_world']    # (T, P, 3)
        shape = data['shape']                     # (P, 300)
        gender_arr = data['gender']               # (P,) of strings
        people_len = int(data['people_len'].item())
        seq_name_base = str(data['seq_name'].item())

        # Subset name derived from path: external/Mamma/data/<subset>/<seq>/gt/global.npz
        subset_name = sequence_path.split(os.sep)[-4]

        results = []
        for person_idx in range(people_len):
            fullpose = pose_world[:, person_idx, :].astype(np.float32)
            trans = trans_world[:, person_idx, :].astype(np.float32)
            betas = shape[person_idx].astype(np.float32)

            # Pad/truncate to NUM_BETAS
            if betas.shape[0] < self.config.NUM_BETAS:
                betas = np.concatenate(
                    [betas, np.zeros(self.config.NUM_BETAS - betas.shape[0], dtype=np.float32)]
                )
            else:
                betas = betas[:self.config.NUM_BETAS]

            gender = str(gender_arr[person_idx]) if person_idx < len(gender_arr) else 'neutral'
            if gender not in self.body_models:
                gender = 'neutral'

            # Downsample to target fps
            fullpose = fullpose[::sample_freq]
            trans = trans[::sample_freq]
            T = len(fullpose)

            if T < self.config.WINDOW:
                continue

            seq_name = f"{subset_name}/{seq_name_base}_p{person_idx}"

            for augment_flag in [False, True]:
                variant = self._process_single_variant(
                    fullpose, trans, betas, gender,
                    subset_name, seq_name, augment_flag, T,
                )
                if variant:
                    results.extend(variant)

        return results if results else None

    def _process_single_variant(
        self,
        fullpose: np.ndarray,
        trans: np.ndarray,
        betas: np.ndarray,
        gender: str,
        subset_name: str,
        seq_name: str,
        augment_flag: bool,
        T: int,
    ) -> Optional[List[Dict[str, Any]]]:
        # Mirror augmentation (YZ-plane reflection)
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

        # SMPL-X 165-dim pose layout: root(3) + body(63) + jaw(3) + leye(3) + reye(3) + lhand(45) + rhand(45)
        root_orient_ = fullpose_[..., :3]
        pose_body_ = fullpose_[..., 3:66]
        pose_jaw_ = fullpose_[..., 66:69]
        pose_eye_ = fullpose_[..., 69:75]
        pose_lhand_ = fullpose_[..., 75:120]
        pose_rhand_ = fullpose_[..., 120:165]

        # Rest pose offsets
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

        chunks = SequenceChunker.chunk_sequence(T, self.config.WINDOW, overlap=0)

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
                'expression': torch.zeros((chunk_len, 10)).float().to(self.device),
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
    processor = MammaProcessor()
    processor.run()
    print('Done processing MAMMA')
