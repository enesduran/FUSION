import gc
import os
import glob
import json
import torch
import smplx
import shutil
import psutil
import ctypes
import joblib
import trimesh
import nvidia_smi
import numpy as np
from tqdm import tqdm
from typing import List
from copy import deepcopy
from functools import lru_cache
from omegaconf import DictConfig
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import DataLoader, WeightedRandomSampler
from src.data.amass_dataset import AmassDataset
from src.utils.data_utils import concatenate_data
from src.utils.genutils import cast_dict_to_tensors, freeze

nvidia_smi.nvmlInit()

class LazyLoadDict:
    """Dict-like object that lazily loads precomputed .p files from disk.
    Files are loaded on demand and cached with an LRU cache.
    If eager=True, all files are loaded into memory upfront."""

    def __init__(self, folder_path, cache_size=256, eager=False):
        self.folder_path = folder_path
        file_list = sorted(glob.glob(os.path.join(folder_path, '*.p')))
        self._key_to_path = {
            int(os.path.basename(f).split('.')[0]): f for f in file_list
        }
        self._keys = sorted(self._key_to_path.keys())
        self._eager = eager

        if eager:
            print(f'Eagerly loading {len(self._keys)} samples from {folder_path}...')
            self._data = {}
            for k in tqdm(self._keys, desc='Loading into RAM'):
                self._data[k] = joblib.load(self._key_to_path[k])
        else:
            self._load_cached = lru_cache(maxsize=cache_size)(self._load_file)

    @staticmethod
    def _load_file(filepath):
        return joblib.load(filepath)

    def __getitem__(self, key):
        key = int(key)
        if key not in self._key_to_path:
            raise KeyError(key)
        if self._eager:
            return self._data[key]
        return self._load_cached(self._key_to_path[key])

    def __len__(self):
        return len(self._keys)

    def __contains__(self, key):
        return int(key) in self._key_to_path

    def keys(self):
        return self._keys

    def values(self):
        for k in self._keys:
            yield self[k]

    def items(self):
        for k in self._keys:
            yield k, self[k]

    def __iter__(self):
        return iter(self._keys)


def ram_check(label=""):
    """Check RAM usage in Condor PyTorch jobs."""
    ram_mb = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
    gpu_info = f" | GPU: {torch.cuda.memory_allocated() / (1024**2):.0f}MB" if torch.cuda.is_available() else ""
    condor_limit = os.environ.get('_CONDOR_MEMORY', os.environ.get('CONDOR_MEMORY', ''))
    limit_info = f" | Limit: {condor_limit}MB" if condor_limit else ""
    print(f"RAM: {ram_mb:.0f}MB{gpu_info}{limit_info} {label}")
    return ram_mb


def get_gpu():
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    print(f"Device 0: {nvidia_smi.nvmlDeviceGetName(handle)}, "
          f"Memory: ({100 * info.free / info.total:.2f}% free): "
          f"{info.total / 1073741824:.2f} GB (total), "
          f"{info.free / 1073741824:.2f} GB (free), "
          f"{info.used / 1073741824:.2f} GB (used)")


class AmassWrapper:
    def __init__(self,
                 load_feats: List[str],
                 include_pose_augmentation: bool,
                 include_time_augmentation: bool,
                 train_batch_size: int = 32,
                 val_batch_size: int = 32,
                 num_workers: int = 0,
                 body_train_datapath_list: List[str] = [],
                 body_val_datapath_list: List[str] = [],
                 body_test_datapath_list: List[str] = [],
                 hand_datapath_list: List[str] = [],
                 object_dataset_list: List[str] = [],
                 hand_art_dataset_list: List[str] = [],
                 load_splits: List[str] = ['train', 'test', 'val'],
                 preproc: DictConfig = None,
                 smplx_path: str = "",
                 rot_repr: str = "6d",
                 wind: int = 120,
                 device: str = 'cpu',
                 check_penetration_flag: bool = False,
                 **kwargs):

        self.train_dataloader_options = {'batch_size': train_batch_size, 'num_workers': num_workers, 'drop_last': True}
        self.val_dataloader_options = {'batch_size': val_batch_size, 'num_workers': num_workers, 'drop_last': False}

        self.device = torch.device(device)
        assert len(load_splits) > 0, "At least one split must be selected"

        self.WIND = wind
        
        self.body_train_datapath_list = body_train_datapath_list
        self.body_val_datapath_list = body_val_datapath_list
        self.body_test_datapath_list = body_test_datapath_list

        self.hand_datapath_list = hand_datapath_list
        self.object_dataset_list = object_dataset_list
        self.hand_art_dataset_list = hand_art_dataset_list
        
        self.split_id_dict = {'train': 0, 'val': 1, 'test': 2}
        
        self.check_penetration_flag = check_penetration_flag
        self.load_feats = load_feats
        # Per-dataset weighted sampling config (see configs/data.yaml `sampling:`).
        # train_sample_weights stays None unless sampling is enabled -> uniform shuffle.
        self.sampling_cfg = kwargs.get('sampling', None)
        self.train_sample_weights = None
        self.dataset = {}
        self.preproc = preproc
        self.smplx_path = smplx_path
        self.rot_repr = rot_repr
        self.stat_path = self.preproc.stats_file

        self._create_body_models(smplx_path, device)
        
        self.watertight_conversion_dict = joblib.load('data/body_models/watertight/conversion_dict.pkl')

        if self.check_penetration_flag:
            self._init_penetration_data()

        self.body_processed_train_list = [p.replace('.p', '_processed.p') for p in self.body_train_datapath_list]
        self.body_processed_val_list = [p.replace('.p', '_processed.p') for p in self.body_val_datapath_list]
        self.body_processed_test_list = [p.replace('.p', '_processed.p') for p in self.body_test_datapath_list]

        self.all_precomputed_foldernames = {
            splt: os.path.join(os.path.dirname(self.body_processed_train_list[0]),
                f'../precomputed/{splt}') for splt in ['train', 'test', 'val']}

        for folder in self.all_precomputed_foldernames.values():
            os.makedirs(folder, exist_ok=True)
 
        self.hand_seq_idx = 0
        self.unprocessed_hand_flag = False
        unprocessed_train_flag, unprocessed_val_flag, unprocessed_test_flag = False, False, False
        hand_dataset_dict, body_train_dataset_dict, body_val_dataset_dict, body_test_dataset_dict = {}, {}, {}, {}
        
        self.demo_mode = kwargs.get('demo_mode', False)
        self.ram_threshold_gb_by_split = {s: kwargs.get(f'{s.upper()}_RAM_THRESHOLD_GB') for s in ['train', 'val', 'test']}
        precomp_exists_flag_by_split = {s: True for s in ['train', 'val', 'test']}

        dataset_args = {'n_body_joints': self.NUM_JOINTS,
                        'stats_file': self.preproc.stats_file,
                        'norm_type': self.preproc.norm_type,
                        'rot_repr': self.rot_repr,
                        'device': self.device,
                        'object_dataset_list': object_dataset_list,
                        'load_feats': self.load_feats,
                        'body_model_dict': self.bm_dict}

        # Demo mode: process only the provided body motion file and reuse stats.
        if self.demo_mode:
            self.motion_filepath = kwargs['motion_filepath']
            demo_processed_path = self.motion_filepath.replace('.p', '_processed.p')
            
            # it is the path to the single .p file 
            demo_precomputed_path = demo_processed_path.replace('sample_data', 'sample_data_precomputed').replace('_processed', '')
            
            os.makedirs(os.path.dirname(demo_precomputed_path), exist_ok=True)
            
            body_demo_dataset_dict, _ = self.gather_body_data(
                [self.motion_filepath], [demo_processed_path], include_pose_augment=True)
            
            self.stats = np.load(self.stat_path, allow_pickle=True)[()]
            save_split = load_splits[0]
            
            self.merge_precompute_and_save_streaming(
                    body_demo_dataset_dict, hand_dataset_dict, False,
                    demo_precomputed_path, object_dataset_list, save_stats=False)
            
            demo_dataset = AmassDataset({0: joblib.load(demo_precomputed_path)}, **dataset_args)
            self.dataset[save_split] = demo_dataset

            print(f"Set up {save_split} set with {len(self.dataset[save_split])} items.")
            self.nfeats = self.dataset[save_split].nfeats
            
            return 
            
        else:
            if 'train' in load_splits:
                body_train_dataset_dict, unprocessed_train_flag = self.gather_body_data(
                    self.body_train_datapath_list, self.body_processed_train_list, include_pose_augmentation)
                hand_dataset_dict = self.gather_hand_data()

            if 'val' in load_splits:
                body_val_dataset_dict, unprocessed_val_flag = self.gather_body_data(
                    self.body_val_datapath_list, self.body_processed_val_list, include_pose_augmentation)

            if 'test' in load_splits:
                body_test_dataset_dict, unprocessed_test_flag = self.gather_body_data(
                    self.body_test_datapath_list, self.body_processed_test_list, False)
        
            processed_flag = not (unprocessed_train_flag or unprocessed_val_flag
                                  or unprocessed_test_flag or self.unprocessed_hand_flag)

            for split in load_splits:
                precomp_files = glob.glob(
                    os.path.join(self.all_precomputed_foldernames[split], '*.p'))
                precomp_exists_flag_by_split[split] = (
                    len(precomp_files) == kwargs[f'{split.upper()}_THRESHOLD'])
                
        precomp_exists_flag = all(precomp_exists_flag_by_split[s] for s in load_splits)

        if (not processed_flag) and self.check_penetration_flag:
            for splt in load_splits:
                collision_dir = self.collision_output_foldername.replace('SPLIT', splt)
                if os.path.exists(collision_dir):
                    shutil.rmtree(collision_dir)
                os.makedirs(collision_dir, exist_ok=True)

        
        # All data is processed and precomputed — just load
        # Demo mode bypasses this path (single sequence is provided directly).
        if processed_flag and os.path.exists(self.stat_path) and precomp_exists_flag:
            print('\nLoading precomputed data and stats.')
       
            # Just delete the processed data to free up RAM if we already have the precomputed data.
            del body_train_dataset_dict, body_val_dataset_dict, body_test_dataset_dict
            gc.collect()
            ctypes.CDLL("libc.so.6").malloc_trim(0)
                 
            self.stats = np.load(self.stat_path, allow_pickle=True)[()]
            
            if not self.demo_mode:
                for splt in tqdm(load_splits, desc='Loading precomputed splits'):
                    self.dataset[splt] = AmassDataset(
                        self.load_dataset(
                            self.all_precomputed_foldernames[splt],
                            ram_threshold_gb=self.ram_threshold_gb_by_split[splt]), **dataset_args)

        else:

            if hand_dataset_dict is None: 
                hand_dataset_dict = self.gather_hand_data()

            # Train split needs (re)processing
            need_train_for_stats = (('train' not in load_splits) and (not os.path.exists(self.stat_path))
                and (not self.demo_mode))

            if (unprocessed_train_flag or need_train_for_stats
                    or ('train' in load_splits and (not os.path.exists(self.stat_path)
                         or not precomp_exists_flag_by_split['train']))):

                # load to compute stats
                if 'train' not in load_splits:
                    body_train_dataset_dict, unprocessed_train_flag = self.gather_body_data(
                        self.body_train_datapath_list, self.body_processed_train_list, include_pose_augmentation)

                self.merge_precompute_and_save_streaming(
                    body_train_dataset_dict, hand_dataset_dict, include_time_augmentation,
                    self.all_precomputed_foldernames['train'], object_dataset_list, save_stats=True)

                del body_train_dataset_dict

                self.dataset['train'] = AmassDataset(
                    self.load_dataset(
                        self.all_precomputed_foldernames['train'],
                        ram_threshold_gb=self.ram_threshold_gb_by_split['train']),
                    **dataset_args)

            elif 'train' in load_splits:
                self.dataset['train'] = AmassDataset(
                    self.load_dataset(
                        self.all_precomputed_foldernames['train'],
                        ram_threshold_gb=self.ram_threshold_gb_by_split['train']),
                    **dataset_args)

            # Val split needs (re)processing
            if unprocessed_val_flag or not precomp_exists_flag_by_split['val']:
                body_val_dataset_dict, _ = self.gather_body_data(
                    self.body_val_datapath_list, self.body_processed_val_list, False)

                self.merge_precompute_and_save_streaming(
                    body_val_dataset_dict, hand_dataset_dict, False,
                    self.all_precomputed_foldernames['val'], object_dataset_list, save_stats=False)
                del body_val_dataset_dict

                self.dataset['val'] = AmassDataset(
                    self.load_dataset(
                        self.all_precomputed_foldernames['val'],
                        ram_threshold_gb=self.ram_threshold_gb_by_split['val']),
                    **dataset_args)

            elif 'val' in load_splits:
                self.dataset['val'] = AmassDataset(
                    self.load_dataset(
                        self.all_precomputed_foldernames['val'],
                        ram_threshold_gb=self.ram_threshold_gb_by_split['val']),
                    **dataset_args)

            if unprocessed_test_flag or not precomp_exists_flag_by_split['test']:
                body_test_dataset_dict, _ = self.gather_body_data(
                    self.body_test_datapath_list, self.body_processed_test_list, False)

                self.merge_precompute_and_save_streaming(
                    body_test_dataset_dict, hand_dataset_dict, False,
                    self.all_precomputed_foldernames['test'], object_dataset_list, save_stats=False)
                del body_test_dataset_dict

                self.dataset['test'] = AmassDataset(
                    self.load_dataset(
                        self.all_precomputed_foldernames['test'],
                        ram_threshold_gb=self.ram_threshold_gb_by_split['test']),
                    **dataset_args)

            elif 'test' in load_splits:
                self.dataset['test'] = AmassDataset(
                    self.load_dataset(
                        self.all_precomputed_foldernames['test'],
                        ram_threshold_gb=self.ram_threshold_gb_by_split['test']),
                    **dataset_args)

        for splt in load_splits:
            print(f"Set up {splt} set with {len(self.dataset[splt])} items.")

        self.nfeats = self.dataset[load_splits[0]].nfeats

        # Build per-sample weights for weighted training sampling (if enabled).
        if (not self.demo_mode) and ('train' in load_splits) and self._sampling_enabled():
            self._build_train_sample_weights()


    def _create_body_models(self, smplx_path, device):
        """Create SMPL-X body models for all genders and freeze them."""
        self.bm_dict = {}
        for gender in ['male', 'female', 'neutral']:
            self.bm_dict[gender] = smplx.create(
                f'{smplx_path}/smplx/SMPLX_{gender.upper()}.npz',
                model_type='smplx', num_betas=300, gender=gender,
                flat_hand_mean=True, use_pca=False,
            ).to(device)

        self.default_vtemp_dict = {g: m.v_template for g, m in self.bm_dict.items()}

        for model in self.bm_dict.values():
            freeze(model)

        self.NUM_JOINTS = self.bm_dict['neutral'].NUM_JOINTS + 1

    def _init_penetration_data(self):
        from mesh_intersection.bvh_search_tree import BVH
        
        """Set up watertight mesh and collision-checking data structures."""
        wt = self.watertight_conversion_dict

        self.base_rhand_face_ids = torch.tensor(wt['base_rhand_face_ids'])
        self.base_lhand_face_ids = torch.tensor(wt['base_lhand_face_ids'])
        self.watertight_rhand_faces = wt['watertight_rhand_faces']
        self.watertight_lhand_faces = wt['watertight_lhand_faces']
        self.watertight_lhand_vertex_ids = wt['watertight_w_hand_lhand_vertex_ids']
        self.watertight_rhand_vertex_ids = wt['watertight_w_hand_rhand_vertex_ids']

        self.watertight_w_hand_lhand_face_ids = np.array(wt['watertight_w_hand_lhand_face_ids'])
        self.watertight_w_hand_rhand_face_ids = np.array(wt['watertight_w_hand_rhand_face_ids'])
        self.base2watertight_w_hand = list(wt['watertight2base_w_hand'].values())
        self.base2watertight_faces_dict = wt['base2watertight_faces_dict']
        self.watertight_w_hand_faces = np.array(list(wt['watertight_w_hand_faces']))

        all_wt_ids = np.arange(self.watertight_w_hand_faces.shape[0])
        self.watertight_w_hand_body_face_ids = np.setdiff1d(
            np.setdiff1d(all_wt_ids, self.watertight_w_hand_lhand_face_ids),
            self.watertight_w_hand_rhand_face_ids)

        all_base_ids = np.arange(self.bm_dict['neutral'].faces.shape[0])
        self.base_hand_body_face_ids = np.setdiff1d(
            np.setdiff1d(all_base_ids, self.base_lhand_face_ids),
            self.base_rhand_face_ids)

        self.base_lhand_face_ids = self.base_lhand_face_ids.clone().to(self.device)
        self.base_rhand_face_ids = self.base_rhand_face_ids.clone().to(self.device)
        self.base_hand_body_face_ids = torch.tensor(self.base_hand_body_face_ids).to(self.device)

        self.collision_output_foldername = 'data/collision_output/SPLIT/'


    @staticmethod
    def _pad_betas(betas, target=300):
        """Pad betas to target size with zeros if needed."""
        if isinstance(betas, np.ndarray):
            current = betas.shape[-1]
            if current < target:
                pad_shape = list(betas.shape)
                pad_shape[-1] = target - current
                betas = np.concatenate([betas, np.zeros(pad_shape, dtype=betas.dtype)], axis=-1)
        else:
            current = betas.shape[-1]
            if current < target:
                pad_shape = list(betas.shape)
                pad_shape[-1] = target - current
                betas = torch.cat([betas, torch.zeros(pad_shape, dtype=betas.dtype, device=betas.device)], dim=-1)
        return betas

    def _build_pose_shape_dict(self, betas, trans, fullbody_aa):
        """Construct the SMPL-X forward-pass input dict from axis-angle params."""
        dev = self.device
        return {
            'betas': betas.to(dev),
            'expression': torch.zeros((self.WIND, 10), device=dev),
            'transl': trans.to(dev),
            'global_orient': fullbody_aa[:, :3].to(dev),
            'body_pose': fullbody_aa[:, 3:66].to(dev),
            'jaw_pose': fullbody_aa[:, 66:69].to(dev),
            'leye_pose': fullbody_aa[:, 69:72].to(dev),
            'reye_pose': fullbody_aa[:, 72:75].to(dev),
            'left_hand_pose': fullbody_aa[:, 75:120].to(dev),
            'right_hand_pose': fullbody_aa[:, 120:].to(dev),
        }

    def _set_vertex_template(self, gender, body_vtemp):
        """Set vertex template on the body model (default if unavailable)."""
        if body_vtemp == 'no_vtemp':
            self.bm_dict[gender].v_template = self.default_vtemp_dict[gender]
        else:
            verts = trimesh.load(body_vtemp).vertices
            self.bm_dict[gender].v_template = torch.tensor(
                verts, device=self.device, dtype=torch.float32)

    def _run_body_model_forward(self, gender, betas, trans, fullbody_aa):
        """Run SMPL-X forward pass with and without hand poses.

        Returns (joints, merged_vertices, raw_vertices) as numpy arrays.
        """
        psd = self._build_pose_shape_dict(betas, trans, fullbody_aa)
        merged_out = self.bm_dict[gender](**psd)

        # Zero out hands for the collision-check baseline
        psd['left_hand_pose'] = torch.zeros_like(psd['left_hand_pose'])
        psd['right_hand_pose'] = torch.zeros_like(psd['right_hand_pose'])
        base_out = self.bm_dict[gender](**psd)
        
        joints = merged_out.joints.detach().cpu().numpy()
        merged_verts = merged_out.vertices.detach().cpu().numpy()
        raw_verts = base_out.vertices.detach().cpu().numpy()

        del merged_out, base_out, psd
        torch.cuda.empty_cache()

        return joints, merged_verts, raw_verts

    # ------------------------------------------------------------------ #
    #  Hand-body merge helpers
    # ------------------------------------------------------------------ #

    def _merge_hand_into_body_item(self, v, k, hand_dataset_dict,
                                   hand_data_idx_list, even_hand_data_idx_list,
                                   time_augment_flag):
        """Merge randomly sampled hand poses into a single body item.

        Writes hand metadata into *v* and returns the full-body axis-angle tensor.
        """
        
        if v['body_dataset_name'].upper() not in self.hand_art_dataset_list \
            and not self.demo_mode:
            # Sample hand motion indices
            if time_augment_flag:
                rh_idx, lh_idx = np.random.choice(hand_data_idx_list, 2)
            else:
                rh_idx, lh_idx = np.random.choice(even_hand_data_idx_list, 2, replace=True)
                assert not hand_dataset_dict[rh_idx]['time_augment_flag']
                assert not hand_dataset_dict[lh_idx]['time_augment_flag']

            # Copy hand poses; mirror right-hand pose to get left-hand pose
            rh_pose = deepcopy(hand_dataset_dict[rh_idx]['rhand_pose'])
            lh_pose = deepcopy(hand_dataset_dict[lh_idx]['rhand_pose'])
            lh_pose.reshape(self.WIND, 15, 3)[:, :, 1:] *= -1

            body_pose = deepcopy(v['rots'][:, :75])

            # Adjust wrist orient for object-interaction datasets
            if hand_dataset_dict[rh_idx]['datasetname'] in self.object_dataset_list:
                body_pose[:, 72:75] = hand_dataset_dict[rh_idx]['relative_wrist_orient']

            if hand_dataset_dict[lh_idx]['datasetname'] in self.object_dataset_list:
                orient = deepcopy(hand_dataset_dict[lh_idx]['relative_wrist_orient'])
                orient[:, 1:] *= -1
                body_pose[:, 69:72] = orient

            # Store hand metadata
            v['rhand_motion_idx'] = rh_idx
            v['lhand_motion_idx'] = lh_idx
            v['rhand_dataset_name'] = hand_dataset_dict[rh_idx]['datasetname']
            v['lhand_dataset_name'] = hand_dataset_dict[lh_idx]['datasetname']
            v['rhand_pose_augment_flag'] = hand_dataset_dict[rh_idx]['pose_augment_flag']
            v['lhand_pose_augment_flag'] = hand_dataset_dict[lh_idx]['pose_augment_flag']
            v['rhand_time_augment_flag'] = hand_dataset_dict[rh_idx]['time_augment_flag']
            v['lhand_time_augment_flag'] = hand_dataset_dict[lh_idx]['time_augment_flag']

            fullbody_aa = torch.cat([body_pose, lh_pose, rh_pose], axis=1)
            v['rots'] = fullbody_aa
            assert fullbody_aa.shape[1] == self.NUM_JOINTS * 3

        else:
            # Dataset already has articulated hands — keep as-is
            v['rhand_motion_idx'] = k
            v['lhand_motion_idx'] = k
            v['rhand_dataset_name'] = v['body_dataset_name']
            v['lhand_dataset_name'] = v['body_dataset_name']
            v['rhand_pose_augment_flag'] = v['augment_flag']
            v['lhand_pose_augment_flag'] = v['augment_flag']
            v['rhand_time_augment_flag'] = False
            v['lhand_time_augment_flag'] = False
            fullbody_aa = v['rots']
            
        return fullbody_aa

    def _export_collision_mesh(self, merge_w_hand_verts, check_ids_list,
                               collision_mask, indices_in_collision_dict,
                               split, k):
        """Export a debug mesh with spheres marking collision vertices."""
        erroneous_indices = check_ids_list[
            torch.tensor(collision_mask).nonzero(as_tuple=True)[0]]
        if isinstance(erroneous_indices, np.int64):
            erroneous_indices = [erroneous_indices]

        idx = erroneous_indices[-1]
        frame_pos = np.where(check_ids_list == idx)[0].item()

        base_mesh = trimesh.Trimesh(
            vertices=merge_w_hand_verts[idx], faces=self.watertight_w_hand_faces)

        sphere_meshes = []
        for side in ['right', 'left']:
            for pos in indices_in_collision_dict[side][frame_pos]:
                sphere = trimesh.creation.icosphere(radius=0.003)
                sphere.vertices += base_mesh.vertices[pos]
                if side == 'right':
                    sphere.visual.face_colors = [200, 200, 250, 100]
                sphere_meshes.append(sphere)

        combined_mesh = base_mesh.copy()
        for sphere in sphere_meshes:
            combined_mesh = combined_mesh + sphere

        out_dir = self.collision_output_foldername.replace('SPLIT', split)
        combined_mesh.export(f"{out_dir}/{k:06d}.ply")

    def _check_collision_for_item(self, v, k, merged_verts, raw_verts, time_augment_flag):
        """Check if a merged item has hand-body collision.

        Returns True if collision is detected and the item needs retry.
        """
        if v['body_dataset_name'] in self.hand_art_dataset_list or not self.check_penetration_flag:
            return False

        collision_hand_dict, indices_dict, check_ids = \
            self.check_self_intersections(
                raw_verts, merged_verts, SAMPLING_FREQ=30,
                COLLISION_THRESHOLD_1=16, COLLISION_THRESHOLD_2=0)

        collision_mask = np.logical_or(
            collision_hand_dict['right'], collision_hand_dict['left'])

        if any(collision_mask):
            merge_w_hand_verts = merged_verts[:, self.base2watertight_w_hand]
            split = 'train' if time_augment_flag else 'val'
            self._export_collision_mesh(
                merge_w_hand_verts, check_ids, collision_mask,
                indices_dict, split, k)
            return True

        return False

    def _merge_single_item(self, v, k, hand_dataset_dict,
                           hand_data_idx_list, even_hand_data_idx_list,
                           time_augment_flag):
        """Merge hands into one body item, run SMPL-X forward, handle collisions.

        Retries with a different hand sample when a collision is detected.
        """
        collision_th1, collision_th2 = 16, 0

        while True:
            fullbody_aa = self._merge_hand_into_body_item(
                v, k, hand_dataset_dict, hand_data_idx_list,
                even_hand_data_idx_list, time_augment_flag)

            gender = v['gender']
            if 'obj_moving_frames' not in v:
                v['obj_moving_frames'] = torch.zeros((self.WIND))

            self._set_vertex_template(gender, v['body_vtemp'])

            joints, merged_verts, raw_verts = self._run_body_model_forward(
                gender, self._pad_betas(v['betas']), v['trans'], fullbody_aa)
            v['joint_positions'] = joints

            # Check for hand-body penetration
            if (v['body_dataset_name'] not in self.hand_art_dataset_list) and self.check_penetration_flag:
                merge_w_hand_verts = merged_verts[:, self.base2watertight_w_hand]

                collision_hand_dict, indices_dict, check_ids = \
                    self.check_self_intersections(
                        raw_verts, merged_verts, SAMPLING_FREQ=30,
                        COLLISION_THRESHOLD_1=collision_th1,
                        COLLISION_THRESHOLD_2=collision_th2)

                collision_th1 += 8  # relax thresholds on each retry
                collision_th2 += 8

                collision_mask = np.logical_or(
                    collision_hand_dict['right'], collision_hand_dict['left'])
                if any(collision_mask):
                    split = 'train' if time_augment_flag else 'val'
                    self._export_collision_mesh(
                        merge_w_hand_verts, check_ids, collision_mask,
                        indices_dict, split, k)
                    continue  # retry with new hand sample

            break

    # ------------------------------------------------------------------ #
    #  Collision detection helpers
    # ------------------------------------------------------------------ #

    def _detect_hand_body_collision(self, collision_idxs, hand_face_ids):
        """Detect bidirectional collisions between hand faces and body faces."""
        hand_to_body = torch.logical_and(
            torch.isin(collision_idxs[:, 0], hand_face_ids),
            torch.isin(collision_idxs[:, 1], self.base_hand_body_face_ids))
        body_to_hand = torch.logical_and(
            torch.isin(collision_idxs[:, 0], self.base_hand_body_face_ids),
            torch.isin(collision_idxs[:, 1], hand_face_ids))
        return torch.logical_or(hand_to_body, body_to_hand)

    def _get_collision_vertex_indices(self, collision_idxs, collision_mask):
        """Map colliding base-mesh face pairs to watertight vertex indices."""
        face_indices = np.unique(collision_idxs[collision_mask].flatten().cpu())
        face_indices = [self.base2watertight_faces_dict[e]
                        for e in face_indices if e in self.base2watertight_faces_dict]
        return np.unique(self.watertight_w_hand_faces[face_indices].flatten())

    # ------------------------------------------------------------------ #
    #  Dataset I/O
    # ------------------------------------------------------------------ #

    def save_dataset(self, dataset_dict, datapath):
        print(f'Saving data to {datapath}...')
        for k, v in tqdm(dataset_dict.items()):
            joblib.dump(v, os.path.join(datapath, f'{k:06d}.p'))

    def load_dataset(self, datapath, ram_threshold_gb=750):
        """Return a LazyLoadDict backed by precomputed .p files on disk.
        If free RAM exceeds ram_threshold_gb, load everything eagerly."""
        free_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
        eager = free_ram_gb > ram_threshold_gb
        mode = "eager" if eager else "lazy"
        print(f'Free RAM: {free_ram_gb:.1f} GB (threshold: {ram_threshold_gb} GB) '
              f'-> {mode} loading from {datapath}')
        return LazyLoadDict(datapath, eager=eager)

    # ------------------------------------------------------------------ #
    #  Data gathering
    # ------------------------------------------------------------------ #

    def gather_hand_data(self):
        dataset_dict_list = []
        print("")
        processed_paths = [p.replace('.p', '_processed.p') for p in self.hand_datapath_list]

        for raw_path, proc_path in tqdm(
                zip(self.hand_datapath_list, processed_paths),
                total=len(processed_paths),
                desc='Loading hand datasets'):
            if os.path.exists(proc_path):
                print(f'Already processed hand data. Loading from {proc_path}...')
                dataset_dict = joblib.load(proc_path)
                self.hand_seq_idx += len(dataset_dict)
            else:
                print(f'No processed hand data. Loading from {raw_path}...')
                self.unprocessed_hand_flag = True
                dataset_dict = self.process_hand_data(raw_path, proc_path)
            dataset_dict_list.append(dataset_dict)

        return cast_dict_to_tensors(concatenate_data(dataset_dict_list))

    def shift_indices(self, datadict):
        expected = np.arange(self.body_seq_idx, self.body_seq_idx + len(datadict))
        if (expected == np.array(list(datadict.keys()))).all():
            return datadict
        return {self.body_seq_idx + i: v for i, (_, v) in enumerate(datadict.items())}

    def gather_body_data(self, raw_list, processed_list, include_pose_augment):
        dataset_dict_list = []
        self.body_seq_idx = 0
        print("")
        unprocessed_body_flag = False

        for raw_path, proc_path in tqdm(
                zip(raw_list, processed_list),
                total=len(processed_list),
                desc='Loading body datasets'):
            if os.path.exists(proc_path):
                print(f'Already processed body data. Loading from {proc_path}...')
                dataset_dict = self.shift_indices(joblib.load(proc_path))
                self.body_seq_idx += len(dataset_dict)
            else:
                print(f'No processed body data. Loading from {raw_path}...')
                unprocessed_body_flag = True
                dataset_dict = self.process_body_data(raw_path, proc_path, include_pose_augment)
            dataset_dict_list.append(dataset_dict)

        body_dataset_dict = cast_dict_to_tensors(concatenate_data(dataset_dict_list))
        return body_dataset_dict, unprocessed_body_flag

    # ------------------------------------------------------------------ #
    #  Merge datasets (batch version — kept for external callers)
    # ------------------------------------------------------------------ #

    def merge_datasets(self, body_dataset_dict, hand_dataset_dict, time_augment_flag):
        print('')
        hand_data_idx_list = list(hand_dataset_dict.keys())
        even_hand_data_idx_list = [e for e in hand_data_idx_list
                                   if not hand_dataset_dict[e]['time_augment_flag']]

        merged = body_dataset_dict.copy()
        for k, v in tqdm(merged.items()):
            self._merge_single_item(
                v, k, hand_dataset_dict, hand_data_idx_list,
                even_hand_data_idx_list, time_augment_flag)

        print("Merged hand and body datasets.")
        return merged

    # ------------------------------------------------------------------ #
    #  Self-intersection checking
    # ------------------------------------------------------------------ #

    def check_self_intersections(self, raw_body_vertices, merge_body_vertices,
                                  SAMPLING_FREQ, COLLISION_THRESHOLD_1, COLLISION_THRESHOLD_2):

        search_tree = BVH(max_collisions=2)
        check_ids_list = np.arange(0, self.WIND, SAMPLING_FREQ)

        merge_verts = merge_body_vertices[check_ids_list]
        raw_verts = raw_body_vertices[check_ids_list]
        bs, nv = merge_verts.shape[:2]

        face_tensor = torch.tensor(
            self.bm_dict['neutral'].faces.astype(np.int64),
            dtype=torch.long, device=self.device,
        ).unsqueeze_(0).repeat([bs, 1, 1])

        faces_idx = face_tensor + (
            torch.arange(bs, dtype=torch.long, device=self.device) * nv
        )[:, None, None]

        triangles = torch.from_numpy(merge_verts).to(self.device).reshape([-1, 3])[faces_idx]
        collision_idxs = search_tree(triangles)
        sentinel = torch.tensor([-1, -1], device=self.device)
        collision_idxs_list = [e[torch.where(e != sentinel)[0]] for e in collision_idxs]

        indices_in_collision = {"right": [], "left": []}
        collision_hand_dict = {"right": [], "left": []}

        for idx in range(len(check_ids_list)):
            cidxs = collision_idxs_list[idx]

            lhand_collision = self._detect_hand_body_collision(cidxs, self.base_lhand_face_ids)
            rhand_collision = self._detect_hand_body_collision(cidxs, self.base_rhand_face_ids)

            lhand_count = lhand_collision.sum().cpu().item()
            rhand_count = rhand_collision.sum().cpu().item()

            if lhand_count > COLLISION_THRESHOLD_1 or rhand_count > COLLISION_THRESHOLD_1:
                # Compare against raw (un-augmented) body collisions
                raw_triangles = torch.from_numpy(raw_verts).to(self.device).reshape([-1, 3])[faces_idx]
                raw_collision_idxs = search_tree(raw_triangles)
                raw_cidxs_list = [e[torch.where(e != sentinel)[0]] for e in raw_collision_idxs]

                raw_cidxs = raw_cidxs_list[idx]
                raw_lhand = self._detect_hand_body_collision(raw_cidxs, self.base_lhand_face_ids)
                raw_rhand = self._detect_hand_body_collision(raw_cidxs, self.base_rhand_face_ids)

                collision_hand_dict['left'].append(
                    lhand_count > raw_lhand.sum().cpu().item() + COLLISION_THRESHOLD_2)
                collision_hand_dict['right'].append(
                    rhand_count > raw_rhand.sum().cpu().item() + COLLISION_THRESHOLD_2)
            else:
                collision_hand_dict['left'].append(lhand_count > COLLISION_THRESHOLD_1)
                collision_hand_dict['right'].append(rhand_count > COLLISION_THRESHOLD_1)

            indices_in_collision['left'].append(
                self._get_collision_vertex_indices(cidxs, lhand_collision))
            indices_in_collision['right'].append(
                self._get_collision_vertex_indices(cidxs, rhand_collision))

        return collision_hand_dict, indices_in_collision, check_ids_list

    # ------------------------------------------------------------------ #
    #  Streaming merge + precompute + save
    # ------------------------------------------------------------------ #
    
    def merge_precompute_and_save_streaming(self, body_dataset_dict, hand_dataset_dict,
                                            time_augment_flag, save_folder,
                                            object_dataset_list, save_stats=False):
        """Streaming pipeline: merge body+hand, compute features, and save to disk
        one item at a time to avoid OOM. Uses running statistics."""

        print('\nStreaming merge + precompute + save ...')

        hand_data_idx_list = list(hand_dataset_dict.keys())
        even_hand_data_idx_list = [e for e in hand_data_idx_list
                                   if not hand_dataset_dict[e]['time_augment_flag']]

        load_feats = (['body_transl'] + self.load_feats) \
            if 'body_transl' not in self.load_feats else list(self.load_feats)

        running_stats = {}
        temp_dataset, feature_names = None, None
        # idx -> dataset name sidecar for weighted sampling (written below).
        index_meta = {}

        for k in tqdm(list(body_dataset_dict.keys())):
            v = body_dataset_dict[k]

            # ---- Merge hands into body ----
            self._merge_single_item(
                v, k, hand_dataset_dict, hand_data_idx_list,
                even_hand_data_idx_list, time_augment_flag)

            # ---- Compute features ----
            if temp_dataset is None:
                temp_dataset = AmassDataset(
                    {k: v},
                    n_body_joints=self.NUM_JOINTS,
                    stats_file=self.preproc.stats_file,
                    norm_type=self.preproc.norm_type,
                    rot_repr=self.rot_repr,
                    device=self.device,
                    object_dataset_list=object_dataset_list,
                    load_feats=self.load_feats)

            temp_dataset.data = {k: v}
            x = dict(temp_dataset.get_all_features(k, load_feats))
            v['precomputed_features'] = x

            # ---- Update running statistics ----
            if feature_names is None:
                feature_names = [name for name in x if torch.is_tensor(x[name])]
                for name in feature_names:
                    D = x[name].shape[-1]
                    running_stats[name] = {
                        'sum':    torch.zeros(D, dtype=torch.float64),
                        'sum_sq': torch.zeros(D, dtype=torch.float64),
                        'min':    torch.full((D,), float('inf')),
                        'max':    torch.full((D,), float('-inf')),
                        'count':  0,
                    }

            for name in feature_names:
                feat = x[name].float()
                s = running_stats[name]
                s['sum']    += feat.sum(0).double()
                s['sum_sq'] += (feat ** 2).sum(0).double()
                s['min']     = torch.min(s['min'], feat.min(0)[0])
                s['max']     = torch.max(s['max'], feat.max(0)[0])
                s['count']  += feat.shape[0]

            # ---- Save & free ----
            if self.demo_mode:
                joblib.dump(v, os.path.join(save_folder))
            else:
                joblib.dump(v, os.path.join(save_folder, f'{k:06d}.p'))
                index_meta[int(k)] = v['body_dataset_name']
            del body_dataset_dict[k], x, v

        # ---- Write dataset-index sidecar (for weighted sampling) ----
        if not self.demo_mode:
            sidecar = os.path.join(save_folder, 'dataset_index.json')
            with open(sidecar, 'w') as f:
                json.dump({str(k): nm for k, nm in index_meta.items()}, f)
            print(f'Wrote dataset index sidecar -> {sidecar}')

        # ---- Finalize statistics ----
        if save_stats and feature_names is not None:
            self._save_feature_stats(running_stats, feature_names)

        print("Streaming pipeline complete.")

    # ------------------------------------------------------------------ #
    #  Feature precomputation (non-streaming, kept for external callers)
    # ------------------------------------------------------------------ #

    def precompute_features(self, dataset, save_stats=False):
        """Precompute features from pose/translation/shape/joints across an entire dataset."""
        load_feats = (['body_transl'] + self.load_feats) \
            if 'body_transl' not in self.load_feats else self.load_feats

        random_idx = np.random.choice(list(dataset.data.keys()), 1).item()
        feature_names = list(dataset.get_all_features(random_idx, load_feats).keys())
        feature_dict = {name: [] for name in feature_names}

        for i in tqdm(dataset.data.keys()):
            x = dict(dataset.get_all_features(i, load_feats))
            dataset.data[i]['precomputed_features'] = x
            for name in feature_names:
                if torch.is_tensor(x[name]):
                    feature_dict[name].append(x[name])

        feature_dict = {name: torch.cat(vals, dim=0).float()
                        for name, vals in feature_dict.items()}

        if save_stats:
            self.stats = {
                name: {
                    'max': tensor.max(0)[0].numpy(),
                    'min': tensor.min(0)[0].numpy(),
                    'mean': tensor.mean(0).numpy(),
                    'std': tensor.std(0).numpy(),
                }
                for name, tensor in feature_dict.items()
            }
            self.stats['concatenated_features'] = {}
            for stat_name in ['max', 'min', 'mean', 'std']:
                self.stats['concatenated_features'][stat_name] = np.concatenate(
                    [self.stats[fn][stat_name] for fn in self.load_feats])

            np.save(self.stat_path, self.stats)
            print(f"\nSaved feature stats to {self.stat_path}")

        return dataset.data

    def _save_feature_stats(self, running_stats, feature_names):
        """Compute and save final statistics from running accumulators."""
        self.stats = {}
        for name in feature_names:
            s = running_stats[name]
            mean = (s['sum'] / s['count']).float()
            variance = torch.clamp((s['sum_sq'] / s['count']).float() - mean ** 2, min=0)
            self.stats[name] = {
                'max':  s['max'].numpy(),
                'min':  s['min'].numpy(),
                'mean': mean.numpy(),
                'std':  variance.sqrt().numpy(),
            }

        self.stats['concatenated_features'] = {}
        for stat_name in ['max', 'min', 'mean', 'std']:
            self.stats['concatenated_features'][stat_name] = np.concatenate(
                [self.stats[fn][stat_name] for fn in self.load_feats])

        np.save(self.stat_path, self.stats)
        print(f"\nSaved feature stats to {self.stat_path}")

    # ------------------------------------------------------------------ #
    #  Weighted sampling (per-dataset coefficients)
    # ------------------------------------------------------------------ #

    def _sampling_enabled(self):
        """True when the optional `sampling:` config block is present & enabled."""
        s = self.sampling_cfg
        return s is not None and bool(s.get('enabled', True))

    def _canonical_body_dataset(self, name):
        """Map a stored body_dataset_name to its canonical body dataset.

        AMASS is the only body dataset stored per sub-dataset (EKUT, KIT, CMU,
        HumanEva, WEIZMANN, ...); every other dataset stores its canonical name.
        For weighted sampling we aggregate all AMASS sub-datasets under the
        single 'AMASS' coefficient. The non-AMASS canonical names are derived
        once from the training datapath filenames (e.g. 'omomo_train.p' ->
        'OMOMO') so the set stays in sync with the config without a hardcoded
        list; anything not in that set is treated as an AMASS sub-dataset.
        """
        if not hasattr(self, '_canonical_dataset_names'):
            names = set()
            for p in self.body_train_datapath_list:
                base = os.path.basename(p)
                for sfx in ('_train.p', '_val.p', '_test.p', '.p'):
                    if base.endswith(sfx):
                        base = base[:-len(sfx)]
                        break
                names.add(base.upper())
            self._canonical_dataset_names = names
        return name if name in self._canonical_dataset_names else 'AMASS'

    def _train_index_to_dataset(self):
        """Return {train_index -> body_dataset_name} for the train precomputed set.

        Reads the `dataset_index.json` sidecar written during precompute. If it
        is missing (e.g. data precomputed before this feature), it is rebuilt
        once from the contiguous index ranges of the per-dataset
        `*_processed.p` files (indices are contiguous per dataset, in
        datapath-list order) and cached. This is a one-time startup cost; it
        adds no per-step overhead.
        """
        sidecar = os.path.join(self.all_precomputed_foldernames['train'],
                               'dataset_index.json')
        if os.path.exists(sidecar):
            with open(sidecar) as f:
                return {int(k): v for k, v in json.load(f).items()}

        print('No dataset_index.json sidecar found; building it once from '
              'processed body files (one-time cost)...')
        idx2name, offset = {}, 0
        for proc_path in self.body_processed_train_list:
            if not os.path.exists(proc_path):
                raise FileNotFoundError(
                    f'Cannot build weighted-sampling index: missing {proc_path}. '
                    f'Re-run precompute or disable sampling in configs/data.yaml.')
            data = joblib.load(proc_path)
            keys = sorted(data.keys())
            # Read each sequence's own body_dataset_name. AMASS files are
            # heterogeneous (sequences carry sub-dataset names: EKUT, KIT, ...),
            # so the previous first-key shortcut collapsed all of AMASS to a
            # single sub-dataset. Aggregation to canonical names happens in
            # _build_train_sample_weights via _canonical_body_dataset.
            for j, key in enumerate(keys):
                idx2name[offset + j] = data[key]['body_dataset_name']
            offset += len(keys)
            del data
            gc.collect()

        with open(sidecar, 'w') as f:
            json.dump({str(k): v for k, v in idx2name.items()}, f)
        print(f'Wrote sampling index sidecar -> {sidecar} ({offset} sequences).')
        return idx2name

    def _build_train_sample_weights(self):
        """Compute one weight per train sample from its dataset's coefficient.

        strategy == 'target_share'   : weight = c_d / N_d  -> P(dataset d) ∝ c_d
        strategy == 'size_multiplier': weight = c_d         -> P(dataset d) ∝ N_d·c_d
        Leaves train_sample_weights = None (uniform shuffle) on any mismatch.
        """
        from collections import Counter

        n = len(self.dataset['train'])
        idx2name = self._train_index_to_dataset()
        if len(idx2name) != n:
            print(f'WARNING: sampling index size {len(idx2name)} != train set '
                  f'size {n}. Falling back to uniform shuffle.')
            return

        # Aggregate AMASS sub-datasets (EKUT, KIT, HumanEva, ...) under the
        # single 'AMASS' coefficient; non-AMASS names pass through unchanged.
        # Robust to either a per-sequence sidecar or the stale collapsed one.
        idx2name = {i: self._canonical_body_dataset(nm) for i, nm in idx2name.items()}

        counts = Counter(idx2name.values())
        s = self.sampling_cfg
        strategy = s.get('strategy', 'target_share')
        default_coef = float(s.get('default_coefficient', 1.0))
        coeffs = dict(s.get('coefficients', {}) or {})

        weights = torch.ones(n, dtype=torch.double)
        for i in range(n):
            name = idx2name[i]
            c = float(coeffs.get(name, default_coef))
            if strategy == 'target_share':
                weights[i] = c / max(counts[name], 1)
            elif strategy == 'size_multiplier':
                weights[i] = c
            else:
                raise ValueError(f"Unknown sampling.strategy '{strategy}' "
                                 f"(use 'target_share' or 'size_multiplier').")

        self.train_sample_weights = weights

        # Report resulting expected per-dataset sampling shares.
        total = weights.sum().item()
        share = {name: 0.0 for name in counts}
        for i in range(n):
            share[idx2name[i]] += weights[i].item()
        print(f"Weighted training sampling enabled (strategy='{strategy}'). "
              f"Expected per-dataset shares:")
        for name in sorted(share):
            print(f"  {name:14s} N={counts[name]:>8d}  "
                  f"coef={float(coeffs.get(name, default_coef)):.4g}  "
                  f"share={share[name] / total:7.2%}")

    # ------------------------------------------------------------------ #
    #  DataLoaders
    # ------------------------------------------------------------------ #

    def train_dataloader(self):
        if self.train_sample_weights is not None:
            # WeightedRandomSampler precomputes all indices in one vectorized
            # multinomial draw per epoch — same per-step cost as shuffle=True.
            sampler = WeightedRandomSampler(
                self.train_sample_weights,
                num_samples=len(self.dataset['train']),
                replacement=True)
            return DataLoader(self.dataset['train'], sampler=sampler,
                              **self.train_dataloader_options)
        return DataLoader(self.dataset['train'], shuffle=True, **self.train_dataloader_options)

    def val_dataloader(self):
        return DataLoader(self.dataset['val'], shuffle=False, **self.val_dataloader_options)

    def test_dataloader(self):
        return DataLoader(self.dataset['test'], shuffle=False, **self.val_dataloader_options)

    # ------------------------------------------------------------------ #
    #  Raw data processing
    # ------------------------------------------------------------------ #

    def process_body_data(self, raw_datapath, processed_datapath, include_pose_augment):
        raw_data = joblib.load(raw_datapath)
        processed_data = dict()

        for k, v in tqdm(raw_data.items()):
            T = len(v['pose_body'])
            if (not include_pose_augment) and v['augment_flag']:
                continue
            assert T == self.WIND

            frames = torch.arange(T, dtype=torch.long)
            chunks = frames.unfold(dimension=0, size=self.WIND, step=self.WIND)

            for chunk in chunks:
                trans = torch.tensor(v['trans'][chunk]).float()
                betas = self._pad_betas(torch.tensor(v['betas']).float())
                betas = betas[None].repeat_interleave(self.WIND, dim=0).reshape(self.WIND, -1)

                fullbody_aa = torch.tensor(np.concatenate(
                    [v['root_orient'][chunk], v['pose_body'][chunk],
                     v['pose_jaw'][chunk], v['pose_eye'][chunk],
                     v['pose_lhand'][chunk], v['pose_rhand'][chunk]],
                    axis=1)[chunk].reshape(self.WIND, -1)).float()

                entry = {
                    'rots': fullbody_aa,
                    'trans': trans,
                    'betas': betas,
                    'fps': 30,
                    'gender': v['gender'],
                    'body_dataset_name': v['body_dataset_name'],
                    'augment_flag': v['augment_flag'],
                    'contacts_mask': v['contacts_mask'],
                    'root_offset': v['root_offset'],
                    'pos_offset': v['pos_offset'],
                }

                if v['body_dataset_name'] in self.object_dataset_list:
                    entry.update({
                        'obj_name': v['obj_name'],
                        'obj_trans': v['obj_trans'],
                        'obj_arti': v['obj_arti'],
                        'obj_orient': v['obj_orient'],
                        'obj_scale': v['obj_scale'],
                    })
                    assert v['obj_scale'].shape == (self.WIND,), \
                        str(v['obj_scale'].shape) + v['body_dataset_name']

                    # Two-part objects (OMOMO) — only for non-augmented/test data
                    has_bottom = any(a in v for a in
                                     ['obj_bottom_scale', 'obj_bottom_trans', 'obj_bottom_orient'])
                    if has_bottom and not include_pose_augment:
                        entry['obj_bottom_scale'] = v['obj_bottom_scale']
                        entry['obj_bottom_trans'] = v['obj_bottom_trans']
                        entry['obj_bottom_orient'] = v['obj_bottom_orient']
                    else:
                        entry['obj_bottom_scale'] = np.zeros((self.WIND))
                        entry['obj_bottom_trans'] = np.zeros((self.WIND, 3))
                        entry['obj_bottom_orient'] = np.zeros((self.WIND, 3))

                    if 'obj_moving_frames' in v:
                        entry['obj_moving_frames'] = v['obj_moving_frames']

                else:
                    entry.update({
                        'obj_name': 'no_object',
                        'obj_trans': np.zeros((self.WIND, 3)),
                        'obj_arti': np.zeros((self.WIND)),
                        'obj_orient': np.zeros((self.WIND, 3)),
                        'obj_scale': np.ones((self.WIND)),
                        'obj_bottom_scale': np.zeros((self.WIND)),
                        'obj_bottom_trans': np.zeros((self.WIND, 3)),
                        'obj_bottom_orient': np.zeros((self.WIND, 3)),
                    })
             
                entry['body_vtemp'] = v.get('body_vtemp', 'no_vtemp')
                entry['id'] = self.body_seq_idx

                processed_data[self.body_seq_idx] = entry
                self.body_seq_idx += 1

        joblib.dump(processed_data, processed_datapath)
        return processed_data

    def process_hand_data(self, raw_datapath, processed_datapath):
        raw_data = joblib.load(raw_datapath)
        hand_processed_data = dict()

        for k, v in tqdm(raw_data.items()):
            T = len(v['pose_rhand'])
            assert T == self.WIND
            assert v['pose_rhand'].shape[1] == 45, v['datasetname']

            entry = {
                'fps': 30,
                'gender': v['gender'],
                'rhand_pose': v['pose_rhand'],
                'datasetname': v['datasetname'],
                'pose_augment_flag': v['pose_augment_flag'],
                'time_augment_flag': v['time_augment_flag'],
            }

            if v['datasetname'] in self.object_dataset_list:
                entry['rhand_vtemp'] = v['rhand_vtemp']
                entry['lhand_vtemp'] = v['lhand_vtemp']
            hand_processed_data[self.hand_seq_idx] = entry
            self.hand_seq_idx += 1

        joblib.dump(hand_processed_data, processed_datapath)
        return hand_processed_data

