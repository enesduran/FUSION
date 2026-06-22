"""Append-only patch: add InterX windows to an already-populated precomputed cache.

The full precomputed pipeline (process body datasets, gather hands, merge,
compute features, dump per-window .p files) is a multi-day run. This script
covers the case where every dataset *except* InterX is already cached in
`data/motion/precomputed/{train,val}/` and we just want to splice InterX in
without redoing the rest.

Steps performed:
  1. Run `AmassWrapper.process_body_data` on `interx_{train,val}.p` to
     produce the `_processed.p` chunked files (fast, single-threaded).
  2. Load the cached hand dataset dict via `gather_hand_data` (also fast --
     just reads existing `Hand_Processed/*_processed.p`).
  3. For each InterX window: merge a hand sample, run SMPL-X forward, compute
     `load_feats` features, dump to `precomputed/{split}/NNNNNN.p` starting at
     `max_existing_index + 1`.
  4. Append entries to `precomputed/train/dataset_index.json` (val has no
     sidecar; the train sidecar is what the weighted sampler reads).
  5. Print the new TRAIN_THRESHOLD / VAL_THRESHOLD values to set in
     `configs/data.yaml`.

Feature normalization stats (`statistics.npy`) are intentionally NOT
updated: InterX's contribution to mean/std across ~400k windows is below the
noise floor of normal training.

Usage:
    python -m src.scripts.process.append_interx_precomputed
"""
import os
import sys
import glob
import json
import joblib
import numpy as np
import torch
from tqdm import tqdm
from omegaconf import OmegaConf

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.data.amass_wrapper import AmassWrapper
from src.data.amass_dataset import AmassDataset
from src.utils.genutils import cast_dict_to_tensors


DATA_CFG_PATH = 'configs/data.yaml'
INTERX_TRAIN_RAW = 'data/motion/Body_Processed/interx_train.p'
INTERX_VAL_RAW = 'data/motion/Body_Processed/interx_val.p'
PRECOMPUTED_ROOT = 'data/motion/precomputed'


def _max_existing_index(folder: str) -> int:
    """Highest NNNNNN.p index in `folder`, or -1 if empty."""
    files = glob.glob(os.path.join(folder, '*.p'))
    indices = []
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        if stem.isdigit():
            indices.append(int(stem))
    return max(indices) if indices else -1


def _build_partial_wrapper(cfg) -> AmassWrapper:
    """Construct an AmassWrapper-shaped object with just the bits we need.

    Skips the full __init__ -- which would re-run gather/process for every
    dataset and overwrite the existing precomputed cache.
    """
    w = AmassWrapper.__new__(AmassWrapper)
    w.device = torch.device(cfg.get('device', 'cuda:0') if torch.cuda.is_available() else 'cpu')
    w.WIND = cfg.get('wind', 120)
    w.rot_repr = cfg.rot_repr
    w.preproc = cfg.preproc
    w.stat_path = cfg.preproc.stats_file
    w.smplx_path = cfg.smplx_path
    w.object_dataset_list = list(cfg.object_dataset_list)
    w.hand_art_dataset_list = list(cfg.hand_art_dataset_list)
    w.hand_datapath_list = list(cfg.hand_datapath_list)
    w.body_train_datapath_list = list(cfg.body_train_datapath_list)
    w.load_feats = list(cfg.load_feats)
    w.check_penetration_flag = bool(cfg.check_penetration_flag)
    w.demo_mode = False
    w.hand_seq_idx = 0
    w.body_seq_idx = 0
    w.unprocessed_hand_flag = False

    # Body models (sets w.bm_dict, w.default_vtemp_dict, w.NUM_JOINTS).
    w._create_body_models(w.smplx_path, w.device)

    # Optional collision data (only when check_penetration_flag).
    if w.check_penetration_flag:
        import joblib as _jl
        w.watertight_conversion_dict = _jl.load('data/body_models/watertight/conversion_dict.pkl')
        w._init_penetration_data()

    return w


def _process_split(
    wrapper: AmassWrapper,
    interx_raw_path: str,
    interx_processed_path: str,
    split: str,
    hand_dataset_dict: dict,
    include_time_augment: bool,
    include_pose_augment: bool,
):
    """Process and append one InterX split. Returns the number of windows added."""
    print(f'\n=== {split.upper()} ===')

    # ---- Step 1: chunk raw -> _processed.p (if not already there) ----
    if os.path.exists(interx_processed_path):
        print(f'Loading existing {interx_processed_path}')
        chunked = joblib.load(interx_processed_path)
    else:
        print(f'Chunking {interx_raw_path} -> {interx_processed_path}')
        wrapper.body_seq_idx = 0
        chunked = wrapper.process_body_data(
            interx_raw_path, interx_processed_path, include_pose_augment)

    chunked = cast_dict_to_tensors(chunked)
    print(f'InterX {split}: {len(chunked)} windows')

    if len(chunked) == 0:
        return 0
    
    # ---- Step 2: locate the precomputed folder and starting index ----
    folder = os.path.join(PRECOMPUTED_ROOT, split)
    os.makedirs(folder, exist_ok=True)
    start_idx = _max_existing_index(folder) + 1
    print(f'Appending starting at index {start_idx} in {folder}')

    # ---- Step 3: per-item merge + features + dump ----
    hand_idx_list = list(hand_dataset_dict.keys())
    even_hand_idx_list = [k for k in hand_idx_list
                          if not hand_dataset_dict[k]['time_augment_flag']]

    load_feats = (['body_transl'] + wrapper.load_feats) \
        if 'body_transl' not in wrapper.load_feats else list(wrapper.load_feats)

    # The canonical-name set built by the wrapper from datapath filenames is
    # all-uppercase ({'INTERX', 'OMOMO', ...}). process_interx_body.py stores
    # 'InterX' (mixed case), which would fall back to 'AMASS' in
    # _canonical_body_dataset and break the INTERX: 1.3 sampling coefficient.
    # Normalize to upper-case here so the weighted sampler picks it up.

    # Defer AmassDataset construction until after the first merge -- its
    # __init__ probes a sample item, and feature getters need joint_positions
    # (set inside _merge_single_item) plus 'id'.
    temp_dataset = None

    new_index_meta = {}
    chunk_keys = list(chunked.keys())
    for i, k in enumerate(tqdm(chunk_keys, desc=f'Merging {split}')):
        v = chunked[k]

        # Normalize BEFORE merge: _merge_single_item branches on this name
        # against wrapper.hand_art_dataset_list, which holds upper-case
        # entries ('INTERX'). Without this, 'InterX' would fall through to
        # the random-hand-sampling branch.
        v['body_dataset_name'] = str(v['body_dataset_name']).upper()

        wrapper._merge_single_item(
            v, k, hand_dataset_dict, hand_idx_list, even_hand_idx_list,
            include_time_augment)

        new_idx = start_idx + i
        v['id'] = new_idx

        if temp_dataset is None:
            temp_dataset = AmassDataset(
                {k: v},
                n_body_joints=wrapper.NUM_JOINTS,
                stats_file=wrapper.preproc.stats_file,
                norm_type=wrapper.preproc.norm_type,
                rot_repr=wrapper.rot_repr,
                device=wrapper.device,
                object_dataset_list=wrapper.object_dataset_list,
                load_feats=wrapper.load_feats,
            )
        temp_dataset.data = {k: v}
        x = dict(temp_dataset.get_all_features(k, load_feats))
        v['precomputed_features'] = x

        out_path = os.path.join(folder, f'{new_idx:06d}.p')
        joblib.dump(v, out_path)
        new_index_meta[int(new_idx)] = v['body_dataset_name']

        del chunked[k], x

    # ---- Step 4: append to dataset_index.json (train only matters for sampler) ----
    sidecar = os.path.join(folder, 'dataset_index.json')
    existing_meta = {}
    if os.path.exists(sidecar):
        with open(sidecar) as f:
            existing_meta = json.load(f)
    existing_meta.update({str(k): nm for k, nm in new_index_meta.items()})
    with open(sidecar, 'w') as f:
        json.dump(existing_meta, f)
    print(f'Updated {sidecar} (+{len(new_index_meta)} entries)')

    return len(new_index_meta)


def main():
    cfg = OmegaConf.load(DATA_CFG_PATH)

    # Sanity: raw InterX outputs from process_interx_body.py must exist.
    for p in [INTERX_TRAIN_RAW, INTERX_VAL_RAW]:
        if not os.path.exists(p):
            sys.exit(f'Missing {p} -- run src/scripts/process/body/process_interx_body.py first.')

    print('Constructing partial AmassWrapper (skipping full init) ...')
    wrapper = _build_partial_wrapper(cfg)

    # Hand dataset dict: gather_hand_data reads existing _processed.p files.
    print('Loading hand dataset dict ...')
    hand_dataset_dict = wrapper.gather_hand_data()

    added_train = _process_split(
        wrapper,
        INTERX_TRAIN_RAW,
        INTERX_TRAIN_RAW.replace('.p', '_processed.p'),
        split='train',
        hand_dataset_dict=hand_dataset_dict,
        include_time_augment=bool(cfg.include_time_augmentation),
        include_pose_augment=bool(cfg.include_pose_augmentation),
    )

    added_val = _process_split(
        wrapper,
        INTERX_VAL_RAW,
        INTERX_VAL_RAW.replace('.p', '_processed.p'),
        split='val',
        hand_dataset_dict=hand_dataset_dict,
        include_time_augment=False,    # val never gets time augmentation
        include_pose_augment=False,    # val never gets pose augmentation
    )

    new_train_threshold = int(cfg.TRAIN_THRESHOLD) + added_train
    new_val_threshold = int(cfg.VAL_THRESHOLD) + added_val
    print('\n========================================')
    print(f'Added {added_train} train windows, {added_val} val windows.')
    print(f'Update configs/data.yaml:')
    print(f'  TRAIN_THRESHOLD: {new_train_threshold}   (was {int(cfg.TRAIN_THRESHOLD)})')
    print(f'  VAL_THRESHOLD:   {new_val_threshold}   (was {int(cfg.VAL_THRESHOLD)})')
    print('========================================')


if __name__ == '__main__':
    main()
