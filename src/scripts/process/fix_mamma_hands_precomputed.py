"""Repair MAMMA / MAMMA_PREDS precomputed entries that got random hands.

The first full precompute run was launched with a `hand_art_dataset_list`
that was missing MAMMA and MAMMA_PREDS. AmassWrapper._merge_hand_into_body_item
(amass_wrapper.py:484) hard-branches on that list: datasets that are not in it
get their natural `pose_lhand` / `pose_rhand` replaced with random samples
from the hand datasets. As a result every MAMMA / MAMMA_PREDS precomputed
window has body motion from MAMMA paired with hand motion from some unrelated
recording -- the body-hand synchronicity is gone.

Recovery is possible without re-running the rest of the pipeline because:
  * `data/motion/Body_Processed/mamma{,_preds}_train_processed.p` still carry
    the natural hands inside `rots[..., 75:]` (process_body_data packs them
    in at amass_wrapper.py:1155-1159).
  * `precomputed/train/dataset_index.json` shows MAMMA at indices
    [393017, 393785) and MAMMA_PREDS at [393785, 398912) -- contiguous, and
    the chunked dicts have exactly matching lengths (768 / 5127), so the
    precomputed-to-chunked mapping is `precomputed_idx = range_start + i`.

This script:
  1. Loads each MAMMA / MAMMA_PREDS chunk from its `_processed.p` file.
  2. Re-runs `_merge_single_item` with both datasets injected into
     `hand_art_dataset_list` so the natural-hands branch
     (amass_wrapper.py:531-541) fires.
  3. Re-computes `load_feats` features.
  4. Overwrites the corresponding `precomputed/train/NNNNNN.p` files in place.

No sidecar update is needed (idx -> name mapping is unchanged). No threshold
update is needed (counts unchanged). Feature normalization stats are left
alone -- the change is small enough not to matter.

Usage:
    python -m src.scripts.process.fix_mamma_hands_precomputed
"""
import os
import sys
import json
import joblib
import torch
from tqdm import tqdm
from omegaconf import OmegaConf

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.data.amass_wrapper import AmassWrapper
from src.data.amass_dataset import AmassDataset
from src.utils.genutils import cast_dict_to_tensors


DATA_CFG_PATH = 'configs/data.yaml'
PRECOMPUTED_TRAIN = 'data/motion/precomputed/train'
SIDECAR = os.path.join(PRECOMPUTED_TRAIN, 'dataset_index.json')

# (dataset_name, chunked .p path)
DATASETS_TO_FIX = [
    ('MAMMA',       'data/motion/Body_Processed/mamma_train_processed.p'),
    ('MAMMA_PREDS', 'data/motion/Body_Processed/mamma_preds_train_processed.p'),
]


def _build_partial_wrapper(cfg) -> AmassWrapper:
    """Same skeleton as append_interx_precomputed._build_partial_wrapper, but
    with MAMMA / MAMMA_PREDS injected into hand_art_dataset_list so the
    natural-hands branch of _merge_single_item is taken."""
    w = AmassWrapper.__new__(AmassWrapper)
    w.device = torch.device(cfg.get('device', 'cuda:0') if torch.cuda.is_available() else 'cpu')
    w.WIND = cfg.get('wind', 120)
    w.rot_repr = cfg.rot_repr
    w.preproc = cfg.preproc
    w.stat_path = cfg.preproc.stats_file
    w.smplx_path = cfg.smplx_path
    w.object_dataset_list = list(cfg.object_dataset_list)
    # The fix: union the YAML list with the two missing names. Idempotent.
    w.hand_art_dataset_list = list(set(list(cfg.hand_art_dataset_list))
                                   | {'MAMMA', 'MAMMA_PREDS'})
    w.hand_datapath_list = list(cfg.hand_datapath_list)
    w.body_train_datapath_list = list(cfg.body_train_datapath_list)
    w.load_feats = list(cfg.load_feats)
    w.check_penetration_flag = bool(cfg.check_penetration_flag)
    w.demo_mode = False
    w.hand_seq_idx = 0
    w.body_seq_idx = 0
    w.unprocessed_hand_flag = False

    w._create_body_models(w.smplx_path, w.device)

    if w.check_penetration_flag:
        w.watertight_conversion_dict = joblib.load(
            'data/body_models/watertight/conversion_dict.pkl')
        w._init_penetration_data()

    return w


def _range_for(sidecar_path: str, name: str) -> tuple:
    """Return [start, end) of contiguous precomputed indices labelled `name`."""
    with open(sidecar_path) as f:
        d = json.load(f)
    matching = sorted(int(k) for k, v in d.items() if v == name)
    if not matching:
        raise RuntimeError(f'No entries with name {name!r} in {sidecar_path}')
    start, end = matching[0], matching[-1] + 1
    if list(range(start, end)) != matching:
        raise RuntimeError(
            f'{name} indices in sidecar are not contiguous; cannot use '
            f'simple range mapping')
    return start, end


def _fix_one(wrapper: AmassWrapper, dataset_name: str, chunked_path: str,
             hand_dataset_dict: dict, temp_dataset_holder: list):
    """Overwrite the precomputed files for one dataset with natural-hand merges."""
    print(f'\n--- {dataset_name} ---')

    start, end = _range_for(SIDECAR, dataset_name)
    chunked = joblib.load(chunked_path)
    chunked = cast_dict_to_tensors(chunked)

    chunked_keys = sorted(chunked.keys())
    if len(chunked_keys) != end - start:
        raise RuntimeError(
            f'{dataset_name}: chunked has {len(chunked_keys)} entries but '
            f'sidecar range is {end - start}; mapping is ambiguous')

    print(f'Rewriting precomputed indices [{start}, {end}) '
          f'from {chunked_path}')

    hand_idx_list = list(hand_dataset_dict.keys())
    even_hand_idx_list = [k for k in hand_idx_list
                          if not hand_dataset_dict[k]['time_augment_flag']]

    load_feats = (['body_transl'] + wrapper.load_feats) \
        if 'body_transl' not in wrapper.load_feats else list(wrapper.load_feats)

    for offset, ck in enumerate(tqdm(chunked_keys, desc=dataset_name)):
        v = chunked[ck]
        precomp_idx = start + offset

        # _merge_single_item now takes the natural-hands branch because
        # dataset_name is in wrapper.hand_art_dataset_list.
        wrapper._merge_single_item(
            v, ck, hand_dataset_dict, hand_idx_list, even_hand_idx_list,
            time_augment_flag=True,
        )

        # Match the canonical pipeline: id == filename index.
        v['id'] = precomp_idx
        # body_dataset_name in chunked is already uppercase ('MAMMA' /
        # 'MAMMA_PREDS'); preserve it for the sidecar's idx->name mapping.

        if temp_dataset_holder[0] is None:
            temp_dataset_holder[0] = AmassDataset(
                {ck: v},
                n_body_joints=wrapper.NUM_JOINTS,
                stats_file=wrapper.preproc.stats_file,
                norm_type=wrapper.preproc.norm_type,
                rot_repr=wrapper.rot_repr,
                device=wrapper.device,
                object_dataset_list=wrapper.object_dataset_list,
                load_feats=wrapper.load_feats,
            )
        td = temp_dataset_holder[0]
        td.data = {ck: v}
        x = dict(td.get_all_features(ck, load_feats))
        v['precomputed_features'] = x

        out_path = os.path.join(PRECOMPUTED_TRAIN, f'{precomp_idx:06d}.p')
        joblib.dump(v, out_path)

        del chunked[ck], x

    print(f'{dataset_name}: rewrote {end - start} files.')


def main():
    cfg = OmegaConf.load(DATA_CFG_PATH)

    for p in [SIDECAR] + [pp for _, pp in DATASETS_TO_FIX]:
        if not os.path.exists(p):
            sys.exit(f'Missing {p}')

    print('Constructing partial AmassWrapper (skipping full init) ...')
    print('  Injecting MAMMA/MAMMA_PREDS into hand_art_dataset_list.')
    wrapper = _build_partial_wrapper(cfg)

    print('Loading hand dataset dict ...')
    hand_dataset_dict = wrapper.gather_hand_data()

    temp_dataset_holder = [None]  # shared across datasets so __init__ runs once

    for name, chunked_path in DATASETS_TO_FIX:
        _fix_one(wrapper, name, chunked_path, hand_dataset_dict,
                 temp_dataset_holder)

    print('\n========================================')
    print('Done. Reminder: also add MAMMA and MAMMA_PREDS to')
    print('configs/data.yaml -> hand_art_dataset_list so future precomputes')
    print('keep their natural hands.')
    print('========================================')


if __name__ == '__main__':
    main()
