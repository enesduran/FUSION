"""Compact the train precomputed cache: drop orphan InterX block, renumber, rebuild sidecar.

Background
----------
Two InterX appends left the train precomputed folder in an inconsistent state:

  * An aborted FIRST append wrote 2276 InterX windows at indices
    [398912, 401187] but died BEFORE the sidecar-update step (which runs only
    after the whole loop). Those windows carry RANDOM hands (the run predated
    INTERX being in hand_art_dataset_list) and are duplicates of the good block.
    They exist on disk but are absent from dataset_index.json.

  * A complete SECOND append wrote the full 31762 natural-hand InterX windows at
    [401188, 432949] and updated the sidecar.

Net on-disk state: 432950 files (contiguous 0..432949) but the sidecar only
knows 430674 of them (gap at [398912, 401187]). The data module keys precomputed
files by *filename index == positional index* (LazyLoadDict / AmassDataset) and
the weighted sampler aligns the sidecar to those positions, so the cache must be
contiguous 0..N-1 with a matching sidecar. The load gate is
`len(files) == TRAIN_THRESHOLD`, so the extra 2276 files force a ~28h rebuild.

This script makes disk == sidecar == TRAIN_THRESHOLD (430674), contiguous
0..430673:

  Phase 0  back up sidecar; delete orphan files [398912, 401187].
  Phase 1  shift good block [401188, 432949] down by 2276 -> [398912, 430673]
           via os.rename (ascending, collision-free).
  Phase 2  rebuild dataset_index.json contiguous: [0,398911] keep names,
           [398912,430673] = 'INTERX'.
  Phase 3  set each shifted file's 'id' to its new index (idempotent, resumable).
  Phase 4  verify counts, contiguity, sidecar match, and spot-check hands/ids.

Run with TRAINING STOPPED. Phases 0-2 are fast and leave a loadable cache;
Phase 3 only restores the id==index invariant (used by eval/render naming) and
is safe to interrupt/re-run.

Usage:
    python -m src.scripts.process.compact_train_precomputed
"""
import os
import sys
import json
import shutil
import joblib
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

FOLDER = 'data/motion/precomputed/train'
SIDECAR = os.path.join(FOLDER, 'dataset_index.json')

ORPHAN_LO, ORPHAN_HI = 398912, 401187      # inclusive; 2276 files, not in sidecar
GOOD_LO, GOOD_HI = 401188, 432949          # inclusive; 31762 INTERX files
SHIFT = GOOD_LO - ORPHAN_LO                 # 2276
NEW_MAX = GOOD_HI - SHIFT                    # 430673
EXPECTED_TOTAL = NEW_MAX + 1                 # 430674


def _p(idx):
    return os.path.join(FOLDER, f'{idx:06d}.p')


def _disk_indices():
    out = set()
    with os.scandir(FOLDER) as it:
        for e in it:
            if e.is_file() and e.name.endswith('.p'):
                stem = e.name[:-2]
                if stem.isdigit():
                    out.add(int(stem))
    return out


def _check_preconditions():
    assert os.path.isdir(FOLDER), f'missing {FOLDER}'
    assert os.path.exists(SIDECAR), f'missing {SIDECAR}'

    disk = _disk_indices()
    side = {int(k): v for k, v in json.load(open(SIDECAR)).items()}

    expected_disk = set(range(0, GOOD_HI + 1))
    if disk != expected_disk:
        extra = sorted(disk - expected_disk)[:5]
        missing = sorted(expected_disk - disk)[:5]
        sys.exit(f'ABORT: disk is not contiguous [0, {GOOD_HI}]. '
                 f'extra={extra} missing={missing} (disk has {len(disk)} files). '
                 f'State differs from the expected post-double-append layout; '
                 f'do not run blindly.')

    expected_side = set(range(0, GOOD_HI + 1)) - set(range(ORPHAN_LO, ORPHAN_HI + 1))
    if set(side.keys()) != expected_side:
        sys.exit(f'ABORT: sidecar keys differ from expected. '
                 f'sidecar has {len(side)} keys; expected {len(expected_side)}.')

    bad = [k for k in range(GOOD_LO, GOOD_HI + 1) if side.get(k) != 'INTERX']
    if bad:
        sys.exit(f'ABORT: good block has non-INTERX sidecar entries, e.g. '
                 f'{bad[:5]} -> {[side[k] for k in bad[:5]]}')

    print('Preconditions OK:')
    print(f'  disk files       : {len(disk)}  (contiguous 0..{GOOD_HI})')
    print(f'  sidecar entries  : {len(side)}  (gap at [{ORPHAN_LO},{ORPHAN_HI}])')
    print(f'  orphans to delete: {ORPHAN_HI - ORPHAN_LO + 1}  [{ORPHAN_LO},{ORPHAN_HI}]')
    print(f'  good block       : {GOOD_HI - GOOD_LO + 1}  [{GOOD_LO},{GOOD_HI}] -> shift -{SHIFT}')
    print(f'  final            : {EXPECTED_TOTAL}  [0,{NEW_MAX}]')
    return side


def main():
    side = _check_preconditions()

    # ---- Phase 0: backup sidecar, delete orphans -------------------------- #
    bak = SIDECAR + '.bak'
    shutil.copy2(SIDECAR, bak)
    print(f'\nPhase 0: backed up sidecar -> {bak}')
    for idx in tqdm(range(ORPHAN_LO, ORPHAN_HI + 1), desc='Deleting orphans'):
        fp = _p(idx)
        if os.path.exists(fp):
            os.remove(fp)

    # ---- Phase 1: shift good block down by SHIFT (ascending = safe) ------- #
    # ascending so target t=s-SHIFT is always already free (orphan-deleted, or a
    # lower source already moved away).
    print(f'\nPhase 1: shifting [{GOOD_LO},{GOOD_HI}] down by {SHIFT}')
    for s in tqdm(range(GOOD_LO, GOOD_HI + 1), desc='Renaming good block'):
        src, dst = _p(s), _p(s - SHIFT)
        if os.path.exists(src):
            os.rename(src, dst)
        elif not os.path.exists(dst):
            sys.exit(f'ABORT: neither source {src} nor target {dst} exists '
                     f'(unexpected mid-state).')

    # ---- Phase 2: rebuild contiguous sidecar ------------------------------ #
    new_side = {}
    for k in range(0, ORPHAN_LO):          # [0, 398911] keep original names
        new_side[str(k)] = side[k]
    for k in range(ORPHAN_LO, NEW_MAX + 1):  # [398912, 430673] = INTERX
        new_side[str(k)] = 'INTERX'
    assert len(new_side) == EXPECTED_TOTAL, (len(new_side), EXPECTED_TOTAL)
    with open(SIDECAR, 'w') as f:
        json.dump(new_side, f)
    print(f'\nPhase 2: rewrote sidecar with {len(new_side)} contiguous entries.')

    # ---- Phase 3: restore id == index for shifted files (idempotent) ------ #
    print(f'\nPhase 3: fixing id field on shifted block [{ORPHAN_LO},{NEW_MAX}]')
    fixed = 0
    for idx in tqdm(range(ORPHAN_LO, NEW_MAX + 1), desc='Fixing ids'):
        fp = _p(idx)
        v = joblib.load(fp)
        if v.get('id') != idx:
            v['id'] = idx
            joblib.dump(v, fp)
            fixed += 1
    print(f'Phase 3: updated id on {fixed} files.')

    # ---- Phase 4: verify -------------------------------------------------- #
    disk = _disk_indices()
    side2 = {int(k): v for k, v in json.load(open(SIDECAR)).items()}
    ok = (disk == set(range(0, EXPECTED_TOTAL))
          and set(side2.keys()) == set(range(0, EXPECTED_TOTAL)))
    print('\n========================================')
    print(f'disk files   : {len(disk)}  contiguous 0..{EXPECTED_TOTAL - 1}: '
          f'{disk == set(range(0, EXPECTED_TOTAL))}')
    print(f'sidecar      : {len(side2)}  contiguous 0..{EXPECTED_TOTAL - 1}: '
          f'{set(side2.keys()) == set(range(0, EXPECTED_TOTAL))}')
    # spot-check a few shifted files: id matches, hands natural
    for idx in (ORPHAN_LO, ORPHAN_LO + 1, NEW_MAX):
        v = joblib.load(_p(idx))
        print(f'  idx {idx}: id={v.get("id")} body={v.get("body_dataset_name")} '
              f'rhand={v.get("rhand_dataset_name")} lhand={v.get("lhand_dataset_name")}')
    print(f'\nRESULT: {"OK -- set/keep TRAIN_THRESHOLD: %d" % EXPECTED_TOTAL if ok else "MISMATCH -- investigate"}')
    print('========================================')


if __name__ == '__main__':
    main()
