"""Detect and render every Inter-X window flagged by the arm-pose plausibility filter.

The Inter-X SMPL-X fits occasionally have severe shoulder/wrist over-rotation that
makes the elbow position look broken even though elbow flexion itself stays in
range. ``InterXProcessor._arm_pose_plausible`` flags those windows during
processing. This script reproduces the same per-window check across the dataset
and renders each flagged window to a video for visual inspection.

Two phases:
  1. Fast count pass (no rendering) to report how many windows are flagged.
  2. Render every flagged window to fusion_runs/<branch>/arm_implausible/InterX/.

Run from the repo root:
    python src/scripts/render/render_interx_arm_implausible.py
"""
import os
import sys
import time
import glob

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.scripts.process.body.process_interx_body import (
    InterXProcessor, rotate_global,
)
from src.scripts.process.config import SMPLX_JOINT_MIRROR_ARR
from src.utils.process_utils import BRANCH_NAME


print('initializing processor (loads SMPL-X x3 with num_betas=300; ~30s)...')
t0 = time.time()
proc = InterXProcessor()
print(f'  ready in {time.time()-t0:.1f}s')
W = proc.config.WINDOW

OUT_DIR = f'fusion_runs/{BRANCH_NAME}/arm_implausible/InterX'
os.makedirs(OUT_DIR, exist_ok=True)

# Iterate the same way the processor would: orig + mir, all chunks per file.
files = sorted(glob.glob(os.path.join(proc.dataset_path, 'motions', '*', 'P*.npz')))
print(f'sequences: {len(files)}  (x2 variants x ~chunks-per-clip = total windows to check)')


def iter_windows(path):
    """Yield (seq_name, variant_tag, joints, body_pose_chunk, motion_params, vertices)."""
    sid = os.path.basename(os.path.dirname(path))
    pn = os.path.splitext(os.path.basename(path))[0]
    seq_name = f'{sid}_{pn}'
    d = np.load(path, allow_pickle=True)
    gender = str(d['gender'])
    if gender not in proc.body_models:
        gender = 'neutral'
    betas = d['betas'].reshape(-1).astype(np.float32)
    if len(betas) < proc.config.NUM_BETAS:
        betas = np.concatenate([betas, np.zeros(proc.config.NUM_BETAS - len(betas), dtype=np.float32)])
    sf = proc.sample_freq
    ro = d['root_orient'][::sf].astype(np.float32)
    tr = d['trans'][::sf].astype(np.float32)
    pb = d['pose_body'][::sf].reshape(-1, 63).astype(np.float32)
    plh = d['pose_lhand'][::sf].reshape(-1, 45).astype(np.float32)
    prh = d['pose_rhand'][::sf].reshape(-1, 45).astype(np.float32)
    T = len(ro)
    if T < W:
        return
    pj = np.zeros((T, 3), np.float32); pe = np.zeros((T, 6), np.float32)
    rest = proc.body_models[gender](
        betas=torch.from_numpy(betas[None]).float().to(proc.device)
    ).joints.detach().cpu().numpy()[0, :22]
    root_offset = rest[0]
    ro, tr = rotate_global(ro, tr, root_offset)
    for augment in (False, True):
        if augment:
            fp = np.hstack([ro, pb, pj, pe, plh, prh]).reshape(-1, len(SMPLX_JOINT_MIRROR_ARR), 3)
            fp = fp[:, SMPLX_JOINT_MIRROR_ARR]; fp[..., 1:] *= -1
            fp = fp.reshape(-1, len(SMPLX_JOINT_MIRROR_ARR) * 3)
            ro_, pb_, pj_, pe_, plh_, prh_ = (
                fp[..., :3], fp[..., 3:66], fp[..., 66:69],
                fp[..., 69:75], fp[..., 75:120], fp[..., 120:],
            )
            tr_ = tr.copy(); tr_[:, 0] *= -1
        else:
            ro_, pb_, pj_, pe_, plh_, prh_, tr_ = ro, pb, pj, pe, plh, prh, tr
        for s, e in proc._chunk_indices(T):
            sl = slice(s, e)
            mp = {
                'betas': torch.from_numpy(betas[None]).float().to(proc.device),
                'global_orient': torch.from_numpy(ro_[sl]).float().to(proc.device),
                'body_pose': torch.from_numpy(pb_[sl]).float().to(proc.device),
                'left_hand_pose': torch.from_numpy(plh_[sl]).float().to(proc.device),
                'right_hand_pose': torch.from_numpy(prh_[sl]).float().to(proc.device),
                'jaw_pose': torch.from_numpy(pj_[sl]).float().to(proc.device),
                'leye_pose': torch.from_numpy(pe_[sl][:, :3]).float().to(proc.device),
                'reye_pose': torch.from_numpy(pe_[sl][:, 3:]).float().to(proc.device),
                'transl': torch.from_numpy(tr_[sl]).float().to(proc.device),
                'expression': torch.zeros((W, 10)).float().to(proc.device),
            }
            out = proc.body_models[gender](**mp)
            joints = out.joints.detach().cpu().numpy()[:, :22]
            tag = f"{'mir' if augment else 'orig'}_{s:04d}"
            yield seq_name, tag, joints, pb_[sl], mp, out.vertices.detach().cpu().numpy()


# --- Phase 1: count only (no rendering) ---
print('\n[phase 1] counting flagged windows (no rendering)...')
n_total = 0
n_flagged = 0
flagged_list = []  # store metadata for phase 2 (avoid re-computing)
t1 = time.time()
for path in tqdm(files, desc='Phase 1 count'):
    for seq_name, tag, joints, pb_sl, mp, verts in iter_windows(path):
        n_total += 1
        ok, reason = proc._arm_pose_plausible(joints, pb_sl)
        if not ok:
            n_flagged += 1
            flagged_list.append((seq_name, tag, reason, mp, verts, joints))
print(f'phase 1 done in {time.time()-t1:.0f}s: '
      f'{n_flagged}/{n_total} flagged ({100*n_flagged/max(n_total,1):.1f}%)')

# --- Phase 2: render ---
print(f'\n[phase 2] rendering {n_flagged} videos to {OUT_DIR}/ ...')
t2 = time.time()
for seq_name, tag, reason, mp, verts, joints in tqdm(flagged_list, desc='Phase 2 render'):
    fname = f'{OUT_DIR}/{seq_name}_{tag}_{reason}'
    mesh_dict = {
        'transl': mp['transl'].cpu(),
        'global_orient': mp['global_orient'].cpu(),
        'faces': proc.body_models['neutral'].faces,
        'vertices': verts,
    }
    camera_dict = {
        'camera_rot': mp['global_orient'].cpu(),
        'camera_transl': mp['transl'].cpu(),
        'coef': 2.0,
    }
    skeleton_dict = {
        'positions': joints,
        'contact_masks': np.zeros((W, 4), dtype=bool),
        'color': (160 / 255, 16 / 255, 16 / 255, 0.9),
    }
    proc.renderer.render_motion(
        mesh_dict, fname,
        skeleton_dict=skeleton_dict, camera_dict=camera_dict,
        color=(220 / 255, 70 / 255, 70 / 255, 1),
    )
print(f'\nphase 2 done in {time.time()-t2:.0f}s. all renders in: {OUT_DIR}')
print(f'TOTAL: {n_flagged}/{n_total} flagged, {time.time()-t0:.0f}s')
