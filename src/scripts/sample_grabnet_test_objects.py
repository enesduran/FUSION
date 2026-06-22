"""Sample GrabNet grasps for the GRAB test objects.

For each unseen GRAB test object (binoculars, camera, fryingpan, mug,
toothpaste, wineglass) this script:
  1. Loads the object's contact mesh from GRAB's ``contact_meshes`` directory.
  2. Centers and (if needed) downscales it the same way GrabNet's reference
     ``grab_new_objects.py`` does.
  3. Samples ``--n_samples`` random rotations.
  4. Runs GrabNet (coarse + refine) on each rotated point cloud.
  5. Exports the predicted MANO right-hand mesh concatenated with the rotated
     object mesh to ``<obj>_sample<i>_hand_object.obj`` — identical layout to
     the dumps produced by ``gen_dno_app.forward_grabnet`` so the existing
     renderer (``src/scripts/render/render_grabnet.py``) can consume them.

Example:
    python src/scripts/sample_grabnet_test_objects.py \
        --out_dir fusion_runs/grabnet_test_objects \
        --n_samples 15
"""

import argparse
import os
import sys

import numpy as np
import torch
import trimesh

# GrabNet lives under external/ and isn't installed as a package.
sys.path.append(os.path.join(os.getcwd(), 'external/GrabNet'))

from bps_torch.bps import bps_torch  # noqa: E402
from grabnet.tests.tester import Tester  # noqa: E402
from grabnet.tools.cfg_parser import Config  # noqa: E402
from grabnet.tools.train_tools import point2point_signed  # noqa: E402
from grabnet.tools.utils import aa2rotmat, euler  # noqa: E402

import smplx  # noqa: E402


# Unseen objects in GrabNet's standard split (matches
# external/GrabNet/data/grabnet_data/test/frame_names.npz).
GRAB_TEST_OBJECTS = [
    'binoculars', 'camera', 'fryingpan', 'mug', 'toothpaste', 'wineglass',
]


def load_and_normalize_object(mesh_path):
    """Replicate ``load_obj_verts`` from grab_new_objects.py: load the mesh,
    auto-downscale very large objects, then center on the AABB midpoint.

    Returns ``(verts (V, 3) float32, faces (F, 3) int32)`` in the canonical
    (un-rotated) frame.
    """
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    max_radius = np.linalg.norm(verts, axis=1).max()
    if max_radius > 0.3:
        re_scale = max_radius / 0.08
        print(f"    object is large, down-scaling by {re_scale:.2f}x")
        verts = verts / re_scale

    aabb_center = (verts.max(axis=0) + verts.min(axis=0)) / 2.0
    verts = verts - aabb_center
    return verts.astype(np.float32), faces


def build_grabnet():
    """Mirror the bootstrap in gen_dno_app.py around line 895-911."""
    cfg = Config(
        default_cfg_path='external/GrabNet/grabnet/configs/grabnet_cfg.yaml',
        **{
            'work_dir': os.path.join(os.getcwd(), 'external/GrabNet/tests'),
            'best_cnet': 'external/GrabNet/grabnet/models/coarsenet.pt',
            'best_rnet': 'external/GrabNet/grabnet/models/refinenet.pt',
            'bps_dir': 'external/GrabNet/grabnet/configs/bps.npz',
        },
    )
    cfg.dataset_dir = 'external/GrabNet/data/grabnet_data'
    cfg.rhm_path = 'data/body_models/mano'

    grabnet = Tester(cfg=cfg)
    grabnet.coarse_net.eval()
    grabnet.refine_net.eval()
    return grabnet


def sample_grasps(grabnet, bps, rh_model, verts_canon, faces, n_samples,
                  device, rng):
    """Run GrabNet on ``n_samples`` random rotations of one object.

    Returns a list of length ``n_samples``; each entry is
    ``(hand_verts (778, 3), obj_verts_rot (V, 3))`` in the rotated frame.
    """
    # Random rotations (same scheme as grab_new_objects.py line 130-132).
    rand_rotdeg = rng.random((n_samples, 3)) * np.array([360.0, 360.0, 360.0])
    rand_rotmat = euler(rand_rotdeg).astype(np.float32)  # (N, 3, 3)

    # Apply each rotation to the canonical object verts: v' = R @ v.
    verts_rot = np.einsum('nij,vj->nvi', rand_rotmat, verts_canon).astype(np.float32)
    verts_object = torch.from_numpy(verts_rot).to(device)  # (N, V, 3)

    bps_object = bps.encode(verts_object, feature_type='dists')['dists']  # (N, 4096)

    with torch.no_grad():
        # Coarse net -> initial pose proposals.
        drec_cnet = grabnet.coarse_net.sample_poses(bps_object)
        rh_gen_cnet = rh_model(**drec_cnet)
        verts_rh_gen_cnet = rh_gen_cnet.vertices
        joints_rh_gen_cnet = rh_gen_cnet.joints

        # Hand-to-object signed distances drive the refinement.
        _, h2o, _ = point2point_signed(verts_rh_gen_cnet, verts_object)

        drec_cnet['trans_rhand_f'] = drec_cnet['transl']
        drec_cnet['global_orient_rhand_rotmat_f'] = aa2rotmat(
            drec_cnet['global_orient']).view(-1, 3, 3)
        drec_cnet['fpose_rhand_rotmat_f'] = aa2rotmat(
            drec_cnet['hand_pose']).view(-1, 15, 3, 3)
        drec_cnet['verts_object'] = verts_object
        drec_cnet['h2o_dist'] = h2o.abs()
        drec_cnet['joints'] = joints_rh_gen_cnet
        drec_cnet['vertices'] = verts_rh_gen_cnet

        drec_rnet = grabnet.refine_net(**drec_cnet)
        verts_rh_gen_rnet = rh_model(**drec_rnet).vertices.cpu().numpy()

    samples = []
    for i in range(n_samples):
        samples.append((verts_rh_gen_rnet[i], verts_rot[i]))
    return samples


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--out_dir', default='fusion_runs/grabnet_test_objects',
                        help='Where to write the .obj files.')
    parser.add_argument('--mesh_dir',
                        default='data/motion/Hand_Raw/GRAB/grab/tools/'
                                'object_meshes/contact_meshes',
                        help='Directory of GRAB contact_meshes .ply files.')
    parser.add_argument('--objects', nargs='+', default=GRAB_TEST_OBJECTS,
                        help='Object names (without .ply) to sample.')
    parser.add_argument('--n_samples', type=int, default=15,
                        help='Number of grasp samples per object.')
    parser.add_argument('--seed', type=int, default=42,
                        help='RNG seed for object rotations.')
    parser.add_argument('--device', default=None,
                        help='Torch device; defaults to cuda if available.')
    args = parser.parse_args()

    device = torch.device(args.device
                          or ('cuda' if torch.cuda.is_available() else 'cpu'))
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[sample_grabnet] Loading GrabNet on {device} ...")
    grabnet = build_grabnet()

    # MANO right-hand model. batch_size must match n_samples for the refine
    # net's stored ``rhm_train`` to broadcast correctly.
    rh_model = smplx.create(
        'data/body_models',
        model_type='mano',
        gender='neutral',
        num_betas=10,
        is_rhand=True,
        flat_hand_mean=True,
        use_pca=False,
        batch_size=args.n_samples,
    ).to(device)
    grabnet.refine_net.rhm_train = rh_model

    bps = bps_torch(custom_basis=grabnet.bps)

    base_rng = np.random.default_rng(args.seed)

    for obj_name in args.objects:
        mesh_path = os.path.join(args.mesh_dir, f"{obj_name}.ply")
        if not os.path.isfile(mesh_path):
            print(f"[sample_grabnet] [WARN] mesh missing: {mesh_path} — skipping")
            continue

        print(f"[sample_grabnet] === {obj_name} ({mesh_path}) ===")
        verts_canon, faces = load_and_normalize_object(mesh_path)

        # Deterministic per-object RNG so reruns reproduce the same rotations.
        obj_rng = np.random.default_rng(base_rng.integers(0, 2**31 - 1))

        samples = sample_grasps(
            grabnet=grabnet, bps=bps, rh_model=rh_model,
            verts_canon=verts_canon, faces=faces,
            n_samples=args.n_samples, device=device, rng=obj_rng,
        )

        # Each sample is one combined hand+object .obj — same layout as
        # gen_dno_app.py's ``{seq_id}_hand_object.obj`` so the renderer works.
        for i, (hand_v, obj_v) in enumerate(samples):
            hand_mesh = trimesh.Trimesh(
                vertices=hand_v, faces=rh_model.faces, process=False)
            obj_mesh = trimesh.Trimesh(
                vertices=obj_v, faces=faces.astype(np.uint32), process=False)
            combined = trimesh.util.concatenate([hand_mesh, obj_mesh])
            out_path = os.path.join(
                args.out_dir,
                f"{obj_name}_sample{i:02d}_hand_object.obj",
            )
            combined.export(out_path)
        print(f"[sample_grabnet]   saved {args.n_samples} samples to "
              f"{args.out_dir}/{obj_name}_sample*_hand_object.obj")

    print("[sample_grabnet] Done.")


if __name__ == '__main__':
    main()
