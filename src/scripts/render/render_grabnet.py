"""Render GrabNet hand-object grasps as static frames.

GrabNet outputs are saved in ``gen_dno_app.forward_grabnet`` as a single
``.obj`` file per sequence (see ``src/optim/gen_dno_app.py`` around the
``{seq_id}_hand_object.obj`` export). The file is the concatenation of:

    [MANO right-hand mesh  (778 verts, 1538 faces)] + [object mesh]

This script loads such files, splits the hand from the object, colors them
distinctly, and renders one or more pyrender views to PNG.

Examples
--------
    # Single file -> one PNG next to it
    python src/scripts/render/render_grabnet.py \
        --input fusion_runs/<run>/vis/000033_hand_object.obj

    # Directory of grasps -> PNG per file in --output
    python src/scripts/render/render_grabnet.py \
        --input fusion_runs/<run>/vis \
        --output fusion_runs/<run>/vis_renders

    # Turntable: 4 views (front, right, back, left)
    python src/scripts/render/render_grabnet.py \
        --input fusion_runs/<run>/vis \
        --num_views 4
"""

import argparse
import glob
import os
import sys

import numpy as np
import trimesh
from PIL import Image

# pyrender needs an offscreen backend on the cluster.
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import pyrender  # noqa: E402  (must come after env var)


# MANO right-hand constants (matches smplx.create(model_type='mano', is_rhand=True)).
MANO_NUM_VERTS = 778
MANO_NUM_FACES = 1538
# Fingertip vertex indices on the MANO right-hand mesh (gen_dno_app.py:313).
MANO_TIP_IDS = [744, 320, 443, 554, 671]  # thumb, index, middle, ring, pinky

HAND_COLOR = np.array([106, 168, 79, 255], dtype=np.uint8)   # green, matches render_smplx_mano_smplx.py
OBJECT_COLOR = np.array([61, 133, 198, 255], dtype=np.uint8)   # blue, matches render_smplx_mano_smplx.py


def split_hand_object(mesh: trimesh.Trimesh,
                      num_hand_verts: int = MANO_NUM_VERTS,
                      num_hand_faces: int = MANO_NUM_FACES):
    """Split a concatenated hand+object trimesh into two trimeshes.

    The concatenation in ``gen_dno_app.py`` preserves vertex/face ordering: the
    hand vertices and faces come first, then the object's. Faces in the object
    block reference vertex indices offset by ``num_hand_verts``.
    """
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    if verts.shape[0] < num_hand_verts or faces.shape[0] < num_hand_faces:
        raise ValueError(
            f"Mesh has {verts.shape[0]} verts / {faces.shape[0]} faces; "
            f"expected at least MANO right hand ({num_hand_verts} verts, "
            f"{num_hand_faces} faces). Is this a GrabNet hand+object dump?"
        )

    hand_verts = verts[:num_hand_verts]
    hand_faces = faces[:num_hand_faces]

    obj_faces_raw = faces[num_hand_faces:]
    if obj_faces_raw.size == 0:
        obj_mesh = None
    else:
        # Faces in the object block index into the global vertex list, so the
        # smallest index gives us the start of the object's vertex block.
        obj_v_start = int(obj_faces_raw.min())
        obj_verts = verts[obj_v_start:]
        obj_faces = obj_faces_raw - obj_v_start
        obj_mesh = trimesh.Trimesh(
            vertices=obj_verts, faces=obj_faces,
            vertex_colors=np.broadcast_to(OBJECT_COLOR, (obj_verts.shape[0], 4)).copy(),
            process=False,
        )

    hand_mesh = trimesh.Trimesh(
        vertices=hand_verts, faces=hand_faces,
        vertex_colors=np.broadcast_to(HAND_COLOR, (hand_verts.shape[0], 4)).copy(),
        process=False,
    )
    return hand_mesh, obj_mesh


def look_at(eye, target, up=(0.0, 1.0, 0.0)):
    """OpenGL-style look-at: returns a 4x4 pose for a camera that looks from
    ``eye`` toward ``target``. pyrender cameras look down -Z in their local
    frame, so we build the rotation accordingly."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    forward = target - eye
    forward /= np.linalg.norm(forward) + 1e-12
    right = np.cross(forward, up)
    rn = np.linalg.norm(right)
    if rn < 1e-6:
        # Eye and up are colinear; pick an arbitrary right.
        right = np.array([1.0, 0.0, 0.0])
    else:
        right /= rn
    true_up = np.cross(right, forward)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward  # camera looks down -Z
    pose[:3, 3] = eye
    return pose


def _rotate_around(axis, vector, angle_rad):
    """Rodrigues rotation of ``vector`` around unit ``axis`` by ``angle_rad``."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return (vector * c + np.cross(axis, vector) * s
            + axis * np.dot(axis, vector) * (1.0 - c))


def hand_aware_basis(hand_verts, obj_verts):
    """Compute a hand-relative orthonormal basis (palm, finger, thumb) so the
    camera can be placed to actually show the fingers.

    Returns a dict with unit vectors:
        palm_normal: direction the palm faces (~ hand → object during a grasp).
        finger_dir : wrist → fingertips, made orthogonal to palm_normal.
        thumb_dir  : palm_normal x finger_dir; for a right hand this points
                     toward the thumb side.
        grasp_center, radius: framing target and bounding-sphere radius.
    """
    hand_centroid = hand_verts.mean(axis=0)
    fingertip_centroid = hand_verts[MANO_TIP_IDS].mean(axis=0)

    if obj_verts is not None and obj_verts.shape[0] > 0:
        obj_centroid = obj_verts.mean(axis=0)
        palm_normal = obj_centroid - hand_centroid
    else:
        # No object: assume the palm faces toward where the fingers curl,
        # which is roughly the direction from the wrist to the fingertips.
        obj_centroid = fingertip_centroid
        palm_normal = fingertip_centroid - hand_centroid

    pn_norm = np.linalg.norm(palm_normal)
    palm_normal = palm_normal / pn_norm if pn_norm > 1e-6 else np.array([0.0, 0.0, 1.0])

    finger_dir = fingertip_centroid - hand_centroid
    finger_dir = finger_dir - np.dot(finger_dir, palm_normal) * palm_normal
    fd_norm = np.linalg.norm(finger_dir)
    if fd_norm < 1e-6:
        # Pick any vector perpendicular to palm_normal.
        helper = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(helper, palm_normal)) > 0.95:
            helper = np.array([0.0, 1.0, 0.0])
        finger_dir = helper - np.dot(helper, palm_normal) * palm_normal
        finger_dir /= np.linalg.norm(finger_dir)
    else:
        finger_dir /= fd_norm

    thumb_dir = np.cross(palm_normal, finger_dir)
    thumb_dir /= max(np.linalg.norm(thumb_dir), 1e-6)

    grasp_center = 0.5 * (hand_centroid + obj_centroid)
    all_verts = (hand_verts if obj_verts is None
                 else np.vstack([hand_verts, obj_verts]))
    radius = float(np.linalg.norm(all_verts - grasp_center, axis=1).max())
    radius = max(radius, 0.05)

    return {
        'palm_normal': palm_normal,
        'finger_dir': finger_dir,
        'thumb_dir': thumb_dir,
        'grasp_center': grasp_center,
        'radius': radius,
    }


def grasp_camera_pose(basis, yfov, azimuth_deg=0.0, elevation_deg=10.0,
                      roll_deg=0.0, padding=1.4):
    """Camera pose that views the grasp from the thumb side by default.

    ``azimuth_deg`` rotates the camera around the palm-normal axis (0° = thumb
    side, 90° = looking toward fingertips, 180° = pinky side, 270° = wrist).
    ``elevation_deg`` tilts the camera toward the palm side so we look slightly
    onto the palm rather than purely edge-on.
    ``roll_deg`` rotates the camera around its own viewing axis: 0° = fingers
    up in the image; 180° = wrist up / fingers down; 90° / 270° = fingers
    horizontal.
    """
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    roll = np.deg2rad(roll_deg)

    # Start on the thumb side; rotate around the palm normal for turntable
    # views around the grasp.
    view_dir = _rotate_around(basis['palm_normal'], basis['thumb_dir'], az)
    # Tilt toward the palm (positive elevation -> look at palm side of hand).
    view_dir = np.cos(el) * view_dir + np.sin(el) * basis['palm_normal']
    view_dir /= np.linalg.norm(view_dir)

    distance = padding * basis['radius'] / np.tan(yfov / 2.0)
    eye = basis['grasp_center'] + view_dir * distance

    # Default up = finger_dir so fingertips point up in the image, but at
    # |az| near 90° view_dir becomes parallel to finger_dir and the look-at
    # frame degenerates. Fall back to palm_normal as up when that happens.
    up = basis['finger_dir']
    if abs(np.dot(view_dir, up)) > 0.95:
        up = basis['palm_normal']
    # Apply camera roll around the viewing axis. ``up`` only needs to be
    # non-parallel to ``view_dir``; look_at re-orthogonalises it, and rotating
    # around view_dir keeps that property intact.
    if abs(roll) > 1e-6:
        up = _rotate_around(view_dir, up, roll)

    return look_at(eye, basis['grasp_center'], up=up)


def build_scene(hand_mesh, obj_mesh, bg_color=(255, 255, 255, 255)):
    scene = pyrender.Scene(
        bg_color=np.asarray(bg_color, dtype=np.float32) / 255.0,
        ambient_light=np.array([0.25, 0.25, 0.25]),
    )

    hand_material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.05, roughnessFactor=0.7,
        baseColorFactor=HAND_COLOR.astype(np.float32) / 255.0,
        alphaMode='OPAQUE',
    )
    scene.add(pyrender.Mesh.from_trimesh(hand_mesh, material=hand_material, smooth=True))

    if obj_mesh is not None:
        obj_material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0, roughnessFactor=0.85,
            baseColorFactor=OBJECT_COLOR.astype(np.float32) / 255.0,
            alphaMode='OPAQUE',
        )
        scene.add(pyrender.Mesh.from_trimesh(obj_mesh, material=obj_material, smooth=True))

    return scene


def save_png_and_pdf(img_array, out_path):
    """Save the image to ``out_path`` (PNG) plus a sibling .pdf.

    PIL writes single-page PDFs from RGB arrays; the PDF goes next to the PNG
    with the same stem. PDFs need flat RGB (no alpha), so we drop alpha if
    present.
    """
    img = Image.fromarray(img_array)
    img.save(out_path)

    pdf_path = os.path.splitext(out_path)[0] + '.pdf'
    pdf_img = img.convert('RGB') if img.mode == 'RGBA' else img
    pdf_img.save(pdf_path, 'PDF', resolution=200.0)
    return [out_path, pdf_path]


def add_three_point_lighting(scene, cam_pose, intensity_scale=1.0):
    """Add a key/fill/back light triad relative to the camera frame."""
    key = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0 * intensity_scale)
    scene.add(key, pose=cam_pose)

    fill_pose = cam_pose.copy()
    fill_pose[:3, 3] = cam_pose[:3, 3] + cam_pose[:3, 0] * 0.5  # shift along camera right
    fill = pyrender.DirectionalLight(color=np.ones(3), intensity=1.5 * intensity_scale)
    scene.add(fill, pose=fill_pose)

    back_pose = cam_pose.copy()
    back_pose[:3, 3] = cam_pose[:3, 3] - cam_pose[:3, 2] * 2.0  # behind subject
    back = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0 * intensity_scale)
    scene.add(back, pose=back_pose)


def render_grasp(obj_path, out_path, width=800, height=800,
                 num_azimuths=1, num_elevations=1, num_rolls=1,
                 yfov_deg=35.0,
                 elevation_min_deg=-30.0, elevation_max_deg=50.0,
                 azimuth_min_deg=-80.0, azimuth_max_deg=80.0,
                 num_hand_verts=MANO_NUM_VERTS, num_hand_faces=MANO_NUM_FACES):
    mesh = trimesh.load(obj_path, process=False, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        # ``trimesh.load`` may return a Scene if the .obj has groups; merge it.
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    hand_mesh, obj_mesh = split_hand_object(mesh, num_hand_verts, num_hand_faces)

    # Build a hand-relative camera basis so the default view shows the fingers
    # wrapping the object (thumb side) instead of the back of the wrist.
    basis = hand_aware_basis(
        np.asarray(hand_mesh.vertices),
        np.asarray(obj_mesh.vertices) if obj_mesh is not None else None,
    )

    yfov = np.deg2rad(yfov_deg)
    aspect = width / height
    renderer = pyrender.OffscreenRenderer(width, height)

    num_azimuths = max(num_azimuths, 1)
    num_elevations = max(num_elevations, 1)
    num_rolls = max(num_rolls, 1)
    if num_elevations == 1:
        # Single-row mode: use the midpoint between min and max as the elevation,
        # so callers can stick to ``--num_azimuths`` without thinking about el.
        elevations = [0.5 * (elevation_min_deg + elevation_max_deg)]
    else:
        elevations = np.linspace(elevation_max_deg, elevation_min_deg, num_elevations).tolist()

    # Azimuths are restricted to a half-circle around the palm-normal axis
    # so the right hand always lands on the LEFT side of the image. Hand-on-
    # left iff cos(az) > 0 (i.e. |az| < 90°); the user-chosen range should
    # stay strictly inside (-90°, 90°).
    if num_azimuths == 1:
        azimuths = [0.5 * (azimuth_min_deg + azimuth_max_deg)]
    else:
        azimuths = np.linspace(azimuth_min_deg, azimuth_max_deg, num_azimuths).tolist()

    # Rolls iterate over camera-roll around the viewing axis. roll=0 puts
    # fingertips up; roll=180 puts the wrist up. With num_rolls > 1 we sample
    # the full 360° circle (e.g. 2 -> [0, 180], 4 -> [0, 90, 180, 270]).
    if num_rolls == 1:
        rolls = [0.0]
    else:
        rolls = np.linspace(0.0, 360.0, num_rolls, endpoint=False).tolist()

    # One grid per roll: rows index elevation (top = highest = looking down on
    # palm), cols index azimuth (col 0 = thumb-most side).
    per_roll_grids = []
    try:
        for roll_deg in rolls:
            grid = []
            for el_deg in elevations:
                row = []
                for azimuth in azimuths:
                    cam_pose = grasp_camera_pose(
                        basis, yfov,
                        azimuth_deg=azimuth,
                        elevation_deg=el_deg,
                        roll_deg=roll_deg,
                    )
                    scene = build_scene(hand_mesh, obj_mesh)
                    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=aspect)
                    scene.add(camera, pose=cam_pose)
                    add_three_point_lighting(scene, cam_pose)
                    color, _ = renderer.render(scene)
                    row.append(color)
                grid.append(row)
            per_roll_grids.append(grid)
    finally:
        renderer.delete()

    if num_azimuths == 1 and num_elevations == 1 and num_rolls == 1:
        return save_png_and_pdf(per_roll_grids[0][0][0], out_path)

    base, ext = os.path.splitext(out_path)
    saved = []
    for rr, grid in enumerate(per_roll_grids):
        for r, row in enumerate(grid):
            for c, img in enumerate(row):
                parts = []
                if num_rolls > 1:
                    parts.append(f"roll{rr}")
                if num_elevations > 1:
                    parts.append(f"el{r}")
                if num_azimuths > 1:
                    parts.append(f"view{c}")
                tag = "_" + "_".join(parts) if parts else ""
                saved.extend(save_png_and_pdf(img, f"{base}{tag}{ext}"))

    # Composite per roll: stack rows horizontally, then rows vertically.
    sheet_tag = '_strip' if num_elevations == 1 else '_grid'
    for rr, grid in enumerate(per_roll_grids):
        rows_concat = [np.concatenate(row, axis=1) for row in grid]
        sheet = rows_concat[0] if len(rows_concat) == 1 else np.concatenate(rows_concat, axis=0)
        roll_tag = f"_roll{rr}" if num_rolls > 1 else ""
        saved.extend(save_png_and_pdf(sheet, f"{base}{roll_tag}{sheet_tag}{ext}"))
    return saved


def collect_inputs(input_path):
    if os.path.isdir(input_path):
        files = sorted(glob.glob(os.path.join(input_path, "*_hand_object.obj")))
        if not files:
            files = sorted(glob.glob(os.path.join(input_path, "*.obj")))
        return files
    if any(ch in input_path for ch in "*?["):
        return sorted(glob.glob(input_path))
    return [input_path]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', required=True,
                        help='Path to a GrabNet hand+object .obj file, a directory '
                             'of such files, or a glob.')
    parser.add_argument('--output', default=None,
                        help='Output directory (one PNG per input) or single PNG '
                             'path (only valid when --input is a single file). '
                             'Defaults to alongside each input with a .png suffix.')
    parser.add_argument('--width', type=int, default=800)
    parser.add_argument('--height', type=int, default=800)
    parser.add_argument('--num_azimuths', '--num_views', type=int, default=1,
                        dest='num_azimuths',
                        help='Number of azimuth views rotating around the palm '
                             'normal (>=1). Alias: --num_views.')
    parser.add_argument('--num_elevations', type=int, default=1,
                        help='Number of elevation tilts from the palm side '
                             'toward the back of the hand (>=1). With >1 the '
                             'output is a 2D grid (rows=elevations, cols=azimuths).')
    parser.add_argument('--num_rolls', type=int, default=1,
                        help='Number of camera-roll samples around the viewing '
                             'axis. 1 = fingertips up only; 2 = adds wrist-up '
                             '(180° roll); 4 = adds the two horizontal rolls.')
    parser.add_argument('--elevation_min_deg', type=float, default=-30.0,
                        help='Lowest elevation (degrees toward back of hand).')
    parser.add_argument('--elevation_max_deg', type=float, default=50.0,
                        help='Highest elevation (degrees toward palm side).')
    parser.add_argument('--azimuth_min_deg', type=float, default=-80.0,
                        help='Minimum azimuth around the palm-normal axis. Stay '
                             'in (-90, 90) so the right hand remains on the left '
                             'of the image (|az|<90 enforces hand-on-left).')
    parser.add_argument('--azimuth_max_deg', type=float, default=80.0,
                        help='Maximum azimuth around the palm-normal axis. See '
                             '--azimuth_min_deg for the hand-on-left constraint.')
    parser.add_argument('--yfov_deg', type=float, default=35.0)
    parser.add_argument('--num_hand_verts', type=int, default=MANO_NUM_VERTS)
    parser.add_argument('--num_hand_faces', type=int, default=MANO_NUM_FACES)
    args = parser.parse_args()

    inputs = collect_inputs(args.input)
    if not inputs:
        print(f"[render_grabnet] No .obj files found under: {args.input}",
              file=sys.stderr)
        sys.exit(1)

    # Resolve output destination semantics.
    single_file_input = len(inputs) == 1 and os.path.isfile(inputs[0]) and \
        not os.path.isdir(args.input)
    out_dir = None
    out_file = None
    if args.output is None:
        pass  # write next to each input
    elif single_file_input and args.output.lower().endswith('.png'):
        out_file = args.output
        os.makedirs(os.path.dirname(out_file) or '.', exist_ok=True)
    else:
        out_dir = args.output
        os.makedirs(out_dir, exist_ok=True)

    total_views = args.num_azimuths * args.num_elevations * args.num_rolls
    print(f"[render_grabnet] Rendering {len(inputs)} file(s) "
          f"({args.num_azimuths} azimuth(s) x {args.num_elevations} elevation(s) "
          f"x {args.num_rolls} roll(s) = {total_views} view(s) each)")

    for idx, obj_path in enumerate(inputs):
        if out_file is not None:
            target = out_file
        elif out_dir is not None:
            target = os.path.join(
                out_dir, os.path.splitext(os.path.basename(obj_path))[0] + '.png'
            )
        else:
            target = os.path.splitext(obj_path)[0] + '.png'

        try:
            saved = render_grasp(
                obj_path=obj_path, out_path=target,
                width=args.width, height=args.height,
                num_azimuths=args.num_azimuths,
                num_elevations=args.num_elevations,
                num_rolls=args.num_rolls,
                yfov_deg=args.yfov_deg,
                elevation_min_deg=args.elevation_min_deg,
                elevation_max_deg=args.elevation_max_deg,
                azimuth_min_deg=args.azimuth_min_deg,
                azimuth_max_deg=args.azimuth_max_deg,
                num_hand_verts=args.num_hand_verts,
                num_hand_faces=args.num_hand_faces,
            )
        except Exception as e:
            print(f"[render_grabnet] ({idx + 1}/{len(inputs)}) FAILED "
                  f"{obj_path}: {e}", file=sys.stderr)
            continue

        print(f"[render_grabnet] ({idx + 1}/{len(inputs)}) {obj_path} -> "
              f"{', '.join(saved)}")


if __name__ == '__main__':
    main()
