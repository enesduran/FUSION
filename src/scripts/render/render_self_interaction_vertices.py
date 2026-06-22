import os
import sys
import smplx
import torch
import pickle
import trimesh
import pyrender
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from utils.llm_utils import smplx_vertex_ids

def visualize_sampled_vertices(model, vertex_ids, output_dir="fusion_runs/renders/sampled_vertices", 
                               create_hand_views=False,
                              sphere_radius=0.015):
    """
    Visualize sampled vertices on SMPLX T-pose mesh
    
    Args:
        model: SMPLX model instance
        vertex_ids: List/array of vertex indices to visualize
        output_dir: Directory to save outputs
        sphere_radius: Radius of sphere markers    
    Returns:
        dict: Paths to generated files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get T-pose mesh (template with zero parameters)
    batch_size = 1
    zero_params = {
        'global_orient': torch.zeros([batch_size, 3]),
        'body_pose': torch.zeros([batch_size, 21 * 3]),
        'betas': torch.zeros([batch_size, 16]),
        'expression': torch.zeros([batch_size, 10]),
        'jaw_pose': torch.zeros([batch_size, 3]),
        'reye_pose': torch.zeros([batch_size, 3]),
        'leye_pose': torch.zeros([batch_size, 3]),
        'left_hand_pose': torch.zeros([batch_size, 15 * 3]),
        'right_hand_pose': torch.zeros([batch_size, 15 * 3])
    }
    
    # Generate T-pose mesh
    output = model(**zero_params)
    vertices = output.vertices[0].detach().cpu().numpy()
    faces = model.faces
    joints = output.joints[0].detach().cpu().numpy()
    
    # Create base mesh
    base_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    base_mesh.visual.vertex_colors = BODY_COLOR 

    # Get sampled vertex positions
    sampled_positions = vertices[vertex_ids]
    
    # Create sphere markers for sampled vertices
    sphere_meshes = []
    for _i_, pos in enumerate(sampled_positions):
        sphere = trimesh.creation.icosphere(radius=sphere_radius)
        sphere.vertices += pos
        sphere_meshes.append(sphere)
    
    # 1. Save OBJ file with base mesh and spheres
    obj_path = os.path.join(output_dir, "mesh_with_sampled_vertices.obj")
    combined_mesh = base_mesh.copy()
    
    # Combine all sphere meshes with base mesh
    for sphere in sphere_meshes:
        combined_mesh = combined_mesh + sphere
    
    combined_mesh.export(obj_path)
    
    # 2. Create rendering
    create_body_focused_renders(base_mesh, sphere_meshes, joints, output_dir, sphere_radius)

    if create_hand_views:
        create_hand_focused_renders(base_mesh, sphere_meshes, joints, output_dir, sphere_radius)
    
    # 3. Save vertex information
    info_path = os.path.join(output_dir, "vertex_info.txt")
    save_vertex_info(vertex_ids, sampled_positions, info_path)
    
    return {
        'obj_file': obj_path,
        'render': output_dir,
        'vertex_info': info_path,
        'vertex_count': len(vertex_ids)
    }

def create_body_part_camera_pose(part_position, view_type='palm', hand_side='left'):
    """Create camera pose for hand-focused view"""
    
    # Base distance from hand
    hand_distance = 0.4
    head_distance = 0.6
    eps = 1e-3
    
    # Adjust camera position based on hand side and view type
    if hand_side == 'left':
        if view_type == 'palm':
            # View left palm (from front-right)
            cam_pos = part_position + np.array([0.09+eps, -hand_distance, 0.0])
            look_at = part_position + np.array([0.09, 0.0, 0.0])  # Look at fingers
        else:  # back view
            # View left hand back (from back-right)
            cam_pos = part_position + np.array([0.09-eps, hand_distance, 0.0])
            look_at = part_position + np.array([0.09, 0.0, 0.0])
    elif hand_side == 'right':  # right hand
        if view_type == 'palm':
            # View right palm (from front-left)
            cam_pos = part_position + np.array([-0.09-eps, -hand_distance, 0.0])
            look_at = part_position + np.array([-0.09, 0.0, 0.0])
        else:  # back view
            # View right hand back (from back-left)
            cam_pos = part_position + np.array([-0.09+eps, hand_distance, 0.0])
            look_at = part_position + np.array([-0.09, 0.0, 0.0])

    else:
        if view_type == 'front':
            cam_pos = part_position + np.array([-0.01, 0.05, head_distance])
            look_at = part_position + np.array([-0.01, 0.05, 0.0])
        else:
            cam_pos = part_position + np.array([-0.01, 0.05, -head_distance])
            look_at = part_position + np.array([-0.01, 0.05, 0.0])


    
    # Create look-at matrix
    up = np.array([0.0, 1.0, 0.0])
    forward = look_at - cam_pos
    forward = forward / np.linalg.norm(forward)
    
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    
    up = np.cross(right, forward)
    
    # Create camera pose matrix
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward  # OpenGL convention
    pose[:3, 3] = cam_pos

    return pose


def render_body_view(base_mesh, sphere_meshes, cam_pose, out_path, joints_3d, sphere_radius, 
                     image_size=(1600, 1600)):
    
    # Create scene
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], 
                          ambient_light=[0.3, 0.3, 0.3])
    
    # Add base mesh with light gray color
    base_mesh_colored = base_mesh.copy()

    render_mesh = pyrender.Mesh.from_trimesh(base_mesh_colored, smooth=True)
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.9,
        roughnessFactor=0.9,
        baseColorFactor=[0.8, 0.8, 0.8, 1.0]
    )
    render_mesh.material = material
    scene.add(render_mesh)
    
    # Add sphere markers
    for sphere in sphere_meshes:

        sphere.visual.vertex_colors = SPHERE_COLOR

        sphere_render = pyrender.Mesh.from_trimesh(sphere)
        sphere_material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            roughnessFactor=0.3, 
            baseColorFactor=[1.0, 0.2, 0.2, 1.0]
            # baseColorFactor=SPHERE_COLOR  
        )
        sphere_render.material = sphere_material
        scene.add(sphere_render)
    
    # Add joint spheres (optional, smaller and white)
    if joints_3d is not None and SHOW_JOINTS:
        joint_radius = 0.008
        for joint_pos in joints_3d:
            joint_sphere = trimesh.creation.icosphere(radius=joint_radius)
            joint_sphere.vertices += joint_pos
            joint_sphere.visual.vertex_colors = [255, 255, 255, 180]  # Semi-transparent white
            joint_mesh = pyrender.Mesh.from_trimesh(joint_sphere)
            scene.add(joint_mesh)
    
    # Camera setup
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.8, aspectRatio=1)
    
    scene.add(camera, pose=cam_pose)
    
    # Lighting setup
    key_light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(key_light, pose=cam_pose)
    
    fill_pose = cam_pose.copy()
    fill_pose[0, 3] = 1.0
    fill_light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.5)
    scene.add(fill_light, pose=fill_pose)
    
    back_pose = cam_pose.copy()
    back_pose[2, 3] = -2.0
    back_light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
    scene.add(back_light, pose=back_pose)
    
    # Render
    r = pyrender.OffscreenRenderer(*image_size)
    
    color, depth = r.render(scene)
    Image.fromarray(color).save(out_path)
 
    r.delete()


def render_hand_view(base_mesh, sphere_meshes, cam_pose, out_path, sphere_radius, 
                    image_size=(800, 800)):
    """Render a focused hand view"""
    
    # Create scene
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], 
                          ambient_light=[0.3, 0.3, 0.3])
    
    # Add base mesh with light gray color
    base_mesh_colored = base_mesh.copy()

    render_mesh = pyrender.Mesh.from_trimesh(base_mesh_colored, smooth=True)
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.9,
        roughnessFactor=0.9,
        baseColorFactor=[0.8, 0.8, 0.8, 1.0]
    )
    render_mesh.material = material
    scene.add(render_mesh)
    
    # Add sphere markers with slightly larger size for hand views
    hand_sphere_radius = sphere_radius * 0.3

    for sphere in sphere_meshes:
        # Create larger sphere for hand view
        hand_sphere = trimesh.creation.icosphere(radius=hand_sphere_radius)
        hand_sphere.vertices += sphere.vertices.mean(axis=0)  # Center position
        hand_sphere.visual.vertex_colors = SPHERE_COLOR # Bright red
        
        sphere_render = pyrender.Mesh.from_trimesh(hand_sphere)
        sphere_material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.1,
            roughnessFactor=0.3,
            baseColorFactor=[1.0, 0.2, 0.2, 1.0]
        )
        sphere_render.material = sphere_material
        scene.add(sphere_render)
    
    # Camera setup with narrower field of view for close-up
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 6, aspectRatio=1.0)  # Narrower FOV
    scene.add(camera, pose=cam_pose)
    
    # Enhanced lighting for hand details
    # Key light from camera
    key_light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
    scene.add(key_light, pose=cam_pose)
    
    # Fill light from side
    fill_pose = cam_pose.copy()
    fill_pose[:3, 3] += cam_pose[:3, 0] * 0.1  # Offset along right vector
    fill_light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(fill_light, pose=fill_pose)
    
    # Top light
    top_pose = cam_pose.copy()
    top_pose[:3, 3] += cam_pose[:3, 1] * 0.1  # Offset along up vector
    top_light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.5)
    scene.add(top_light, pose=top_pose)
    
    # Render
    r = pyrender.OffscreenRenderer(*image_size)
    color, depth = r.render(scene)
    Image.fromarray(color).save(out_path)
    r.delete()


def create_hand_focused_renders(base_mesh, sphere_meshes, joints, output_dir, sphere_radius):
    """Create focused renderings of left and right hands (palm and back views)"""
    
    # SMPLX joint indices for hands
    LEFT_WRIST_IDX = 20   # Left wrist joint
    RIGHT_WRIST_IDX = 21  # Right wrist joint
    HEAD_IDX = 15
    
    # Get hand joint positions for camera positioning
    left_wrist = joints[LEFT_WRIST_IDX]
    right_wrist = joints[RIGHT_WRIST_IDX]

    head_loc = joints[HEAD_IDX]
    
    part_renders = {}

    part_renders['head_front'] = os.path.join(output_dir, f"head_front.{extension}")
    part_renders['head_back'] = os.path.join(output_dir, f"head_back.{extension}")
     
    # Left hand views
    part_renders['left_hand_palm'] = os.path.join(output_dir, f"left_hand_palm.{extension}")
    part_renders['left_hand_back'] = os.path.join(output_dir, f"left_hand_back.{extension}")
    
    # Right hand views  
    part_renders['right_hand_palm'] = os.path.join(output_dir, f"right_hand_palm.{extension}")
    part_renders['right_hand_back'] = os.path.join(output_dir, f"right_hand_back.{extension}")
    
    # Render left hand palm (front view)
    head_front_pose = create_body_part_camera_pose(head_loc, view_type='front', hand_side='head')
    render_hand_view(base_mesh, sphere_meshes, head_front_pose, 
                    part_renders['head_front'], sphere_radius)
    
    head_back_pose = create_body_part_camera_pose(head_loc, view_type='back', hand_side='head')
    render_hand_view(base_mesh, sphere_meshes, head_back_pose, 
                    part_renders['head_back'], sphere_radius)
    
    # Render left hand palm (front view)
    left_palm_pose = create_body_part_camera_pose(left_wrist, view_type='palm', hand_side='left')
    render_hand_view(base_mesh, sphere_meshes, left_palm_pose, 
                    part_renders['left_hand_palm'], sphere_radius)
    
    # Render left hand back
    left_back_pose = create_body_part_camera_pose(left_wrist, view_type='back', hand_side='left')
    render_hand_view(base_mesh, sphere_meshes, left_back_pose, 
                    part_renders['left_hand_back'], sphere_radius)
    
    # Render right hand palm
    right_palm_pose = create_body_part_camera_pose(right_wrist, view_type='palm', hand_side='right')
    render_hand_view(base_mesh, sphere_meshes, right_palm_pose, 
                    part_renders['right_hand_palm'], sphere_radius)
    
    # Render right hand back
    right_back_pose = create_body_part_camera_pose(right_wrist, view_type='back', hand_side='right')
    render_hand_view(base_mesh, sphere_meshes, right_back_pose, 
                    part_renders['right_hand_back'], sphere_radius)
    
    return part_renders

def create_body_focused_renders(base_mesh, sphere_meshes, joints, output_dir, sphere_radius):
    """Render mesh with sphere markers"""
    
    body_renders = {}
    
    body_renders['front'] = os.path.join(output_dir, f"body_front.{extension}")
    body_renders['back'] = os.path.join(output_dir, f"body_back.{extension}")

    cam_pose_front = np.array([[1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, -0.45],   
                                [0.0, 0.0, 1.0, 2.1],
                                [0.0, 0.0, 0.0, 1.0]])
    cam_pose_back = np.array([[1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, -0.45],   
                                [0.0, 0.0, -1.0, -2.1],
                                [0.0, 0.0, 0.0, 1.0]])
    
    render_body_view(base_mesh, sphere_meshes, cam_pose_front, body_renders['front'], 
                     joints, sphere_radius)
    
    render_body_view(base_mesh, sphere_meshes, cam_pose_back, body_renders['back'], 
                     joints, sphere_radius)
 
    return 


def save_vertex_info(vertex_ids, positions, info_path):
    """Save vertex information to text file"""
    with open(info_path, 'w') as f:
        f.write(f"Sampled Vertices Information\n")
        f.write(f"Total vertices: {len(vertex_ids)}\n")
        f.write(f"Vertex IDs: {vertex_ids.tolist() if hasattr(vertex_ids, 'tolist') else list(vertex_ids)}\n\n")
        f.write("Vertex ID | X | Y | Z\n")
        f.write("-" * 30 + "\n")
        for i, (vid, pos) in enumerate(zip(vertex_ids, positions)):
            f.write(f"{vid:8d} | {pos[0]:8.4f} | {pos[1]:8.4f} | {pos[2]:8.4f}\n")


def generate_random_vertex_sample(total_vertices, num_samples=200, seed=42):
    """Generate random vertex IDs for sampling"""
    np.random.seed(seed)
    return np.random.choice(total_vertices, num_samples, replace=False)


def main_example():
    """Example usage"""
    # Setup (same as your original code)
    # os.environ['DISPLAY'] = ":12"
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    
    model_path = 'data/body_models/smplx/SMPLX_NEUTRAL.npz' 
    model = smplx.create(model_path,
                        model_type='smplx',
                        num_betas=16,
                        gender='neutral',
                        flat_hand_mean=True,
                        use_pca=False)
    
    
    # Generate sample vertex IDs (or use your own list)
    unique_len = len(np.unique(list(smplx_vertex_ids.values())))
    assert unique_len == len(smplx_vertex_ids), f'Non unique values {unique_len} {len(smplx_vertex_ids)}'
    
    unique_len = len(np.unique(list(smplx_vertex_ids.keys())))
    assert unique_len == len(smplx_vertex_ids), f'Non unique keys {unique_len} {len(smplx_vertex_ids)}'
    
    # sampled_vertex_ids = generate_random_vertex_sample(total_vertices, num_samples=200)
    sampled_vertex_ids = np.array(list(smplx_vertex_ids.values()))

    # Visualize sampled vertices
    result = visualize_sampled_vertices(
        model=model,
        vertex_ids=sampled_vertex_ids,
        output_dir="fusion_runs/renders/sampled_vertices",
        sphere_radius=0.01,  # Adjust sphere size as needed
        create_hand_views=True
    )
    
    print(f"Generated files:")
    print(f"- OBJ file: {result['obj_file']}")
    print(f"- Render: {result['render']}")
    print(f"- Vertex info: {result['vertex_info']}")
    print(f"- Total vertices visualized: {result['vertex_count']}")


if __name__ == '__main__':

    BODY_COLOR = [61, 133, 198, 255]
    SPHERE_COLOR = [255, 50, 50, 255]  
    SHOW_JOINTS = False

    # extension = 'png'
    extension = 'pdf'

    main_example()
