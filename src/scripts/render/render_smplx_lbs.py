import os

os.environ['DISPLAY'] = ":12"
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import smplx
import torch
import pickle
import trimesh
import pyrender
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pdf2image import convert_from_path
from smplx.lbs import batch_rodrigues, batch_rigid_transform, vertices2joints

JOINT_RADIUS = 0.01
JOINT_NUM = 55

def get_face_colors_by_body_parts():
    """Assign colors to vertices based on body part segmentation"""

    # parts['parents']
    with open('data/body_models/smplx_parts_segm.pkl', 'rb') as f:
        parts = pickle.load(f, encoding='latin1')


    unique_parts = np.unique(parts['segm'])
    num_parts = unique_parts.shape[0]

    face_colors = np.zeros((parts['segm'].shape[0], 3))    
    
    
    # colors = plt.cm.Set3(np.linspace(0, 1, num_parts))
    colors = generate_contrasting_colors(num_parts)

    for i in unique_parts:

        # discard alpha value. 
        face_colors[parts['segm'] == i] = colors[i, :3]

    return face_colors


def generate_contrasting_colors(n_colors):
    """Generate n highly contrasting colors using HSV color space"""
    colors = []
    
    # Use golden ratio to distribute hues evenly
    golden_angle = np.pi * (3 - np.sqrt(5))  # Golden angle in radians
    
    for i in range(n_colors):
        # Distribute hues evenly using golden angle
        hue = (i * golden_angle) % (2 * np.pi)
        hue_normalized = hue / (2 * np.pi)
        
        # Alternate saturation and value for more contrast
        saturation = 0.9 if i % 2 == 0 else 0.7
        value = 0.9 if i % 3 != 0 else 0.6
        
        # Convert HSV to RGB
        from colorsys import hsv_to_rgb
        r, g, b = hsv_to_rgb(hue_normalized, saturation, value)
        colors.append([r, g, b])
    
    return np.array(colors)

def render_mesh_to_suffix(mesh, joints_3d=None, out_path="output.png", vertex_colors=None, 
                       face_colors=None, cam_rot=None, image_size=(1700, 2000), show_joints=False):
    """Enhanced rendering with joint visualization and better lighting"""
    
    # Create scene with better lighting setup
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], 
                          ambient_light=[0.3, 0.3, 0.3])


    # Create mesh with vertex colors if provided
    if vertex_colors is not None:
        trimesh_obj = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, 
                                     vertex_colors=vertex_colors)
        smooth_flag = True  
        
    elif face_colors is not None:
        trimesh_obj = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, 
                                     face_colors=face_colors)
        smooth_flag = False  
    else:
        trimesh_obj = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
        smooth_flag = False 
    
        # trimesh_obj.visual.vertex_colors = [213, 167, 132, 255] 
        trimesh_obj.visual.vertex_colors = [161, 103, 64, 255] 
 
    
    render_mesh = pyrender.Mesh.from_trimesh(trimesh_obj, smooth=smooth_flag)
    
    # Set material properties for better shading
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        roughnessFactor=0.8,
        baseColorFactor=[1.0, 1.0, 1.0, 1.0]
    )
    render_mesh.material = material
    
    scene.add(render_mesh)
    
    # Add joint spheres if requested - render them AFTER the main mesh for visibility
    joint_nodes = []
    if show_joints and joints_3d is not None:
        for joint_pos in joints_3d:
            joint_sphere = trimesh.creation.icosphere(radius=JOINT_RADIUS , subdivisions=2)
            joint_sphere.vertices += joint_pos
            joint_sphere.visual.vertex_colors = [255, 255, 255, 255]  # White joints
            
            # Create material for joints to make them more visible
            joint_material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.0,
                roughnessFactor=0.1,  # More reflective
                baseColorFactor=[1.0, 1.0, 1.0, 1.0],
                emissiveFactor=[0.2, 0.2, 0.2]  # Make them slightly emissive
            )
            
            joint_mesh = pyrender.Mesh.from_trimesh(joint_sphere, material=joint_material)
            joint_node = scene.add(joint_mesh)
            joint_nodes.append(joint_node)


    # Camera setup
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.8, aspectRatio=0.85)
    cam_pose = np.array([[1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, -0.4],   
                        [0.0, 0.0, 1.0, 2.5],
                        [0.0, 0.0, 0.0, 1.0]])
    
    if cam_rot is not None:
        cam_R = np.eye(4)
        cam_R[:3, :3] = cam_rot
        cam_pose = cam_R @ cam_pose

    scene.add(camera, pose=cam_pose)
    
    # Add multiple lights for better illumination
    # Key light
    key_light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(key_light, pose=cam_pose)
    
    # Fill light from the side
    fill_pose = cam_pose.copy()
    fill_pose[0, 3] = 1.0  # Move to the side
    fill_light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.5)
    scene.add(fill_light, pose=fill_pose)
    
    # Back light
    back_pose = cam_pose.copy()
    back_pose[2, 3] = -2.0  # Behind the subject
    back_light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
    scene.add(back_light, pose=back_pose)

    # Render
    r = pyrender.OffscreenRenderer(*image_size)
    color, depth = r.render(scene)
    
    
    # If we have joints, render them separately with depth disabled and composite
    if show_joints and joints_3d is not None and joint_nodes:
        # Create a new scene with just joints
        joint_scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.3, 0.3, 0.3])
        
        # Add joints to joint scene
        for joint_pos in joints_3d:
            joint_sphere = trimesh.creation.icosphere(radius=JOINT_RADIUS, subdivisions=2)  # Even larger
            joint_sphere.vertices += joint_pos
            joint_sphere.visual.vertex_colors = [255, 255, 255, 255]
            
            joint_material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.0,
                roughnessFactor=0.1,
                baseColorFactor=[1.0, 1.0, 1.0, 1.0],
                emissiveFactor=[0.3, 0.3, 0.3]
            )
            
            joint_mesh = pyrender.Mesh.from_trimesh(joint_sphere, material=joint_material)
            joint_scene.add(joint_mesh)
        
        # Add camera and lights to joint scene
        joint_scene.add(camera, pose=cam_pose)
        joint_scene.add(key_light, pose=cam_pose)
        joint_scene.add(fill_light, pose=fill_pose)
        joint_scene.add(back_light, pose=back_pose)
        
        # Render joints
        joint_color, joint_depth = r.render(joint_scene, flags=pyrender.RenderFlags.RGBA)
        
        # Composite joint rendering on top of main rendering
        # Where joint alpha > 0, use joint color
        joint_alpha = joint_color[:, :, 3:4] / 255.0
        color = color * (1 - joint_alpha) + joint_color[:, :, :3] * joint_alpha
        color = color.astype(np.uint8)
    
    
    Image.fromarray(color).save(out_path)
    r.delete()


def concat_images_horizontally(image_paths, out_path):
   
    
    # read pngs 
    if out_path.endswith('.png'):
        images = [Image.open(p) for p in image_paths]         
    # read pdfs 
    else:
        images = [convert_from_path(p)[0] for p in image_paths]
        
        
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    result = Image.new('RGB', (total_width, max_height), (255, 255, 255))
    
    x_offset = 0
    for im in images:
        result.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    result.save(out_path)


def main():
    model_path = 'data/body_models/smplx/SMPLX_NEUTRAL.npz' 
    gender = 'neutral'
    
    num_betas = 16
    batch_size = 1
    
    model = smplx.create(model_path,
                model_type='smplx',
                num_betas=num_betas,
                gender=gender,
                flat_hand_mean=True,
                use_pca=False)
    
    # Zero parameters
    zero_betas = torch.zeros([batch_size, num_betas])
    zero_body_pose = torch.zeros([batch_size, 21 * 3])
    zero_global_orient = torch.zeros([batch_size, 3])
    zero_expression = torch.zeros([batch_size, 10])
    zero_jaw_pose = torch.zeros([batch_size, 3])
    zero_reye_pose = torch.zeros([batch_size, 3])
    zero_leye_pose = torch.zeros([batch_size, 3])
    zero_left_hand_pose = torch.zeros([batch_size, 15 * 3])
    zero_right_hand_pose = torch.zeros([batch_size, 15 * 3])
    
    # Load AMASS data
    datapath = '/is/cluster/fast/eduran2/omomo_fullbody/data/AMASS/ACCAD/Male2MartialArtsStances_c3d/D9_-_victory_1_stageii.npz'
    data = np.load(datapath)
     
    _k_ = 400
    
    body_pose = torch.tensor(data['pose_body'][_k_:_k_ + 1]).float()
    global_orient = torch.tensor(data['root_orient'][_k_:_k_ + 1]).float()
    global_orient = torch.zeros((1, 3))  # Keep in T-pose for visualization
    
    expression = torch.zeros([batch_size, 10])
    jaw_pose = torch.tensor(data['pose_jaw'][_k_:_k_ + 1]).float()
    left_hand_pose = torch.tensor(data['pose_hand'][_k_:_k_ + 1][:, :45]).float()
    right_hand_pose = torch.tensor(data['pose_hand'][_k_:_k_ + 1][:, 45:]).float()
    
    global_orient_rot_mats = batch_rodrigues(global_orient.view(-1, 3)).view([3, 3])
    betas = torch.tensor(data['betas'][None]).float()    

    # 1. Template mesh (no shape, no pose) - WITH BODY PART COLORS
    template_output = model(global_orient=zero_global_orient)
    template_vertices = template_output.vertices[0].detach()
    template_joints = template_output.joints[0, :JOINT_NUM].detach()
    
    # Get body part colors for template
    template_face_colors = get_face_colors_by_body_parts()
 
    # 2. Shape blend only in T-pose
    shape_output = model(global_orient=zero_global_orient, betas=betas)
    vertices_shape = shape_output.vertices[0].detach()
    joints_shape = shape_output.joints[0, :JOINT_NUM].detach()
    
    # Manual computation for step 3 (shape + pose blends)
    dtype = model.shapedirs.dtype
    device = model.shapedirs.device
    
    pose = torch.cat([global_orient.reshape(-1, 1, 3),
                    body_pose.reshape(-1, 21, 3),
                    jaw_pose.reshape(-1, 1, 3),
                    zero_leye_pose.reshape(-1, 1, 3),
                    zero_reye_pose.reshape(-1, 1, 3),
                    left_hand_pose.reshape(-1, 15, 3),
                    right_hand_pose.reshape(-1, 15, 3)], dim=1)

    rot_mats = batch_rodrigues(pose.view(-1, 3)).view([batch_size, -1, 3, 3])
    
    ident = torch.eye(3, dtype=dtype, device=device)
    pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
    pose_offsets = torch.matmul(pose_feature, model.posedirs).view(batch_size, -1, 3)[0]
    
    # Shape + pose blend shapes in T-pose
    shape_offsets = torch.einsum('bl,mkl->bmk', [betas, model.shapedirs])
    v_shaped = model.v_template + shape_offsets
    
    J_transformed, A = batch_rigid_transform(rot_mats,
                                             vertices2joints(model.J_regressor, v_shaped), 
                                             model.parents, 
                                             dtype=dtype)
    
    vertices_pose_shape = (model.v_template + pose_offsets + shape_offsets) 
    vertices_pose_shape = torch.cat([vertices_pose_shape, 
                              torch.ones([batch_size, v_shaped.shape[1], 1],
                               dtype=dtype, device=device)], dim=2)
    
    vertices_pose_shape = torch.matmul(A[0, 0], torch.unsqueeze(vertices_pose_shape, dim=-1))[0, :, :3, 0]
    
    # Get joints for step 3
    joints_pose_shape = vertices2joints(model.J_regressor, vertices_pose_shape.unsqueeze(0))[0]
    
    # 4. Posed mesh (non-zero pose + shape)
    output_posed = model(return_verts=True,
                         betas=betas,
                         body_pose=body_pose,
                         global_orient=global_orient,
                         expression=expression,
                         jaw_pose=jaw_pose,
                         left_hand_pose=left_hand_pose,
                         right_hand_pose=right_hand_pose)

    # Prepare meshes and data
    mesh_list = [template_vertices.detach().cpu().numpy(),
                vertices_shape.detach().cpu().numpy(),
                vertices_pose_shape.detach().cpu().numpy(),
                output_posed.vertices[0].detach().cpu().numpy()]
    
    joint_list = [template_joints.detach().cpu().numpy(),
                 joints_shape.detach().cpu().numpy(),
                 joints_pose_shape.detach().cpu().numpy(),
                 output_posed.joints[0, :JOINT_NUM].detach().cpu().numpy()]
    
    faces = model.faces
    mesh_objs = [trimesh.Trimesh(vertices=v, faces=faces) for v in mesh_list]
    
    # Color settings
    vertex_colors_list = [
        template_face_colors,  # Body part colors for template
        None,  # Default for shape
        None,  # Default for pose+shape
        None   # Default for final posed
    ]
    
    # Joint visibility settings
    show_joints_list = [True, True, True, False]  # Show joints for all
    
    suffix = 'png'
    
    # Save each mesh to image
    os.makedirs('fusion_runs/renders/smplx_blend_shapes', exist_ok=True)
    image_paths = []
    
    for i, (mesh, joints, face_colors, show_joints) in enumerate(
        zip(mesh_objs, joint_list, vertex_colors_list, show_joints_list)):
        
        img_path = f"fusion_runs/renders/smplx_blend_shapes/view_{i+1}.{suffix}"
        
        render_mesh_to_suffix(mesh, joints, img_path, None, face_colors, 
                          global_orient_rot_mats, show_joints=show_joints)
        image_paths.append(img_path)

    # Concatenate and save
    concat_images_horizontally(image_paths, f"fusion_runs/renders/smplx_blend_shapes/smplx_lbs.{suffix}")
    print(f"Saved all views to: fusion_runs/renders/smplx_blend_shapes/smplx_lbs.{suffix}")
    

if __name__ == '__main__':
    main()