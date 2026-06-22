import os
import sys
import torch
import smplx
import joblib
import trimesh
import pyrender
import numpy as np
from PIL import Image 

os.environ['PYOPENGL_PLATFORM'] = 'egl'

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from utils.transforms import quat_fk
from utils.transforms3d import transform_body_pose
from utils.process_utils import RIGHT_WRIST_BASE_LOC, LEFT_WRIST_BASE_LOC

def concat_images_horizontally(image_paths, out_path):
    images = [Image.open(p) for p in image_paths]
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    result = Image.new('RGB', (total_width, max_height), (255, 255, 255))
    
    x_offset = 0
    for im in images:
        result.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    result.save(out_path)
 
    # Save as PDF
    result.save(out_path.replace('png', 'pdf'))
     

def mesh_to_pyrender(mesh: trimesh.Trimesh, translation=np.zeros(3)):
    """Convert trimesh to pyrender mesh with optional translation."""
    mesh.apply_translation(translation)
    return pyrender.Mesh.from_trimesh(mesh, smooth=True)

def main():
    
    num_betas = 10
    gender = 'neutral'
    device = torch.device('cpu')
    model_path = 'data/body_models/smplx/SMPLX_NEUTRAL.npz'
    
    rhm_path = 'data/body_models/mano/MANO_RIGHT.pkl'
    lhm_path = 'data/body_models/mano/MANO_LEFT.pkl'
    
    # load data 
    data = joblib.load('data/motion/precomputed_val_amass2.pth.tar')
    
    _k_ = 33
    _t_ = 63
    
    assert data[_k_]['body_dataset_name'] == 'OMOMO'
    
    # --- Load Models ---
    smplx_model = smplx.create(model_path,
                    model_type='smplx',
                    num_betas=num_betas,
                    gender=gender,
                    flat_hand_mean=True,
                    use_pca=False)
    
    
    joint_offsets = smplx_model().joints.detach()[:, :55]
    
    lrot_quat = data[_k_]['rots'][_t_ : _t_ + 1].reshape(1, 55, 3)
    lrot_quat[0, 0] = torch.tensor([0.0, 0.0, 0.0])
    
    lrot_quat = transform_body_pose(data[_k_]['rots'][_t_ : _t_ + 1].reshape(1, 55, 3), 'aa->quat')
    
    global_rot, global_pos = quat_fk(lrot_quat, 
                                    joint_offsets, 
                                    smplx_model.parents)
    
    global_rot = transform_body_pose(global_rot, 'quat->aa')
    
    mano_right = smplx.create(rhm_path, model_type='mano', 
                              is_rhand=True, 
                              flat_hand_mean=True,
                              use_pca=False).to(device)
    
    mano_left = smplx.create(lhm_path, 
                             model_type='mano', 
                             flat_hand_mean=True,
                             is_rhand=False, 
                             use_pca=False).to(device)

    # --- Dummy poses (replace with your data) ---
    betas = torch.zeros((1, 10)).to(device)
    global_orient = torch.zeros((1, 3)).to(device)
    body_pose = data[_k_]['rots'][_t_ : _t_ + 1, 3:66]
    mano_right_pose = data[_k_]['rots'][_t_ : _t_ + 1, -45:]
    mano_left_pose = data[_k_]['rots'][_t_ : _t_ + 1, -90:-45]
    
    # --- Generate meshes ---
    smplx_nohands = smplx_model(betas=betas,
                                global_orient=global_orient,
                                body_pose=body_pose,
                                left_hand_pose=torch.zeros_like(mano_left_pose),
                                right_hand_pose=torch.zeros_like(mano_right_pose),
                                return_verts=True)
    
    mano_right_out = mano_right(global_orient=global_rot[0, 21:22],
                                hand_pose=mano_right_pose,
                                transl=smplx_nohands.joints.detach()[0, 21:22] - RIGHT_WRIST_BASE_LOC,
                                return_verts=True)

    mano_left_out = mano_left(global_orient=global_rot[0, 20:21],
                              hand_pose=mano_left_pose,
                              transl=smplx_nohands.joints.detach()[0, 20:21] - LEFT_WRIST_BASE_LOC,
                              return_verts=True)
    
    smplx_withhands = smplx_model(betas=betas,
                                  global_orient=global_orient,
                                  body_pose=body_pose,
                                  left_hand_pose=mano_left_pose,
                                  right_hand_pose=mano_right_pose,
                                  return_verts=True)

    # --- Convert to Trimesh ---
    smplx_faces = smplx_model.faces
    mano_rfaces = mano_right.faces
    mano_lfaces = mano_left.faces

    blue_vertex_colors = [61, 133, 198, 255]
    # blue_vertex_colors = [37, 150, 190, 255]
    green_vertex_colors = [106, 168, 79, 255]

    cyan = [64, 224, 208]
    olive = [100, 100, 0]
        
    mesh1 = trimesh.Trimesh(smplx_nohands.vertices[0].cpu().detach().numpy(), smplx_faces, vertex_colors=blue_vertex_colors)
    mesh2 = trimesh.Trimesh(mano_right_out.vertices[0].cpu().detach().numpy(), mano_rfaces, vertex_colors=green_vertex_colors)
    mesh3 = trimesh.Trimesh(mano_left_out.vertices[0].cpu().detach().numpy(), mano_lfaces, vertex_colors=green_vertex_colors)
    mesh4 = trimesh.Trimesh(smplx_withhands.vertices[0].cpu().detach().numpy(), smplx_faces, vertex_colors=blue_vertex_colors)
        
    # --- Translate for spacing ---
    mesh2.apply_translation([0, 0, 0])
    mesh3.apply_translation([0, 0, 0])
    mesh4.apply_translation([0, 0, 0])
    
    hands_combined = trimesh.util.concatenate([mesh2, mesh3])
    

    for mesh, mesh_name in zip([mesh1, hands_combined, mesh4], 
                               ['smplx_nohands', 'mano_right_mano_left', 'smplx_withhands']):

        # --- Build Pyrender Scene ---
        scene = pyrender.Scene(bg_color=[255, 255, 255, 255])
        scene.add(mesh_to_pyrender(mesh))

        # --- Render Scene ---
        scene_size = (470, 1000) if 'smplx' in mesh_name else (400, 1000)
        aspect_ratio = scene_size[0]/scene_size[1] 

        # Camera & light
        camera = pyrender.PerspectiveCamera(yfov = np.pi/(4.0), 
                                            aspectRatio = aspect_ratio)
        
        if 'smplx' in mesh_name:
            cam_pose = np.array([[1, 0, 0, 0.0],
                                [0, 1, 0, -0.5],
                                [0, 0, 1, 2],
                                [0, 0, 0, 1]])
        else:
            cam_pose = np.array([[1, 0, 0, 0.05],
                                [0, 1, 0, -0.5],
                                [0, 0, 1, 2],
                                [0, 0, 0, 1]])



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


        r = pyrender.OffscreenRenderer(scene_size[0], scene_size[1])
        color, _ = r.render(scene)
        os.makedirs("fusion_runs/renders/hand_merging", exist_ok=True)
        from PIL import Image
        Image.fromarray(color).save(f"fusion_runs/renders/hand_merging/{mesh_name}.png")
        r.delete()
    
    
    concat_images_horizontally(['fusion_runs/renders/hand_merging/smplx_nohands.png', 
                                'fusion_runs/renders/hand_merging/mano_right_mano_left.png', 
                                'fusion_runs/renders/hand_merging/smplx_withhands.png'], 
                                'fusion_runs/renders/hand_merging/smplx_mano_smplx.png')

if __name__ == '__main__':
    main()
