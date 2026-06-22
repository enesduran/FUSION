import os
import sys
import smplx
import torch
import joblib
import random 
import shutil
 
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.render.mesh_viz import RendererWrapper
from src.utils.viz_utils import pack_to_render
from src.utils.process_utils import BRANCH_NAME

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

color1 = (160 / 255, 160 / 255, 0 / 255, 1.0)
color2 = (0 / 255, 160 / 255, 160 / 255, 1.0)
color3 = (160 / 255, 0 / 255, 160 / 255, 1.0)


SMPLX_PATH = 'data/body_models/smplx/SMPLX_NEUTRAL.npz'
MANO_PATH = 'data/body_models/mano/MANO_RIGHT.pkl'

WINDOW = 120

mano_layer = smplx.create(model_path=MANO_PATH, 
                          model_type='MANO',
                          gender='neutral',
                          num_betas=10,
                          batch_size=WINDOW,
                          flat_hand_mean=True,
                          use_pca=False,
                          ).to(device)


smplx_layer = smplx.create(model_path=SMPLX_PATH, 
                          model_type='smplx',
                          gender='neutral',
                          num_betas=300,
                          batch_size=WINDOW,
                          flat_hand_mean=True,
                          use_pca=False,
                          ).to(device)
 

body_flag = True


if body_flag:
    # datasetname = 'OMOMO'
    # datasetname = 'AMASS'
    # datasetname = 'SAMP'
    # datasetname = 'ARCTIC'
    # datasetname = 'GRAB'
    # datasetname = 'BEAT2'
    # datasetname = 'EMBODY3D'
    # datasetname = 'INTERX'
    datasetname = 'MAMMA'

else:
    # datasetname = 'Interhands'
    # datasetname = 'Reinterhands'
    # datasetname = 'Hot3d'
    # datasetname = 'MOYO'
    # datasetname = 'GRAB'
    # datasetname = 'ARCTIC'
    datasetname = 'EMBODY3D'

if len(sys.argv) > 1:
    datasetname = sys.argv[1]

max_renders = int(sys.argv[2]) if len(sys.argv) > 2 else None
 
hand_dataset_path = f'data/motion/Hand_Processed/{datasetname.lower()}_train.p'
body_dataset_path = f'data/motion/Body_Processed/{datasetname.lower()}_train.p'
 
rendering_scale = 2.0 if body_flag else 0.5

colors = [ (160 / 255, 160 / 255, 0 / 255, 1.0), 
           (0 / 255, 160 / 255, 160 / 255, 1.0),
           (160 / 255, 0 / 255, 160 / 255, 1.0),]
  

basename = f'fusion_runs/{BRANCH_NAME}/body_dataset_vis' if body_flag else f'fusion_runs/{BRANCH_NAME}/hand_dataset_vis'
os.makedirs(basename, exist_ok=True)

if os.path.exists(f'{basename}/{datasetname}'):
    shutil.rmtree(f'{basename}/{datasetname}')
os.makedirs(f'{basename}/{datasetname}', exist_ok=False)


data_path = body_dataset_path if body_flag else hand_dataset_path
path2body_models = SMPLX_PATH if body_flag else MANO_PATH
faces = smplx_layer.faces if body_flag else mano_layer.faces
body_model = smplx_layer if body_flag else mano_layer

motion_dict = joblib.load(data_path)


renderer = RendererWrapper(path2body_models)

keys = list(motion_dict.keys())
random.shuffle(keys)
if max_renders is not None:
    keys = keys[:max_renders]

 

for k in keys:
    
    v = motion_dict[k]
    
    if body_flag:
        
        betas = torch.repeat_interleave(torch.tensor(v['betas'][None]).to(device).float(), WINDOW, 0)
        expression = torch.zeros((WINDOW, 10)).to(device).float()
        
        motion_params_gt = {
                            'betas': betas,
                            'expression': expression,
                            'global_orient': torch.tensor(v['root_orient']).to(device).float(), 
                            'body_pose': torch.tensor(v['pose_body']).to(device).float(),
                            'left_hand_pose': torch.tensor(v['pose_lhand']).to(device).float(),
                            'right_hand_pose': torch.tensor(v['pose_rhand']).to(device).float(),
                            'jaw_pose': torch.tensor(v['pose_jaw']).to(device).float(),
                            'leye_pose': torch.tensor(v['pose_eye'])[:, :3].to(device).float(),
                            'reye_pose': torch.tensor(v['pose_eye'])[:, 3:].to(device).float(),
                            'transl': torch.tensor(v['trans']).to(device).float(), 
                            }
        
        filename = f'{basename}/{datasetname}/{k:06d}_{v["augment_flag"]}'
     
    else: 
         motion_params_gt = {
                        # 'global_orient': torch.tensor(v['root_orient']).float().to(device), 
                        'hand_pose': torch.tensor(v['pose_rhand']).float().to(device), 
                        # 'transl': torch.tensor(v['trans']).float().to(device)
                        }
         
         filename = f'{basename}/{datasetname}/{k:06d}_{v["time_augment_flag"]}_{v["pose_augment_flag"]}'
         
         
    
    vertices = body_model(**motion_params_gt).vertices
    
                                   
    mesh_dict = {'vertices': vertices.cpu().detach().numpy(),
                'faces': faces}
     
    root_orient = torch.tensor(v['root_orient']).float()
    root_orient[:, 1] = 0
    root_orient[:, 2] = 0

    camera_dict = {'camera_transl': torch.from_numpy(v['trans']).float(), 
                   'camera_rot': root_orient,
                    'coef': 1.9}
     

     
    renderer.render_motion([mesh_dict],  
                           filename=filename,
                           camera_dict=camera_dict,
                           rendering_scale=rendering_scale,
                           color = [color1, color2, color3],)
