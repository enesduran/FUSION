import os
import bpy
import sys
import time    
import hydra

DIR = os.path.dirname(bpy.data.filepath)
if DIR not in sys.path:
    sys.path.append(DIR)

import json
import torch
import shutil
import joblib    
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf 
from omegaconf import DictConfig
from video import Video, add_text_moviepy
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)
 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
 
from src.blender.internals.render_pipeline_fusion import render_images_blender

def read_json(p):
    with open(p, 'r') as fp:
        json_contents = json.load(fp)
    return json_contents

def convert_time(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%02d" % (hour, minutes, seconds)

 
@hydra.main(config_path="/is/cluster/fast/eduran2/fusion/configs", config_name="render.yaml")  
def _render_cli(cfg: DictConfig) -> None:
     
    init = True
    dataset_path = cfg.dataset_path
    
    start_time = time.time() 
 
    # redirect output to log filei
    dataset_path = f'{dataset_path}' 
    ds = joblib.load(dataset_path)
    ds = {str(k):v for k, v in ds.items()}
    ds_name = os.path.basename(dataset_path).replace('.tar', '')
    logger.info(f'Dataset --> {ds_name}')
 
    if os.path.exists(os.path.dirname(str(cfg.subset))):
        subset_to_render = list(set(read_json(cfg.subset)))
        dict_render = {key: ds[key] for key in subset_to_render}
    elif isinstance(cfg.subset, str):
        subset_to_render = cfg.subset.split('_')
        subset_to_render = [el for el in subset_to_render if el]
        dict_render = {key: ds[key] for key in subset_to_render}
    elif os.path.isfile(dataset_path):
        if cfg.stamp_begin is not None:
            start_frame = cfg.stamp_begin
            end_frame = cfg.stamp_end
            ds['rots'] = ds['rots'][int(start_frame):int(end_frame)]
            ds['trans'] = ds['trans'][int(start_frame):int(end_frame)]
        dict_render = {ds_name: ds}            
    else:
        dict_render = ds
    print('Number of elements to render: ', len(dict_render))
    for keyid, data in tqdm(dict_render.items()):
        ds_fdp = os.path.dirname(dataset_path)

        if not bool(data):
            logger.info(f"empty dictionary aka buggy data sample with keyid: {keyid}")
            continue

        if cfg.mode == "video" or cfg.mode=='sequence':
            if cfg.savedir is not None:
                if cfg.stamp_begin is not None:
                    frames_folder = f'{cfg.savedir}/{keyid}_{cfg.stamp_begin}_{cfg.stamp_end}_{cfg.view}_frames'
                else:
                    frames_folder = f'{cfg.savedir}/{keyid}_{cfg.view}_frames'
            else:
                frames_folder = f'{ds_fdp}/{keyid}_{cfg.view}_frames'
        else:
            frames_folder = f'{ds_fdp}/{keyid}_{cfg.view}.png'

      
        if Path(frames_folder).is_file() and cfg.mode =='sequence':
            continue
                
        if 'object_dict' in ds.keys():
            if 'vertices' in ds['object_dict']:
                ds['object_vertices'] = ds['object_dict']['vertices'].copy()
                ds['object_faces'] = ds['object_dict']['faces'].copy()
            del ds['object_dict']
        

        for el, val in ds.items():
            
            # I added str and float 
            if val is None:
                continue
                
            if type(val) in [str, float, np.float32]:
                continue

            
            elif not torch.is_tensor(val):
                ds[el] = torch.from_numpy(val.astype(np.float64)).to('cuda').squeeze()

        
        if 'object_faces' in ds.keys():
            target_object = {'verts': ds['object_vertices'].cpu().numpy(),
                             'faces': ds['object_faces'].cpu().numpy().astype(np.uint32)}
        else:
            target_object = None
       
        motion_verts = ds['vertices'].detach().cpu().numpy()
        if 'joints' in ds.keys():
            motion_jts = ds['joints'].detach().cpu().numpy()
        molen = motion_verts.shape[0]
        
        if 'text_for_vid' in data:
            text = data['text_for_vid']
        else:
            text = None
        
        if 'lengths' in data:
            lens = [data['lengths']]
        else:
            lens = [molen]
         
        final_goal_locs = None
        if 'target_mask' in ds:
            target_mask = ds['target_mask'].detach().cpu()
            final_goal_locs = ds['target_loc'].detach().cpu() * target_mask
            final_goal_locs = final_goal_locs.numpy()
            
       
        if not cfg.object_visible:
            target_object = None
        if not cfg.targets_visible:
            final_goals_locs = None
        if not cfg.human_visible:
            motion_verts = None

        out = render_images_blender(motion_verts, frames_folder,
                                    render_engine=cfg.engine, 
                                    res=cfg.res,
                                    canonicalize=cfg.canonicalize,
                                    exact_frame=cfg.exact_frame,
                                    num=cfg.num, mode=cfg.mode,
                                    faces=cfg.smplx_faces,
                                    downsample=cfg.downsample,
                                    always_on_floor=cfg.always_on_floor,
                                    init=init,
                                    gt=cfg.gt,
                                    lengths=lens,
                                    cam=cfg.cam,
                                    color=cfg.fixed_color,
                                    fake_translation=cfg.fake_trans,
                                    separate_actions=cfg.separate_actions,
                                    texture_path=cfg.texture_path,
                                    debug=cfg.debug,
                                    sideview=cfg.view, 
                                    goals_tensor=final_goal_locs,
                                    computed_camera={'rot': ds['camera_rot'],
                                                     'trans': ds['camera_transl']},
                                    text=text,
                                    target_object=target_object,
                                    floor_flag=cfg.floor_flag,
                                    long_cam_flag=cfg.long_cam_flag)
            
        init = False                
        
        if cfg.mode == "video" and not cfg.debug:
            video = Video(out, fps=30.0)

            vid_path = out.replace('_frames', '')   
            vid_path = vid_path.rstrip('/')+'.mp4'

            if text is not None:
                video = add_text_moviepy(vid_path, text, position=('center', 'bottom'), font_size=50, color='white')
                video.write_videofile(vid_path)
            else:
                video.save(out_path=vid_path)

            shutil.rmtree(out)
        
        if cfg.mode == "sequence" and not cfg.debug:

            if text is not None:
                add_text_to_image(out, text, out, position=('center', 'bottom'), font_size=50, color=(255, 255, 255))

    logger.info(f"TOTAL TIME ELAPSED:\n-----> H:M:S => {convert_time(time.time() - start_time)}")
 

def add_text_to_image(image_path, text, output_path, position=('bottom'), font_size=40, color=(255, 255, 255)):

    
    color = (186,85,211) # mediumorchid
    
    color = (139,0,139) # dark magenta 

    color = (7,55,99) # dark blue

    # Load the rendered image
    image = Image.open(image_path[0]).convert("RGBA")

    # Create drawing context
    draw = ImageDraw.Draw(image)

    # Load a TTF font
    font_path = "/is/cluster/eduran2/fonts/times.ttf"
    font = ImageFont.truetype(font_path, font_size)

    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]


    if 'center' in position:
        image_x = (image.width - text_width) // 2
        image_y = image.height // 2
        
    if 'left' in position:
        image_x = 0
        
    if 'right' in position:
        image_x = image.width 

    if 'top' in position:
        image_y = 0

    if 'bottom' in position:
        image_y = image.height - text_height - 10
        

    # Draw the text
    draw.text((image_x, image_y), text, font=font, fill=color, align='right')

    # Save the result
    image.save(output_path[0])


if __name__ == '__main__':
    _render_cli()
 