import os
import sys
import torch

os.system("Xvfb :12 -screen 0 2400x1800x24 &")
os.environ['DISPLAY'] = ":12"

from src.render.video import put_text
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.meshes import Meshes
from aitviewer.headless import HeadlessRenderer
from aitviewer.renderables.spheres import Spheres
from src.utils.process_utils import CONTACT_INDICES

from omegaconf import OmegaConf
from aitviewer.renderables.skeletons import Skeletons
from aitviewer.configuration import CONFIG as AITVIEWER_CONFIG
from src.utils.transforms3d import get_z_rot, transform_body_pose


smplx_skeleton = [[-1, 0], [0, 1], [0, 2], [0, 3], [1, 4], [2, 5], 
                  [3, 6], [4, 7], [5, 8], [6, 9], [7, 10], [8, 11], 
                  [9, 12], [9, 13], [9, 14], 
                  [12, 15], [13, 16], [14, 17], [16, 18], [17, 19],
                  [18, 20], [19, 21], [15, 22], [15, 23], [15, 24], 
                  [20, 25], [25, 26], [26, 27], [20, 28], [28, 29], 
                  [29, 30], [20, 31], [31, 32], [32, 33], [20, 34], 
                  [34, 35], [35, 36], [20, 37], [37, 38], [38, 39], 
                  [21, 40], [40, 41], [41, 42], [21, 43], [43, 44], 
                  [44, 45], [21, 46], [46, 47], [47, 48], [21, 49], 
                  [49, 50], [50, 51], [21, 52], [52, 53], [53, 54]] 

smpl_skeleton_wo_hands = [[-1, 0], [0, 1], [0, 2], [0, 3], [1, 4], [2, 5], 
                  [3, 6], [4, 7], [5, 8], [6, 9], [7, 10], [8, 11], 
                  [9, 12], [9, 13], [9, 14], [12, 15], [13, 16], 
                  [14, 17], [16, 18], [17, 19], [18, 20], [19, 21]]

mano_skeleton = [[-1, 0], [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], 
                  [5, 6], [0, 7], [7, 8], [8, 9], [0, 10], [10, 11], 
                  [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], 
                  [3, 17], [6, 18], [12, 19], [9, 20]]

skel_dict = {len(smplx_skeleton): smplx_skeleton, 
             len(smpl_skeleton_wo_hands): smpl_skeleton_wo_hands, 
             len(mano_skeleton): mano_skeleton}
render_contact_dict = {len(smplx_skeleton): True,
                       len(smpl_skeleton_wo_hands): True,
                       len(mano_skeleton): False}


colors_dict = {
   0: (0.863, 0.078, 0.235, 1.0),  # Crimson Red
   1: (0.098, 0.098, 0.439, 1.0),  # Midnight Blue
   2: (0.133, 0.545, 0.133, 1.0),  # Forest Green
   3: (1.000, 0.549, 0.000, 1.0),  # Dark Orange
   4: (0.502, 0.000, 0.502, 1.0),  # Purple
   5: (0.000, 0.502, 0.502, 1.0),  # Teal
   6: (0.275, 0.510, 0.706, 1.0),  # Steel Blue
   7: (0.545, 0.271, 0.075, 1.0),  # Saddle Brown
   8: (0.647, 0.165, 0.165, 1.0),  # Brown
   16: (0.647, 0.300, 0.300, 1.0), # Rosy Brown
   9: (0.282, 0.239, 0.545, 1.0),  # Dark Slate Blue
   10: (0.722, 0.525, 0.043, 1.0), # Dark Goldenrod
   11: (0.333, 0.420, 0.184, 1.0), # Dark Olive Green
   12: (0.804, 0.361, 0.361, 1.0), # Indian Red
   13: (0.627, 0.322, 0.176, 1.0), # Sienna
   14: (0.373, 0.620, 0.627, 1.0), # Cadet Blue
   15: (0.439, 0.502, 0.565, 1.0), # Slate Gray
   17: (0.184, 0.310, 0.310, 1.0), # Dark Slate Gray
   18: (0.502, 0.502, 0.502, 1.0), # Gray
   19: (0.412, 0.412, 0.412, 1.0), # Dim Gray
}

class RendererWrapper:
    def __init__(self, path2body_models,
                 window_width=1600,
                 window_height=1200, 
                 z_up=True,
                 auto_set_floor=True) -> None:

        self.CONTACT_CONF_THRESHOLD = 0.5        

        AITVIEWER_CONFIG.update_conf(OmegaConf.create({
                                "z_up": z_up,
                                "playback_fps": 30,
                                "window_type": "pyglet", 
                                "window_width": window_width,
                                "window_height": window_height,
                                "smplx_models": path2body_models,
                                "auto_set_floor": auto_set_floor}))

        self.renderer = HeadlessRenderer(samples=4)

 
    def render_skeleton(self, 
                        positions: torch.Tensor, 
                        filename: str, 
                        text_for_vid=None,
                        color=(1/255, 1 / 255, 1.0, 1.0)) -> None:
      
        skeletons_seq = Skeletons(joint_positions=positions, 
                                joint_connections=smplx_skeleton,
                                color=color,
                                radius=0.03)

        self.renderer.scene.add(skeletons_seq)
        # camera follows smpl sequence
        camera = self.renderer.lock_to_node(skeletons_seq, (2, 2, 2), smooth_sigma=5.0)
        
        self.renderer.save_video(video_dir=str(filename), output_fps=30)
        os.rename(filename + '_0.mp4', filename + '.mp4')

        # empty scene for the next rendering
        self.renderer.scene.remove(skeletons_seq)
        self.renderer.scene.remove(camera)
    
        if text_for_vid is not None:
            fname = put_text(text_for_vid, f'{filename}.mp4', f'{filename}_ts.mp4')
            os.remove(f'{filename}.mp4')
        else:
            fname = f'{filename}.mp4'

        return fname
 
 
    def render_motion(self, 
                    mesh_list: dict, 
                    filename: str, 
                    skeleton_dict: dict = {},
                    target_dict: dict = {},
                    object_dict: dict = {},
                    camera_dict: dict = {},
                    text_for_vid='', 
                    color=(160 / 255, 160 / 255, 160 / 255, 1.0),
                    rendering_scale=2.0,
                    output_fps=30) -> None:
        """
        Function to render a video of a motion sequence
        renderer: aitviewer renderer
        datum: dictionary containing sequence of poses, body translations and body orientations
            data could be numpy or pytorch tensors
        filename: the absolute path you want the video to be saved at

        """

        if isinstance(mesh_list, dict): mesh_list = [mesh_list]

        if not isinstance(color, list): 
            colors = [color] 
        else:
            colors = color
        
        scene_elements = []
        

        for iid, mesh_seq in enumerate(mesh_list):
  
            smpl_template = Meshes(mesh_seq['vertices'], 
                                    mesh_seq['faces'], 
                                    color=colors[iid],
                                    name="Human")
            
            
            scene_elements.append(smpl_template)
            self.renderer.scene.add(smpl_template)
                   
        
        if skeleton_dict != {}:
            
            joint_pos = skeleton_dict['positions']

            if 'contact_masks' in skeleton_dict.keys():

                contact_mask = skeleton_dict['contact_masks']
                # thresholding 
                contact_mask[contact_mask>self.CONTACT_CONF_THRESHOLD] = 1
                contact_mask[contact_mask<=self.CONTACT_CONF_THRESHOLD] = 0


                # rendering contacts
                if render_contact_dict[joint_pos.shape[1]]:
                    contacts_spheres = Spheres(positions=joint_pos[:, CONTACT_INDICES] * contact_mask[..., None], 
                                        radius=0.02,
                                        color=(0.2, 0.8, 0.2, 1.0))
                
                    scene_elements.append(contacts_spheres)
                    self.renderer.scene.add(contacts_spheres)

            
            if 'render_skeleton' in skeleton_dict.keys():
         
                joint_connections = skel_dict[joint_pos.shape[1]]
                
                skeletons_seq = Skeletons(joint_positions=joint_pos * skeleton_dict['skeleton_masks'], 
                                    joint_connections=joint_connections,
                                    color=(0.1, 0.1, 0.3, 1.0),
                                    radius=0.005)

                scene_elements.append(skeletons_seq)
                self.renderer.scene.add(skeletons_seq)
            
                
        if object_dict != {}:

            obj_mesh = Meshes(object_dict['vertices'], 
                    object_dict['faces'], 
                    is_selectable=False,
                    gui_affine=False,
                    # color=(0.7, 0.1, 0.0, 1.0),
                    color=colors_dict[16],
                    name="Object_mesh")
    
            scene_elements.append(obj_mesh)
            self.renderer.scene.add(obj_mesh)
        
        
        if target_dict != {}:
            loc = target_dict['target_location']
            mask = target_dict['target_mask']

            if 'use_different_colours' in target_dict.keys():

                # get colors 
                remainder_tensor = torch.remainder(torch.arange(loc.shape[1]), max(list(colors_dict.keys())) + 1)
                _color_ = torch.tensor([colors_dict[remain.item()] for remain in remainder_tensor])

                # make it double so that targets match in color.
                _color_ = _color_.repeat_interleave(2, dim=0)[:loc.shape[1]].numpy()

            else:
                _color_ = torch.tensor([1.0, 0.0, 1.0, 1.0])
             
            spheres = Spheres(positions=loc * mask, 
                                radius=0.007,
                                color=_color_)

            
            scene_elements.append(spheres)
            self.renderer.scene.add(spheres)
            
        if camera_dict != {}:
            R_z = get_z_rot(camera_dict['camera_rot'][0], in_format='aa')
            heading = -R_z[:, 1]
            
            xy_facing = camera_dict['camera_transl'][0] + heading * camera_dict['coef']
            
            if 'lock2object' in camera_dict.keys():
                focus_elem = obj_mesh 
                rel_pos = (xy_facing[0], xy_facing[1], 0.15)
            else:
                focus_elem = scene_elements[0]
                rel_pos = (xy_facing[0], xy_facing[1], rendering_scale)
                

            # override rel pos if specified
            if 'rel_pos' in camera_dict.keys():
                rel_pos = camera_dict['rel_pos']
               
            camera = self.renderer.lock_to_node(focus_elem, rel_pos, smooth_sigma=5.0)

        else:
            camera = self.renderer.lock_to_node(scene_elements[0],
                                    (0, 0, 2.5), smooth_sigma=5.0)

        self.renderer.save_video(video_dir=str(filename), output_fps=output_fps)
        sfx = 'mp4'
        os.rename(str(filename) + f'_0.{sfx}', str(filename) + f'.{sfx}')
            
 
        # empty scene for the next rendering
        for mesh in scene_elements:
            self.renderer.scene.remove(mesh)
        
        self.renderer.scene.remove(camera)
        
        fname = f'{filename}.{sfx}'


        if text_for_vid != '':

            put_text(text_for_vid, fname, f'{filename}.{sfx}')

            if fname != f'{filename}.{sfx}':
                os.remove(f'{filename}.{sfx}')
        
        return fname

 