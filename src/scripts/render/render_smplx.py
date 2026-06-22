import os 
import sys
import smplx 

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.utils.llm_utils import smplx_vertex_hands_ids

num_betas = 16
WINDOW = 120


color1 = (160 / 255, 160 / 255, 0 / 255, 1.0)
color2 = (0 / 255, 160 / 255, 160 / 255, 1.0)
color3 = (160 / 255, 0 / 255, 160 / 255, 1.0)



def estimate_grasp_positions(vertices):
    object_center = (0.51, -1.25, 0.84)
    object_width = 1.77
    object_height = 0.36
    object_depth = 0.36

    grasp_positions = {}

    # Calculate offsets based on the object dimensions
    palm_y_offset = object_width / 2  # Assuming palms are grasping from the sides
    finger_y_offset = object_width / 2 + object_depth  # Fingers curl around the object

    # Assign positions to vertices
    for vertex, index in vertices.items():
        name_parts = vertex.split('_')
        hand = name_parts[0]
        segment = name_parts[2]
        vertex_id = int(name_parts[3])

        x = object_center[0]
        y = object_center[1]
        z = object_center[2]

        # Adjustments based on whether it is left or right hand
        if 'left' in hand:
            y_direction = 1
        else:
            y_direction = -1

        # Adjust palm and finger positions
        if segment == 'palm':
            y += y_direction * palm_y_offset * (vertex_id * 0.1)  # Expand palm aligning along width
            x -= (vertex_id - 2.5) * 0.02  # Spread across X slightly based on vertex ID
        else:
            # Assume vertex_id 1 is base of the finger and 6 is tip
            z -= object_height / 2 * (vertex_id / 6.0)  # Move fingers from base to tip
            y += y_direction * finger_y_offset  # Push around the object

        # Place positions in dictionary
        grasp_positions[vertex] = [x, y, z]

    return grasp_positions
 
import trimesh
import scenepic as sp
import numpy as np 
import joblib
import torch
from typing import List, Any, Union
from tqdm import tqdm
from PIL import ImageColor

class HTMLRenderer(object):
    def __init__(self, width: int =1200, height: int = 800, wandb_flag: bool = False, save_html: bool = True, 
                 html_name: str ="unnamed", wandb_project_name: str = "unnamed",  wandb_note: str = None, 
                 wandb_scene_title:str = "", wandb_logs: dict = {}, caller_func: str = "", skel_color: np.array = np.array([1., 0., 0.]), wandb_obj=None): 
        super().__init__()

        self._width = width
        self._height = height
        
        self.skeleton_color = skel_color
        self.wandb_scene_title = wandb_scene_title
        self.wandb_note = wandb_note
        self.wandb_logs = wandb_logs
        self.mocap_html = html_name
        self.save_html_flag = save_html
        self.save_wandb = wandb_flag
        self.wandb_project_name = wandb_project_name
        self.simplify_mesh_flag = True
        self.caller_func = caller_func
        self.wrist_scale = 0.03
        self.fingertip_scale = 0.01

        self.f_decimate = None
 
    def __call__(self, body_output_list: List, 
                 object_output_list: List,
                 camera_rotation: torch.tensor,  
                 camera_translation: np.array = np.array([0, 0, 0]), 
                 focus_points: Union[Any, np.array] = None, 
                 timestep: int = 120,
                 show_ground_floor: bool = True,
                 show_coordinate_system: bool = True,
                 ) -> Any:
    
       
        scene_elements_list = [body_output_list, object_output_list]
     
        for scene_elements in scene_elements_list:    
            # Convert to numpy in case the vertices & faces are tensors.
            for elem in scene_elements:
      
                if "v" in elem.keys():    
                    if torch.is_tensor(elem["v"]):
                        elem["v"] = elem["v"].detach().cpu().numpy().astype(float) 
                if "f" in elem.keys():
                    if torch.is_tensor(elem["f"]):
                        elem["f"] = elem["f"].cpu().numpy().astype(np.uint32)
                if "right_finger_tips" in elem.keys():
                    if torch.is_tensor(elem["right_finger_tips"]):
                        elem["right_finger_tips"] = elem["right_finger_tips"].detach().cpu().numpy().astype(float)
                        
        if torch.is_tensor(camera_rotation):
            camera_rotation = camera_rotation.detach().cpu().numpy()
        
        self.scene = sp.Scene()
      
        canvas = self.scene.create_canvas_3d(width=self._width, height=self._height)

        layer_settings = {}
 
        for mii in tqdm(range(timestep)):

            focus_p = sp.FocusPoint(focus_points[mii]) if focus_points is not None else sp.FocusPoint([0.0, 0.0, 0.0])

            next_frame = canvas.create_frame(focus_point=focus_p)
 
            if show_ground_floor:
                next_frame.add_mesh(self.make_checkerboard_texture())
            
            for element_type in scene_elements_list:
                
                mesh_num = len(element_type)
                            
                # there may be multiple meshes for the same body part
                for diff_parts in range(mesh_num):
                    layer_settings[element_type[diff_parts]["mesh_name"]] = {"opacity": 1.0}

                    mesh = self.scene.create_mesh(shared_color=element_type[diff_parts]["mesh_color"], layer_id=element_type[diff_parts]["mesh_name"])

    
                    if "fingers" in element_type[diff_parts]['mesh_name']:

                        # add for all tips (thumb to pinky)
                        for _i_ in range(element_type[diff_parts]["v"].shape[1]):

                            R_rot = sp.Transforms.translate(element_type[diff_parts]["v"][mii, _i_]) 
  
                            R_rot[0, 0] = self.fingertip_scale
                            R_rot[1, 1] = self.fingertip_scale
                            R_rot[2, 2] = self.fingertip_scale
                            R_rot[3, 3] = self.fingertip_scale

                            mesh.add_sphere(transform=R_rot)
                  
                    else:
                        # we need to have tensors/arrays stored in cpu
                        mesh.add_mesh_without_normals(vertices=element_type[diff_parts]["v"][mii, :], triangles=element_type[diff_parts]["f"])
         
                    next_frame.add_mesh(mesh)
                     
            coord = self.scene.create_mesh(layer_id='camera')

            scale = 0.4
            matrix = scale * np.eye(4, 4)

            if camera_translation.any():
                matrix[:3, 3] = object_output_list[0]['obj_trans'][mii]
 
            if show_coordinate_system:
                coord.add_coordinate_axes(transform=matrix)

            camera_pose = np.eye(4)
            camera_pose[:3, :3] = camera_rotation

            coord.apply_transform(camera_pose)
            next_frame.add_mesh(coord)

        canvas.set_layer_settings(dict(camera={}, **layer_settings))    
 

        self.scene.save_as_html('obj_vis.html', title="processed mocap")

        return

     
    def make_checkerboard_texture(self, color1='gray', color2='white', width=1, height=1, n_tile=50):
        c1 = np.asarray(ImageColor.getcolor(color1, 'RGB')).astype(np.uint8)
        c2 = np.asarray(ImageColor.getcolor(color2, 'RGB')).astype(np.uint8)
        hw = width
        hh = height
        c1_block = np.tile(c1, (hh, hw, 1))
        c2_block = np.tile(c2, (hh, hw, 1))
        tex = np.block([
            [[c1_block], [c2_block]],
            [[c2_block], [c1_block]]
        ])
        tex = np.tile(tex, (n_tile, n_tile, 1))

        # image and texture id should be the same for matching
        floor_img = self.scene.create_image(image_id="ground")
        floor_img.from_numpy(tex)
        floor_mesh = self.scene.create_mesh(texture_id="ground", layer_id="floor")
        floor_mesh.add_image(transform=sp.Transforms.Scale(20.))

        return floor_mesh
     

obj_dict = joblib.load('nikos_llm/llm_trial_086641.tar')
 

positions = estimate_grasp_positions(smplx_vertex_hands_ids)
obj_mesh = trimesh.load('data/motion/OMOMO/captured_objects_simplified/clothesstand_cleaned_simplified.obj')




# object_output_list = [{'v': obj_mesh.vertices[None, ...],'f': obj_mesh.faces, 'mesh_name': 'obj', 'mesh_color': np.array([0.5, 0.5, 0.5])}]
object_output_list = [{'v': obj_dict['object_dict']['vertices'][0:1],'f': obj_mesh.faces, 'mesh_name': 'obj', 'mesh_color': np.array([0.5, 0.5, 0.5])}]
 
pos = torch.tensor(list(positions.values()))
 
body_output_list = [{'v': pos[None, ...], 'mesh_name': 'fingers', 
                     'mesh_color': np.array([0.1, 0.4, 0.3])}]



HTMLRenderer()(camera_rotation = torch.eye(3), object_output_list=object_output_list, body_output_list=body_output_list,
               timestep=1)