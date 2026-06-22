import os
import sys
import bpy
import math 
import torch
import numpy as np
from tqdm import tqdm
from mathutils import Vector, Matrix, Euler

from .materials import colored_material
from .scene import setup_scene  # noqa
from .floor import show_traj, plot_floor, get_trajectory
from .vertices import prepare_vertices
from .tools import load_numpy_vertices_into_blender, delete_objs, mesh_detect
from .moving_camera import MovingCamera
from .sampler import get_frameidx
from .meshes import Meshes, prepare_meshes
from .tools import cleanup
from .geom_transform import rotate_vertices, rotate_for_side_view

from .geom_transform import rotate_vertices_around_center


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
   9: (0.282, 0.239, 0.545, 1.0),  # Dark Slate Blue
   10: (0.722, 0.525, 0.043, 1.0), # Dark Goldenrod
   11: (0.333, 0.420, 0.184, 1.0), # Dark Olive Green
   12: (0.804, 0.361, 0.361, 1.0), # Indian Red
   13: (0.627, 0.322, 0.176, 1.0), # Sienna
   14: (0.373, 0.620, 0.627, 1.0), # Cadet Blue
   15: (0.439, 0.502, 0.565, 1.0), # Slate Gray
   16: (0.737, 0.561, 0.561, 1.0), # Rosy Brown
   17: (0.184, 0.310, 0.310, 1.0), # Dark Slate Gray
   18: (0.502, 0.502, 0.502, 1.0), # Gray
   19: (0.412, 0.412, 0.412, 1.0), # Dim Gray
}


def prune_begin_end(data, perc):
    to_remove = int(len(data)*perc)
    if to_remove == 0:
        return data
    return data[to_remove:-to_remove]

def render_current_frame_(path):

    bpy.context.scene.render.filepath = path
    sys.stdout.flush()
    old = os.dup(1)
    os.close(1)
    os.open(os.devnull, os.O_WRONLY)
    bpy.ops.render.render(use_viewport=True, write_still=True)
    sys.stdout.flush()
    os.close(1)
    os.dup(old)
    os.close(old)


def render_current_frame(path):

    bpy.context.scene.render.filepath = path
    sys.stdout.flush()

    # Redirect stdout to /dev/null
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stdout_fd = os.dup(1)
    os.dup2(devnull_fd, 1)  # Redirect fd 1 to /dev/null
    os.close(devnull_fd)

    # Perform rendering silently
    bpy.ops.render.render(use_viewport=True, write_still=True)

    # Restore original stdout
    sys.stdout.flush()
    os.dup2(saved_stdout_fd, 1)
    os.close(saved_stdout_fd)



def locate_action(idx, lens):
    cumsum = np.cumsum(lens)
    min_v = min(i for i in cumsum if i >= (idx+1))
    min_idx = list(cumsum).index(min_v)
    ind_norm = (idx - cumsum[min_idx-1]) if (idx - cumsum[min_idx-1]) >= 0 else idx
    return min_idx, ind_norm


# Additional utility function for fine-tuning camera position
def adjust_camera_for_scene_coverage(min_coords, max_coords, view_angle_deg=45, elevation_deg=30, distance_multiplier=1.5):
    """
    Calculate optimal camera position to frame all meshes
    
    Args:
        min_coords: (min_x, min_y, min_z) tuple
        max_coords: (max_x, max_y, max_z) tuple
        view_angle_deg: Horizontal viewing angle in degrees
        elevation_deg: Camera elevation angle in degrees
        distance_multiplier: Multiplier for camera distance
    
    Returns:
        Dictionary with camera parameters
    """
    min_x, min_y, min_z = min_coords
    max_x, max_y, max_z = max_coords
    
    center = Vector(((min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2))
    size = Vector((max_x - min_x, max_y - min_y, max_z - min_z))
    diagonal = size.length
    
    # Calculate camera distance to ensure all content is visible
    distance = diagonal * distance_multiplier
    
    # Convert angles to radians
    view_rad = math.radians(view_angle_deg)
    elev_rad = math.radians(elevation_deg)
    
    horizontal_distance = distance * math.cos(elev_rad)
    
    cam_x = center.x + horizontal_distance * math.cos(view_rad)
    cam_y = center.y + horizontal_distance * math.sin(view_rad)
    cam_z = center.z + distance * math.sin(elev_rad)  # Z is
    
    # Calculate appropriate FOV
    fov = math.degrees(2 * math.atan(diagonal / (2 * distance)))
    fov = max(30, min(120, fov))  # Reasonable FOV range
    
    return {
        'camera_locx': cam_x,
        'camera_locy': cam_y,
        'camera_locz': cam_z,
        'camera_fov': fov
    }

def render_images_blender(npydata, frames_folder, *, mode, faces,
                          npydata2=None,
                          gt=False,
                          exact_frame=None,
                          num=2,
                          color=None,
                          downsample=True,
                          canonicalize=True,
                          always_on_floor=False,
                          fake_translation=False,
                          render_engine=True, res='low',
                          init=True,
                          lengths=None,
                          separate_actions=True,
                          bp=False,
                          texture_path=None,
                          debug=False,
                          text=None,
                          cam=None,
                          sideview='front',
                          goals_tensor=None, 
                          computed_camera=None,
                          target_object=None,
                          floor_flag=True,
                          long_cam_flag=False):
    

    from .goal import Goals
    from .meshes import Meshes
    from .joints import Joints
    from .object_meshes import ObjectMeshes
    

    if init:
        # Setup the scene (lights / render engine / resolution etc)
        setup_scene(render_eng=render_engine, res=res,
                    sun_color=(0.362, 0.362, 0.362), 
                    sun_strength=4, 
                    sun_rotation=(-58, 28, -176),
                    sun_location=(1.4, 0, 5))
    
    # CREATE INVISIBLE ITEM FOR TRACKING 
    # Create an empty at the given location
    # bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0.6, 0, 0))
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0.0, 0, 0))
    target_obj = bpy.context.active_object
    target_obj.name = "Target"
    
    # Optional: hide it in the viewport
    target_obj.hide_viewport = True
    
    color_dict = {'blue':0, 'grey':1, 'purple':2, 'green':3, 'red':4}
    
    if npydata is None:
        exists_human = False
    else:
        exists_human = True

    if exists_human:
        is_mesh = mesh_detect(npydata)
    if lengths is not None and isinstance(lengths, list):
        num_of_actions = len(lengths)
    else:
        num_of_actions = 1
    
    # Put everything in this folder
    if mode == "video":

        fake_translation = False
        separate_actions = False
        if always_on_floor:
            frames_folder += "_onfloor"
        os.makedirs(frames_folder, exist_ok=True)

    elif mode == "sequence":
        img_name, ext = os.path.splitext(frames_folder)
        if always_on_floor:
            img_name += "_onfloor"
        img_path = f"{img_name}{ext}"
    elif mode == "frame":
        img_name, ext = os.path.splitext(frames_folder)
        if always_on_floor:
            img_name += "_onfloor"
        img_path = f"{img_name}_{exact_frame}{ext}"
    fake = 'fake' if fake_translation else ''
    if exists_human:
        nframes = len(npydata)
    else:
        nframes = lengths[0]
    
    actions_bodies = []

    # we want to sample only goal frames. Therefore we need to find the superset frames
    if goals_tensor is not None:
        superset_frames = goals_tensor.mean((1, 2)).nonzero()[0]
    else:
        superset_frames = None


    frameidxs = get_frameidx(mode=mode, 
                             nframes=nframes,
                             lengths=lengths,
                             return_lists=True,
                             frames_to_keep=num, 
                             exact_frame=exact_frame,
                             superset_frames=superset_frames)


    if color is None:
        action_id = 0
    elif isinstance(color, str):
        action_id = color
    else:
        action_id = color_dict[color]
    if exists_human:
        npydata_to_render = npydata.copy()
  
    if exists_human:
        
        if goals_tensor is not None:

            mask = (np.all(goals_tensor == 0, axis=-1))
            goals_tensor[mask] = -15 # out of camera

        
            cap_size1 = goals_tensor.shape[1]
            npydata_to_render = np.concatenate((npydata_to_render, goals_tensor), axis=1)
        else:
            cap_size1 = 0

        if target_object is not None:
            cap_size2 = target_object['verts'].shape[1]
            npydata_to_render = np.concatenate((npydata_to_render, target_object['verts']), axis=1)
        else:
            cap_size2 = 0
        
        if sideview == 'right':    
            npydata_to_render = rotate_vertices_around_center(npydata_to_render.copy(), 
                                                        (0, 0, 90))
        elif sideview == 'left':
            npydata_to_render = rotate_vertices_around_center(npydata_to_render.copy(), 
                                                        (0, 0, -90))
        elif sideview == 'front':
            npydata_to_render = rotate_vertices_around_center(npydata_to_render.copy(), 
                                                        (0, 0, 0))
        
        else:   
            npydata_to_render = rotate_vertices_around_center(npydata_to_render.copy(), 
                                                            (0, 0, -45))
            
        # LIFO order
        if target_object is not None:
            target_object['verts'] = npydata_to_render[:, -cap_size2:]
            npydata_to_render = npydata_to_render[:, :-cap_size2]
        if goals_tensor is not None:
            goals_tensor = npydata_to_render[:, -cap_size1:]
            npydata_to_render = npydata_to_render[:, :-cap_size1]
    
        

    if exists_human:
        for frameidx in frameidxs:
            action_bodies = npydata_to_render[frameidx]
            actions_bodies.append(action_bodies)

        all_actions_bodies = np.concatenate(actions_bodies)

    # single action sequence otherwise it should change
    if goals_tensor is not None:
        goals_tensor_to_render = goals_tensor[frameidxs[0]]
    
    

    if target_object is not None:
        target_object['verts'] = target_object['verts'][frameidxs[0]]


    if mode == "sequence":
        total_num_of_rendered_frames = num * num_of_actions
    else:
        total_num_of_rendered_frames = len(frameidxs[0])
 
    if mode == "sequence":
        if fake_translation:
            # center all of them, except in the gravity axis
            if exists_human:

                npydata_to_render[..., :2] -= npydata_to_render.mean(1)[:, None][..., :2]
                actions_bodies_transf = []
                factor = 1.5
                if not separate_actions:
                    factor=3.0 # 3.0 for 2 action sin data
                    shift_vals = factor * np.linspace(-total_num_of_rendered_frames/2, total_num_of_rendered_frames/2, total_num_of_rendered_frames)/total_num_of_rendered_frames
                    # shift_vals = factor * np.linspace(0, total_num_of_rendered_frames, total_num_of_rendered_frames)/total_num_of_rendered_frames
                    for action_bodies, shift in zip(actions_bodies, [shift_vals[num*idx:num*(idx+1)] for idx in range(num_of_actions)]):
                        # put the fake translation
                        # and still 0 for gravity axis
                        action_bodies += np.stack((shift[:, None], shift[:, None], 0 * shift[:, None]), axis=2)
                        actions_bodies_transf.append(action_bodies)
                else:
                    for action_bodies in zip(actions_bodies):
                        shift = factor * np.linspace(-num/2, num/2, num)/num
                        # put the fake translation
                        # and still 0 for gravity axis
                        action_bodies += np.stack((-shift[:, None], -shift[:, None], 0 * shift[:, None]), axis=2)
                        actions_bodies_transf.append(action_bodies)
        else:
            if exists_human:
                actions_bodies_transf = all_actions_bodies.reshape(num_of_actions, 
                                                                num, 
                                                                *all_actions_bodies.shape[1:])

    elif mode == 'video':
        if exists_human:
            actions_bodies_transf = actions_bodies


    if exists_human:
        all_actions_bodies = np.concatenate(actions_bodies_transf)
        mean_root_vector = all_actions_bodies.mean((0, 1))
        
        first_root = Vector(mean_root_vector)
 
        min_x, max_x = all_actions_bodies[:, :, 0].min(), all_actions_bodies[:, :, 0].max()
        min_y, max_y = all_actions_bodies[:, :, 1].min(), all_actions_bodies[:, :, 1].max()
        min_z, max_z = all_actions_bodies[:, :, 2].min(), all_actions_bodies[:, :, 2].max()
 
        # cam = adjust_camera_for_scene_coverage(min_coords=(min_x, min_y, min_z), 
        #                                  max_coords=(max_x, max_y, max_z),
        #                                  view_angle_deg=30,
        #                                  elevation_deg=20,
        #                                  distance_multiplier=1.5)
        
       


    else:
        first_root = Vector((0, 0, 0))

    
    if not separate_actions and mode == 'sequence':    
        if exists_human:
            if is_mesh:    
                data = Meshes(actions_bodies_transf, 
                                fixed_color=color, 
                                gt=gt, 
                                mode=mode,
                                faces=faces,
                                canonicalize=canonicalize,
                                always_on_floor=always_on_floor,
                                lengths=lengths,
                                action_id=action_id)
            else:
                
                data = Joints(actions_bodies_transf, gt=gt, mode=mode,
                canonicalize=canonicalize,
                always_on_floor=always_on_floor)

        if target_object is not None:

            data_object = ObjectMeshes(data=target_object['verts'],
                                       mode=mode, 
                                       faces=target_object['faces'],
                                       fixed_color='yellow',
                                       action_id=action_id,
                                       lengths=None)
  
        if exists_human:
            only_trans = all_actions_bodies.mean((0, 1))
        

        cam1 = {'camera_locx': -4.5, 'camera_locy':-2.3, 'camera_locz':6, 'camera_fov':110}
        cam2 = {'camera_locx': -4.5, 'camera_locy': -2.3, 'camera_locz':6, 'camera_fov':60}
    
        if cam is None:
            cam = cam2 if long_cam_flag else cam1
        
 
        camera = MovingCamera(
            first_root=first_root,
            mode="sequence",
            is_mesh=True,
            smooth_factor=5.0, 
            use_constraints=True,
            **cam)
        
 

    elif mode == 'video':
        if exists_human: 
            only_trans = all_actions_bodies.mean((0, 1))
        else:
            only_trans = np.array([0,0,0])
        camera = MovingCamera(
            first_root=Vector((0, 0, 0)),
            mode="sequence",
            is_mesh=True,
            smooth_factor=5.0,
            use_constraints=True,
            camera_locx=-4.5,
            camera_locy=-2.3,
            camera_locz=6,
            camera_fov=110,
        )

    img_paths = []
    imported_obj_names = []
    lengths_cum = np.cumsum(lengths)

    if not exists_human:
        actions_bodies_transf = [np.zeros((lengths[0], 10475, 3))]
 
    for action_id, action_bodies in enumerate(actions_bodies_transf):
        
        # if exists_human:
        #     action_bodies = np.squeeze(action_bodies)

        print('Rendering this many', num)
        if goals_tensor is not None:
 
            remainder_tensor = torch.remainder(torch.arange(goals_tensor_to_render.shape[1]), max(list(colors_dict.keys())) + 1)
            _color_ = torch.tensor([colors_dict[remain.item()] for remain in remainder_tensor])

            # make it double so that targets match in color.
            _color_ = _color_.repeat_interleave(2, dim=0)[:goals_tensor_to_render.shape[1]].numpy()

            data_goals = Goals(goals_tensor_to_render, 
                               mode=mode, 
                               radius=0.010,
                               num_of_frames_to_rend=num, 
                               fixed_color='green',
                               mat = None if text == '' else _color_
                               )
            

            
        if target_object is not None:
            data_object = ObjectMeshes(data=target_object['verts'],
                                       mode=mode, 
                                       faces=target_object['faces'],
                                       fixed_color='yellow',
                                       action_id=action_id,
                                       lengths=None)
        if exists_human:
            if is_mesh:
                data = Meshes(action_bodies, fixed_color=color,
                            gt=gt, mode=mode,
                            faces=faces,
                            canonicalize=canonicalize,
                            always_on_floor=always_on_floor,
                            lengths=lengths,
                            bp=bp,
                            action_id=action_id)

            else:
                # TODO maybe need an update
                data = Joints(action_bodies, 
                            gt=gt, 
                            mode=mode,
                            canonicalize=canonicalize,
                            always_on_floor=always_on_floor)
        
        # show_traj(data.trajectory)
   
        if exists_human and floor_flag:
            plot_floor(action_bodies, color_alpha=None, 
                        texture_path=texture_path, 
                        rotation=47.2,
                        color1=(0.29, 0.29, 0.29, 1),
                        scale_of_tiles=34.0)
        elif not exists_human and floor_flag:
            plot_floor([0.0, 0.0, 0.0], color_alpha=None, 
                        texture_path=texture_path,
                        rotation=47.2,
                        color1=(0.29, 0.29, 0.29, 1),       
                        scale_of_tiles=34.0)
        else:
            # transparent background
            bpy.context.scene.render.film_transparent = True    


        if separate_actions and mode =='sequence' and floor_flag:

            if exists_human:
                plot_floor([0.0, 0.0, 0.0], color_alpha=None, 
                        texture_path=texture_path,
                        rotation=47.2,
                        color1=(0.29, 0.29, 0.29, 1),
                        scale_of_tiles=14.0)
            else:
                plot_floor([0.0, 0.0, 0.0], color_alpha=None, 
                        texture_path=texture_path,
                        rotation=47.2,
                        color1=(0.29, 0.29, 0.29, 1),
                        scale_of_tiles=14.0)
                
            camera = Camera(first_root=action_bodies.mean((0, 1)), 
                            mode=mode, is_mesh=is_mesh)
            imported_obj_names = []
 
        # TODO camera
        # camera.update(data.get_mean_root())
        # camera.update(npydata[npydata.shape[0]//2].mean(0)+camera._root)


        # render the frames within an action
        number_of_single_action_loops = num if mode == 'sequence' else lengths[action_id]

        for idx in tqdm(range(number_of_single_action_loops)):
            
            islast_within_action = idx == num-1

            frac = idx / (num-1) if num > 1 else 0.5

            if exists_human:
                mat = data.get_sequence_mat(frac)
                objname = data.load_in_blender(idx, mat)

            if goals_tensor is not None:
                objname_goals = data_goals.load_in_blender(idx, use_mat=text is not '')
 
            if target_object is not None:
                mat_object = data_object.get_sequence_mat(frac)
                objname_object = data_object.load_in_blender(idx, mat_object)


            if mode == "video":
                if action_id == 0:
                    name = f"{str(idx).zfill(4)}"
                else:
                    name = f"{str(idx+lengths_cum[action_id-1]).zfill(4)}"
                path = os.path.join(frames_folder, f"frame_{name}.png")
            else:
                name = f"{str(idx).zfill(4)}_{action_id}"
                path = img_path

            if mode == "sequence":
                if exists_human:
                    imported_obj_names.append(objname)
                if goals_tensor is not None:
                    imported_obj_names.append(objname_goals)
                if target_object is not None:
                    imported_obj_names.append(objname_object)

            elif mode == "frame":
                camera.update(data.get_root(frameidx))
            debug_mode = debug
            
            # UPDATE THE CAMERA
            bpy.data.objects['Target'].location = (2.0, 3.0, 1.5)
            camera.update_heading_target(computed_camera)

            

            if mode == 'video' or mode == 'frame': 
                render_current_frame(path)
            
                if debug_mode:
                    if 'frame_0040.png' in path:
                        fn = frames_folder.split('/')[-1]
                        fdn = '/'.join(frames_folder.split('/')[:-1])
                        absp = os.path.abspath(f"{fdn}/{fn.replace('.png', '')}_scene.blend")
                        bpy.ops.wm.save_as_mainfile(filepath=absp)
                        delete_objs(objname)
                        if goals_tensor is not None:
                            delete_objs(objname_goals)
                        delete_objs(["Plane", "myCurve", "Cylinder"])

                        cleanup()
                        return frames_folder
                
                if exists_human:
                    delete_objs(objname)
                if goals_tensor is not None:
                    delete_objs(objname_goals)
                if target_object is not None:
                    delete_objs(objname_object)
                cleanup()

            elif mode == 'sequence':
                if debug_mode:
                    fn = frames_folder.split('/')[-1]
                    fdn = '/'.join(frames_folder.split('/')[:-1])
                    absp = os.path.abspath(f"{fdn}/{fn.replace('.png', '')}_scene.blend")
                    bpy.ops.wm.save_as_mainfile(filepath=absp)

                if separate_actions and islast_within_action:
                    path = path.replace('.png', '')
                    img_paths.append(f'{path}_{action_id}{fake}.png')
                    render_current_frame(f'{path}_{action_id}{fake}.png')
                    # keeping blender file for loading later
                    fn = frames_folder.split('/')[-1]
                    fdn = '/'.join(frames_folder.split('/')[:-1])
                    bpy.ops.wm.save_as_mainfile(filepath=f"{fdn}/{fn.replace('.png', '')}_scene.blend")
                    delete_objs(imported_obj_names)
                    delete_objs(["Plane", "myCurve", "Cylinder"])
     
     
    
    # render in the end only
    if not separate_actions and mode == 'sequence':
        path = path.replace('.png', '')
        img_paths.append(f'{path}_all{fake}.png')
        render_current_frame(f'{path}_all{fake}.png')
    fn = frames_folder.split('/')[-1]
    fdn = '/'.join(frames_folder.split('/')[:-1])

    delete_objs(["Plane", "myCurve", "Cylinder"])
    cleanup()

    if mode == "video":
        return frames_folder
    else:
        return img_paths
