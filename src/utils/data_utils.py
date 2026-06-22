 
import os 
import torch
import trimesh
import numpy as np 
from typing import List 
from multiprocessing import Pool
from src.utils.arctic_utils import ObjectTensors
from scipy.spatial.transform import Rotation, Slerp


def apply_transformation_to_obj_geometry(obj_mesh_path, obj_scale, obj_rot, obj_trans,
                                         reflect_x=False):
        mesh = trimesh.load_mesh(obj_mesh_path)
        obj_mesh_verts = np.asarray(mesh.vertices) # Nv X 3
        obj_mesh_faces = np.asarray(mesh.faces) # Nf X 3

        if reflect_x:
            obj_mesh_verts = obj_mesh_verts.copy()
            obj_mesh_verts[:, 0] *= -1

        ori_obj_verts = torch.from_numpy(obj_mesh_verts).float()[None].repeat(obj_trans.shape[0], 1, 1) # T X Nv X 3
  
        seq_scale = obj_scale.float() # T 
        seq_rot_mat = obj_rot.float()  # T X 3 X 3 
        if obj_trans.shape[-1] != 1:     
            seq_trans = obj_trans.float()[:, :, None] # T X 3 X 1 
        else:
            seq_trans = obj_trans.float() # T X 3 X 1 
  
        transformed_obj_verts = seq_scale.unsqueeze(-1).unsqueeze(-1) * \
        seq_rot_mat.bmm(ori_obj_verts.transpose(1, 2)) + seq_trans
        transformed_obj_verts = transformed_obj_verts.transpose(1, 2) # T X Nv X 3 

        return transformed_obj_verts, obj_mesh_faces


arctic_obj_forward_simplify = ObjectTensors(load_simplified_flag=True).forward
arctic_obj_forward_raw = ObjectTensors(load_simplified_flag=False).forward

def merge_two_parts(verts_list, faces_list):
    verts_num = 0
    merged_verts_list = []
    merged_faces_list = []
    for p_idx in range(len(verts_list)):
        # part_verts = torch.from_numpy(verts_list[p_idx]) # T X Nv X 3 
        part_verts = verts_list[p_idx] # T X Nv X 3 
        part_faces = torch.from_numpy(faces_list[p_idx]) # T X Nf X 3 

        if p_idx == 0:
            merged_verts_list.append(part_verts)
            merged_faces_list.append(part_faces)
        else:
            merged_verts_list.append(part_verts)
            merged_faces_list.append(part_faces+verts_num)

        verts_num += part_verts.shape[1] 

    # merged_verts = torch.cat(merged_verts_list, dim=1).data.cpu().numpy()
    merged_verts = torch.cat(merged_verts_list, dim=1)
    merged_faces = torch.cat(merged_faces_list, dim=0).data.cpu().numpy() 

    return merged_verts, merged_faces 

    
def check_watertight_trimesh(vertices_single, faces):
    mesh = trimesh.Trimesh(vertices=vertices_single, faces=faces, process=False)
    return not mesh.is_watertight

def batch_check_watertight(vertices_batch, faces, num_workers=64):
    args = [(vertices_batch[i], faces) for i in range(vertices_batch.shape[0])]
    with Pool(num_workers) as pool:
        results = pool.starmap(check_watertight_trimesh, args)
    return np.array(results)  # shape (N,), bool array

 
def concatenate_data(data_dict_list: List[dict]):
    """
    Concatenate a list of dictionaries to a single dictionary
    
    Parameters
    ----------
    data_dict_list : list of dict
        List of dictionaries of tensors to be concatenated.  
    Returns
    -------
    data_dict : dict
        A dictionary of tensors where each value is the concatenation of the
        values in the input dictionaries.
    """
    concat_dict = {}
 
    for dict_i in data_dict_list:
        concat_dict.update(dict_i)

    return concat_dict

 
def slerp(quat, trans, key_times, times, mask=True):
    """
    Args:
        quat: (T x J x 4)
        trans: (T x 3)
    """

    if mask:
        quat = quat[key_times].detach().cpu().numpy()
        trans = trans[key_times].detach().cpu().numpy()
    else:
        quat = quat.detach().cpu().numpy()
        trans = trans.detach().cpu().numpy()

    quats = []

    for j in range(quat.shape[1]):
        key_rots = Rotation.from_quat(quat[:, j])
        # Times of the known rotations. At least 2 times must be specified.
        # Rotations to perform the interpolation between
        s = Slerp(key_times, key_rots)
        interp_rots = s(times)
        quats.append(interp_rots.as_quat())

    slerp_quat = np.stack(quats, axis=1)
    lerp_trans = np.zeros((len(times), 3))

    for i in range(3):
        lerp_trans[:, i] = np.interp(times, key_times, trans[:, i])

    return torch.tensor(slerp_quat).float(), \
                torch.tensor(lerp_trans).float()
    