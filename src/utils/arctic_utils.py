import json
import os.path as op
import sys
import os 
import numpy as np
import torch
import torch.nn as nn
import trimesh
from easydict import EasyDict
from scipy.spatial.distance import cdist

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import utils.arctic.thing as thing 
from utils.arctic.xdict import xdict
from utils.transforms import axis_angle_to_quaternion, quaternion_apply, axis_angle_to_matrix


def pad_tensor_list(v_list: list):
    dev = v_list[0].device
    num_meshes = len(v_list)
    num_dim = 1 if len(v_list[0].shape) == 1 else v_list[0].shape[1]
    v_len_list = []
    for verts in v_list:
        v_len_list.append(verts.shape[0])

    pad_len = max(v_len_list)
    dtype = v_list[0].dtype
    if num_dim == 1:
        padded_tensor = torch.zeros(num_meshes, pad_len, dtype=dtype)
    else:
        padded_tensor = torch.zeros(num_meshes, pad_len, num_dim, dtype=dtype)
    for idx, (verts, v_len) in enumerate(zip(v_list, v_len_list)):
        padded_tensor[idx, :v_len] = verts
    padded_tensor = padded_tensor.to(dev)
    v_len_list = torch.LongTensor(v_len_list).to(dev)
    return padded_tensor, v_len_list


# objects to consider for training so far
OBJECTS = ["capsulemachine",
            "box",
            "ketchup",
            "laptop",
            "microwave",
            "mixer",
            "notebook",
            "espressomachine",
            "waffleiron",
            "scissors",
            "phone"]

SCALE = 1000
ARCTIC_PATH = 'data/motion/Hand_Raw/ARCTIC/data/arctic_data/data/meta'


class ObjectTensors(nn.Module):
    def __init__(self, load_simplified_flag=False):
        super(ObjectTensors, self).__init__()
        self.load_simplified_flag = load_simplified_flag
        self.obj_tensors = thing.thing2dev(self.construct_obj_tensors(OBJECTS, load_simplified_flag), "cpu")
        
        self.bottom_shift_dict = {}
        self.non_shifted_obj_tensors = thing.thing2dev(self.construct_obj_tensors(OBJECTS, False), "cpu")
    
    
        for obj_name in OBJECTS:    
            idx = self.non_shifted_obj_tensors['names'].index(obj_name)
            
            bottom_idx = np.where(self.non_shifted_obj_tensors['parts_ids'][idx] == 2)[0]            
            self.bottom_shift_dict[obj_name] = self.non_shifted_obj_tensors['v'][idx][bottom_idx].mean(0)
             
         

    def forward(self,
                angles,
                global_orient,
                transl,
                query_names,
                fwd_template=False,
                obj_com_local=None):
    
        # store output
        out = xdict()
        assert len(np.unique(query_names)) == 1

        # meta info
        obj_idx = np.array([self.obj_tensors["names"].index(name) for name in query_names])
        
        out["f"] = self.obj_tensors["f"][obj_idx]
        out["f_len"] = self.obj_tensors["f_len"][obj_idx]
        out["v_len"] = self.obj_tensors["v_len"][obj_idx]

        max_len = out["v_len"].max()
        out["v"] = self.obj_tensors["v"][obj_idx][:, :max_len]
        out["mask"] = self.obj_tensors["mask"][obj_idx][:, :max_len]
      
        out["parts_ids"] = self.obj_tensors["parts_ids"][obj_idx][:, :max_len]
      
        # prep output
        top_idx = out["parts_ids"] == 1
        bottom_idx = out["parts_ids"] == 2

        out['merged_com'] = out["v"][0].mean(0)
        out['top_com'] = out["v"][0][top_idx[0]].mean(0)
        out['bottom_com'] = out["v"][0][bottom_idx[0]].mean(0)

        out['v_template'] = out["v"][0]
       
        if fwd_template:
            return out

        # articulation + global rotation
        quat_arti = axis_angle_to_quaternion(self.obj_tensors["z_axis"] * angles)
        quat_global = axis_angle_to_quaternion(global_orient.view(-1, 3))
        matrot_arti = axis_angle_to_matrix(self.obj_tensors["z_axis"] * angles)
        matrot_global = axis_angle_to_matrix(global_orient.view(-1, 3))
        
 
        # collect entities to be transformed
        tf_dict = xdict()
        tf_dict["v_top"] = out["v"].clone()
        tf_dict["v_bottom"] = out["v"].clone()
        
        # articulate top parts (rotate around origin — the articulation pivot)
        for key, val in tf_dict.items():
            if "top" in key:
                val_rot = quaternion_apply(quat_arti[:, None, :], val)
                tf_dict.overwrite(key, val_rot)
                
        # global rotation for all
        # When obj_com_local is provided the rotation is applied around the COM
        # and transl is the world-space COM position, matching the convention
        # produced by _canonica_facefront when obj_com_local is supplied.
        for key, val in tf_dict.items():
            if obj_com_local is not None:
                com_l = obj_com_local.to(val.device)
                val_rot = quaternion_apply(quat_global[:, None, :], val - com_l[None, None, :])
            else:
                val_rot = quaternion_apply(quat_global[:, None, :], val)

            if transl is not None:
                val_rot = val_rot + transl[:, None, :]
            tf_dict.overwrite(key, val_rot)
     
        # T, V, 3
        v_tensor = tf_dict["v_bottom"].clone()
        v_tensor[:, top_idx[0], :] = tf_dict["v_top"][:, top_idx[0], :]

        out.overwrite("v", v_tensor)
      
        return out


    def construct_obj(self, object_model_p):
        
        # load vtemplate
        mesh_p = op.join(object_model_p, "mesh.obj")
        parts_p = op.join(object_model_p, f"parts.json") 
    
        assert op.exists(parts_p), f"Not found: {mesh_p}"
        assert op.exists(mesh_p), f"Not found: {mesh_p}"
        
        with open(parts_p, "r") as f:
            parts = np.array(json.load(f), dtype=np.bool_)


        mesh = trimesh.load(mesh_p, process=False)
         
        mesh_v = mesh.vertices
        mesh_f = torch.LongTensor(mesh.faces)
        
        vsk = object_model_p.split("/")[-1]
    
        obj = EasyDict()
        obj.name = vsk
        obj.obj_name = "".join([i for i in vsk if not i.isdigit()])
        obj.v = torch.FloatTensor(mesh_v)
        obj.f = torch.LongTensor(mesh_f)
        obj.parts = torch.LongTensor(parts)
        
        return obj


    def construct_obj_tensors(self, object_names, load_simplified_flag):
        
        obj_list = []
        for k in object_names:
            if load_simplified_flag:
                object_model_p = f"{ARCTIC_PATH}/object_vtemplates_shifted/%s" % (k)           
            else:
                object_model_p = f"{ARCTIC_PATH}/object_vtemplates/%s" % (k)

            obj = self.construct_obj(object_model_p)
            obj_list.append(obj)

        v_list, f_list, parts_list = [], [], []
        
 
        for obj in obj_list:
            v_list.append(obj.v)
            f_list.append(obj.f)
            parts_list.append(obj.parts + 1)
        
        v_list, v_len_list = pad_tensor_list(v_list)
        p_list, p_len_list = pad_tensor_list(parts_list)

        max_len = v_len_list.max()
        mask = torch.zeros(len(obj_list), max_len)
        
        for idx, vlen in enumerate(v_len_list):
            mask[idx, :vlen] = 1.0

        f_list, f_len_list = pad_tensor_list(f_list)
    
        obj_tensors = {}
        obj_tensors["names"] = object_names
        obj_tensors["parts_ids"] = p_list

        scale = 1 if load_simplified_flag else 1 

        obj_tensors["v"] = v_list.float() / scale
 
        obj_tensors["v_len"] = v_len_list
        obj_tensors["f"] = f_list
        obj_tensors["f_len"] = f_len_list
        obj_tensors["mask"] = mask
        obj_tensors["z_axis"] = torch.FloatTensor(np.array([0, 0, -1])).view(1, 3)

    
        return obj_tensors
