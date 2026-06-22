import os
import sys
import torch
import smplx
import pickle
import joblib 
import trimesh
import numpy as np
import torch.nn as nn 
import torch.nn.functional as F 

import chamfer_distance as chd
from pytorch3d.structures import Meshes
import mesh_intersection.loss as collisions_loss
from mesh_intersection.bvh_search_tree import BVH

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.process_utils import SMPLX_JOINTS, CONTACT_TOE_HEIGHT_THRESH, CONTACT_INDICES, GLOBAL_WRIST_INDICES


CONTACT_CONF_THRESH = 0.8



def point2point_signed(
        x,
        y,
        x_normals=None,
        y_normals=None,
):
    """
    signed distance between two pointclouds

    Args:
        x: FloatTensor of shape (N, P1, D) representing a batch of point clouds
            with P1 points in each batch element, batch size N and feature
            dimension D.
        y: FloatTensor of shape (N, P2, D) representing a batch of point clouds
            with P2 points in each batch element, batch size N and feature
            dimension D.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).

    Returns:

        - y2x_signed: Torch.Tensor
            the sign distance from y to x
        - y2x_signed: Torch.Tensor
            the sign distance from y to x
        - yidx_near: Torch.tensor
            the indices of x vertices closest to y

    """
  
    N, P1, D = x.shape
    P2 = y.shape[1]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")

    ch_dist = chd.ChamferDistance()

    x_near, y_near, xidx_near, yidx_near = ch_dist(x,y)

    xidx_near_expanded = xidx_near.view(N, P1, 1).expand(N, P1, D).to(torch.long)
    x_near = y.gather(1, xidx_near_expanded)

    yidx_near_expanded = yidx_near.view(N, P2, 1).expand(N, P2, D).to(torch.long)
    y_near = x.gather(1, yidx_near_expanded)

    x2y = x - x_near
    y2x = y - y_near
 

    if x_normals is not None:
        y_nn = x_normals.gather(1, yidx_near_expanded)
        # in_out = torch.bmm(y_nn.view(-1, 1, 3), y2x.view(-1, 3, 1)).view(N, -1).sign()
        in_out = torch.bmm(y_nn.view(-1, 1, 3), y2x.reshape(-1, 3).unsqueeze(2)).view(N, -1).sign()
        y2x_signed = y2x.norm(dim=2) * in_out

    else:
        y2x_signed = y2x.norm(dim=2)

    if y_normals is not None:
        x_nn = y_normals.gather(1, xidx_near_expanded)
        in_out_x = torch.bmm(x_nn.view(-1, 1, 3), x2y.view(-1, 3, 1)).view(N, -1).sign()
        x2y_signed = x2y.norm(dim=2) * in_out_x
    else:
        x2y_signed = x2y.norm(dim=2)

    return y2x_signed, x2y_signed, yidx_near


def batch_smoothness_loss(joints3d):
    '''
    Penalize discontinuities at the boundaries between consecutive batches.
    Compares the last frame of batch i with the first frame of batch i+1
    to encourage smooth transitions across batch boundaries.
    joints3d: [B, T, J, 3]
    
    Returns:
        Tensor of shape [B] with per-batch smoothness loss.
        Batches 0..B-2 get the boundary loss; the last batch gets 0.
    '''
    B, T, J, D = joints3d.shape
    device = joints3d.device

    if B < 2:
        return torch.zeros(B, device=device)

    # last frame of each batch except the last one: [B-1, J, 3]
    last_frames = joints3d[:-1, -1]
    # first frame of each batch except the first one: [B-1, J, 3]
    first_frames = joints3d[1:, 0]

    # L2 distance per joint, averaged over joints
    boundary_diff = (last_frames - first_frames) ** 2  # [B-1, J, 3]
    boundary_loss = boundary_diff.sum(dim=-1).sqrt().mean(dim=-1)  # [B-1]

    # Also penalize velocity discontinuity at the boundary:
    # velocity at the end of batch i vs velocity at the start of batch i+1
    vel_end = joints3d[:-1, -1] - joints3d[:-1, -2]  # [B-1, J, 3]
    vel_start = joints3d[1:, 1] - joints3d[1:, 0]    # [B-1, J, 3]
    vel_diff = (vel_end - vel_start) ** 2  # [B-1, J, 3]
    vel_loss = vel_diff.sum(dim=-1).sqrt().mean(dim=-1)  # [B-1]

    combined = boundary_loss + vel_loss  # [B-1]

    # Pad to match batch size: last batch has no successor, gets 0
    loss = torch.zeros(B, device=device)
    loss[:-1] = combined

    return loss


def contact_vel_loss(joints3d, contacts_conf):
    '''
    Velocity should be zero at predicted contacts.
    joints3d: [B, T, J, 3]
    
    '''    
    delta_pos = (joints3d[:, 1:] - joints3d[:, :-1])**2
    cur_loss = delta_pos.sum(dim=-1) * contacts_conf[:, 1:]
    
    cur_loss = cur_loss.mean(dim=[1, 2])
    cur_loss = torch.sqrt(cur_loss)
    
    return cur_loss


def contact_height_loss(joints3d, contacts_conf):
    '''
    Contacting joints should be near floor.
    joints3d: [B, T, J, 3]
    
    ''' 
    
    # get z dim of joints
    stable_joints = torch.abs_(joints3d[:,:,:, 2])
    
    # won't be exactly on the floor, just near it (since joints are inside the body)
    floor_diff = F.relu(stable_joints - CONTACT_TOE_HEIGHT_THRESH)

    cur_loss = floor_diff * contacts_conf
        
    cur_loss = cur_loss.mean(dim=[1, 2])

    if torch.isnan(cur_loss).any().item():
        print(cur_loss)


    return cur_loss


def object_attendance_loss(joints3d, vertices_3d, obj_dict_list):
    '''
    Human should be looking at the object.
    joints3d: [B, T, J, 3]
    
    '''

    # find indices containing object
    idx = [i for i, obj_dict in enumerate(obj_dict_list) if obj_dict !={}]
 
    if len(idx) == 0:
        return torch.zeros(len(obj_dict_list)).to(joints3d.device)
    

    joints3d_subset = joints3d[idx]
    vertices_3d_subset = vertices_3d[idx]


    if 'trans' in obj_dict_list[0].keys():
        obj_centers = np.array([obj_dict_list[_idx_]['trans'] for _idx_ in idx])
        # obj_centers = np.vstack([obj_dict_list[_idx_]['trans'] for _idx_ in idx])
    else:    
        obj_centers = np.array([obj_dict_list[_idx_]['vertices'].mean(1) for _idx_ in idx])
    
    obj_centers = torch.tensor(obj_centers).to(joints3d.device)

    try:
        leye_pos = vertices_3d_subset[:, :, 9503]
        reye_pos = vertices_3d_subset[:, :, 10049]
    except:
        leye_pos = joints3d_subset[:, :, SMPLX_JOINTS['left_eye_smplhf']]
        reye_pos = joints3d_subset[:, :, SMPLX_JOINTS['right_eye_smplhf']]
 
    # find the eye 2 eye vector (B, T, 3)
    eye2eye_vector = leye_pos - reye_pos
    
    # find the vector from eyes to object (B, T, 3)
    eye2obj_vector = obj_centers - (leye_pos + reye_pos)/2 
           
    # those two vectors should be perpendicular, dot product should be 0
    epsilon = 1e-8

    eye2eye_len = torch.linalg.norm(eye2eye_vector, dim=-1, keepdim=True) + epsilon
    eye2obj_len = torch.linalg.norm(eye2obj_vector, dim=-1, keepdim=True) + epsilon
    
    # cosine similarity: dot product / (norm * norm) -> (B, T)
    cosine_sim = (eye2eye_vector * eye2obj_vector).sum(dim=-1) / (eye2eye_len.squeeze(-1) * eye2obj_len.squeeze(-1))

    # abs per timestep first, then average over time
    loss = torch.abs(cosine_sim).mean(dim=1)  # (B_subset,)

    # scatter back to full batch
    full_loss = torch.zeros(len(obj_dict_list), device=joints3d.device)
    full_loss[idx] = loss
    return full_loss

 

def self_contact_loss(vertices3d, contact_dict):
    """
    Computes self-contact loss to penalize when human body vertices are not touching.
    Args:
        vertices3d: Tensor of shape [B, T, V, 3]
        contact_dict: Dictionary with 'instruction_dict_list'
    Returns:
        Tensor of shape [B] with per-batch self-contact loss
    """
    B, T, V, _ = vertices3d.shape
    device = vertices3d.device
    contact_loss_sum = torch.zeros(B, device=device)


    for command_idx, command in enumerate(contact_dict['instruction_dict_list']):

        total_loss, command_count_idx = 0.0, 0 

        for keyf_idx, action_vertices in command['command_list']:

            if len(action_vertices) == 0:
                continue

            action_vertices = torch.tensor(action_vertices).to(vertices3d.device).long()
            
            # 1, T, ACT_VERTICES, 2, 3
            verts_subset = vertices3d[command_idx:command_idx+1, :, action_vertices] 
            
            verts_diff = verts_subset[:, :, :, :, None] - verts_subset[:, :, :, None, :]
            distances = torch.norm(verts_diff, dim=-1) # Shape: (B, T, ACT_VERTICES, 2, 2)

            # sum over all pairs of vertices
            contact_loss = distances.sum(dim=(-1, -2, -3)) / 2.0  # Shape: (B,)
            contact_loss = contact_loss[:, torch.arange(keyf_idx[0], keyf_idx[1])].mean(dim=1)  

                

            total_loss += contact_loss
            command_count_idx += len(action_vertices)
        
        # consider the case where there is no contact command. 
        contact_loss_sum[command_idx] = total_loss / command_count_idx if command_count_idx > 0 else 0.0
    
    return contact_loss_sum

 
def closeness_loss(loss_func, value, target, target_mask, impose_wrist_penalty=False):
 
    clo_loss = loss_func(value, target, reduction="none") * target_mask
    mask_sum = target_mask.sum(dim=[1, 2, 3])

    if impose_wrist_penalty:
        # wrists are important
        clo_loss[:, :, GLOBAL_WRIST_INDICES] *= 5

    clo_loss = clo_loss.sum(dim=[1, 2, 3]) / mask_sum
    
    # Optionally zero out loss where mask_sum is zero to avoid nan
    clo_loss = torch.where(mask_sum > 0, clo_loss, torch.zeros_like(clo_loss))

    return clo_loss

def get_contacts_from_output(output, add_fingertips_flag): 
    
    if add_fingertips_flag:
        J_DIM = len(SMPLX_JOINTS) + 10
    else:
        J_DIM = len(SMPLX_JOINTS) 
 
    B, T, _ = output['contact_masks'].shape

    contact_conf = torch.relu(output['contact_masks'])
    
    pred_contacts = (contact_conf > CONTACT_CONF_THRESH).to(torch.float)
    full_contact_conf = torch.zeros((B, T, J_DIM)).to(contact_conf)
    full_contact_conf[:,:, CONTACT_INDICES] = contact_conf # full_contact_conf[:,:,CONTACTS_IDX] + contact_conf
        
    full_contacts = torch.zeros((B, T, J_DIM)).to(pred_contacts)
    full_contacts[:,:, CONTACT_INDICES] = pred_contacts # full_contacts[:,:,CONTACTS_IDX] + pred_contacts
    
    
    return full_contacts, full_contact_conf


def self_penetration_loss(mesh3d, penetration_dict, search_tree_obj, pen_distance):


    head_flag = True
    device = mesh3d['vertices'].device

    watertight_face_ids = torch.tensor(list(penetration_dict['base2watertight_faces_dict'].keys()), device=device)
    vertex_indices = torch.tensor(list(penetration_dict['watertight2base_w_hand'].values()), device=device)


    watertight_vertices = mesh3d['vertices']
    watertight_faces = mesh3d['faces']
   
    if not head_flag: 
        penetration_flags = ~np.array([trimesh.Trimesh(vertices=verts, faces=watertight_faces).is_watertight for verts in watertight_vertices.detach().cpu()])
        penetration_frames = penetration_flags.nonzero()[0]
        
        if len(penetration_frames) == 0:
            return torch.zeros(mesh3d['vertices'].shape[0], device=device)
        
    
    pen_loss_list = []
     
    vertices = watertight_vertices
    faces = watertight_faces

    bs, T, nv = vertices.shape[:3]

    # faces do not change over time
    # face_tensor = torch.tensor(faces.astype(np.int64), dtype=torch.long, 
    #                            device=device).unsqueeze_(0).unsqueeze_(0).repeat([bs, T, 1, 1])
    # faces_idx = face_tensor + \
    #     (torch.arange(bs, dtype=torch.long).to(device) * T * nv)[:, None, None, None]



    # face_tensor = torch.tensor(faces.astype(np.int64), dtype=torch.long, 
    #                            device=device).unsqueeze_(0).repeat([bs, 1, 1])
    # faces_idx = face_tensor + \
    #     (torch.arange(bs, dtype=torch.long).to(device) * T * nv)[:, None, None]
    

    face_tensor = torch.tensor(faces.astype(np.int64), dtype=torch.long, 
                               device=device).unsqueeze_(0).repeat([bs*T, 1, 1])
    
    faces_idx = face_tensor + \
        (torch.arange(bs*T, dtype=torch.long).to(device) * nv)[:, None, None]
    
    triangles = vertices.view([-1, 3])[faces_idx].cuda()
 
 
    with torch.no_grad():
   
        collision_idxs = search_tree_obj(triangles)
        collision_idxs_flag = torch.isin(collision_idxs, watertight_face_ids)

        collision_idxs_watertight = torch.where(collision_idxs_flag, collision_idxs, -1)
       
    pen_loss = pen_distance(triangles, collision_idxs_watertight).reshape([bs, T]).mean(1)
     
    return pen_loss


def penetration_loss(verts3d, obj_dict_list):
 
    faces_data = obj_dict_list[0]['body_faces'].astype(np.int64)

    # Then create the PyTorch tensor
    faces_tensor = torch.tensor(faces_data, dtype=torch.long)

    # Add batch dimension if needed
    if len(faces_tensor.shape) == 2:
        faces_tensor = faces_tensor.unsqueeze(0)

    # Now repeat it
    faces = torch.repeat_interleave(faces_tensor, repeats=120, dim=0)

    # Make sure it's on the same device as verts3d
    faces = faces.to(verts3d.device)


    # Now create the mesh
    rh_mesh = Meshes(verts=verts3d[0], faces=faces)\
        .to(verts3d.device).verts_normals_packed().view(-1, 10475, 3)
    
 
    y2x_signed, x2y_signed, yidx_near = point2point_signed(verts3d[0], 
                    torch.tensor(obj_dict_list[0]['vertices']).to(verts3d), 
                    rh_mesh)

    negative_y2x_signed = y2x_signed[y2x_signed<0]

    if len(negative_y2x_signed) == 0:
        pen_loss = torch.zeros(verts3d.shape[0]).to(verts3d)
    else:
        pen_loss = torch.abs_(negative_y2x_signed.mean(0)[None])

    return pen_loss


# see how far it deviates from the mean.
def diffusion_likelihood_loss(x, stats_dict):   

    if stats_dict is not None:
        STD_THRESHOLD = 1e-8
        std_vector = stats_dict['std'].detach()
        std_vector[std_vector < STD_THRESHOLD] = STD_THRESHOLD
    else:
        std_vector = 1

    nominator = x ** 2 
    denominator = 2 * (std_vector ** 2)

    likelihood_loss = 1 - torch.exp(- nominator/denominator).mean([1, 2])
 

    return likelihood_loss


class CondKeyLocationsLoss:
    def __init__(self,
                 target=None,
                 target_mask=None,
                 motion_length=None,
                 transform=None,
                 inv_transform=None,
                 abs_3d=False,
                 use_mse_loss=False,
                 obs_list=[],
                 penetration_dict={}
                 ):
        
        self.target_dict = target
        self.target_mask_dict = target_mask
        self.motion_length = motion_length
        self.transform = transform
        self.inv_transform = inv_transform

        self.abs_3d = abs_3d
        self.obs_list = obs_list

        self.penetration_dict = penetration_dict

        if self.penetration_dict:
 
            # Create the search tree
            self.search_tree = BVH(max_collisions=2)  

            self.pen_distance = collisions_loss.DistanceFieldPenetrationLoss(sigma=0.01,
                                                                    point2plane=False, 
                                                                    penalize_outside=False,
                                                                    vectorized=True)

     
        self.loss_fn = F.mse_loss if use_mse_loss else F.l1_loss
        self.gt_style = 'target'
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def __call__(self, xstart_in, z, obj_dict_list=None, self_contact_dict=None, stats_dict=None):
        """
        Args:
            xstart_in: [bs, D, 1, 120]
            target: [bs, 120, J, 3]
            target_mask: [bs, 120, J, 3]
        """
  
        bs = xstart_in.shape[0]
       

        with torch.enable_grad():
            
            x_pose_space_dict = self.inv_transform(xstart_in)  # [bs, 120, D]

            x_in_dict, _ = self.transform(x_pose_space_dict) 

            body_flag = 'joints' in x_in_dict.keys()

            if body_flag:
                x_in_joints = x_in_dict['joints'] 
                x_in_vertices = x_in_dict['vertices'] 
    
                _, contact_conf = get_contacts_from_output(x_pose_space_dict, self.transform.keywords['add_fingertips'])
            
                contact_v_loss = contact_vel_loss(x_in_joints, contact_conf)
                contact_h_loss = contact_height_loss(x_in_joints, contact_conf)
 
            else: 
                x_in_joints = torch.cat([x_in_dict['right']['joints'], x_in_dict['left']['joints']], dim=-2)
                
                contact_v_loss = torch.zeros(bs).to(x_in_joints.device)
                contact_h_loss = torch.zeros(bs).to(x_in_joints.device)
                
    
            if self.target_mask_dict['joints'].sum() > 0:
                # Assume the target has dimention [bs, T, J, 3] 
                jts_clo_loss = closeness_loss(self.loss_fn, 
                                              x_in_joints, 
                                              self.target_dict['joints'], 
                                              self.target_mask_dict['joints'], 
                                              impose_wrist_penalty=True)
                
            else:
                jts_clo_loss = torch.zeros(bs).to(x_in_joints.device)


            if self.target_mask_dict['vertices'].sum() > 0:
                vts_clo_loss = closeness_loss(self.loss_fn, 
                                              x_in_vertices, 
                                              self.target_dict['vertices'], 
                                              self.target_mask_dict['vertices'],
                                              impose_wrist_penalty=False)
                
            else:
                vts_clo_loss = torch.zeros(bs).to(x_in_joints.device)

            clo_loss = vts_clo_loss + jts_clo_loss 

            diff_loss = diffusion_likelihood_loss(z, stats_dict)
 
 
            if obj_dict_list:
                obj_attendance_loss = object_attendance_loss(x_in_joints, x_in_vertices, obj_dict_list)       
                  
                try:          
                    obj_penetration_loss = penetration_loss(x_in_vertices, obj_dict_list)
                except:
                    obj_penetration_loss = torch.zeros(bs).to(x_in_joints.device)        

            else:
                obj_attendance_loss = torch.zeros(bs).to(x_in_joints.device)
                obj_penetration_loss = torch.zeros(bs).to(x_in_joints.device)        
                

            if self_contact_dict is not None:
                self_pen_loss = self_penetration_loss({'vertices': x_in_vertices, 
                                                       'faces': self_contact_dict['faces']},
                                                       self.penetration_dict,
                                                       self.search_tree, 
                                                       self.pen_distance)
       
                contact_loss = self_contact_loss(x_in_vertices, self_contact_dict)                
            else:
                self_pen_loss = torch.zeros(bs).to(x_in_joints.device)
                contact_loss = torch.zeros(bs).to(x_in_joints.device)
           

            if torch.isnan(clo_loss).any().item():
                print(clo_loss)

            # Batch smoothness: penalize discontinuities at batch boundaries
            batch_smooth_loss = batch_smoothness_loss(x_in_joints)

            return {'closeness_loss': clo_loss,
                    'contact_v_loss': contact_v_loss,
                    'contact_h_loss': contact_h_loss,
                    'likelihood_loss': diff_loss,
                    'collision_loss':  0.0,
                    'object_attendance_loss': obj_attendance_loss,
                    'object_penetration_loss': obj_penetration_loss, 
                    'self_penetration_loss': self_pen_loss, 
                    'self_contact_loss': contact_loss,
                    'batch_smoothness_loss': batch_smooth_loss}
        
