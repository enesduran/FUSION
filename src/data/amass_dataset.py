import os
import torch
import logging
import numpy as np
from typing import List
from os.path import join
from einops import rearrange
from smplx.joint_names import JOINT_NAMES

from torch.utils.data import Dataset
from src.utils.process_utils import SMPLX_JOINTS
from src.utils.genutils import DotDict, cast_dict_to_tensors, to_tensor
from src.utils.transforms import matrix_to_euler_angles, matrix_to_rotation_6d, axis_angle_to_matrix
from src.utils.data_utils import concatenate_data, arctic_obj_forward_simplify, arctic_obj_forward_raw, \
    apply_transformation_to_obj_geometry, merge_two_parts


# from pytorch_lightning import LightningDataModule
from src.utils.transforms3d import change_for, local_to_global_orient, transform_body_pose, remove_z_rot, rot_diff, \
    get_z_rot, canonicalize_rotations, matrix_to_axis_angle, rotate_trans

                
# A logger for this file
log = logging.getLogger(__name__)
 
SMPLX_BODY_CHAIN = [-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14,
 16, 17, 18, 19, 15, 15, 15, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34,
35, 20, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52, 53]


class AmassDataset(Dataset):
    def __init__(self, data: list,
                 n_body_joints: int,
                 stats_file: str,
                 norm_type: str,
                 rot_repr: str = "6d",
                 device: str = 'cpu',
                 object_dataset_list: List[str] = [],
                 load_feats: List[str] = None,
                 body_model_dict: dict = None):

        self.data = data
        self.device = device
        self.norm_type = norm_type
        self.rot_repr = rot_repr
        self.load_feats = load_feats
        self.object_dataset_list = object_dataset_list
        self.body_model_dict = body_model_dict
  
        self.body_chain = torch.tensor(SMPLX_BODY_CHAIN)
        stat_path = join(stats_file)
        self.stats = None
        self.n_body_joints = n_body_joints
        self.joint_idx = {name: i for i, name in enumerate(JOINT_NAMES)}
        
        if os.path.exists(stat_path):
            stats = np.load(stat_path, allow_pickle=True)[()]
            self.stats = cast_dict_to_tensors(stats)
        
        self._feat_get_methods = {
            "body_transl": self._get_body_transl,
            "body_transl_z": self._get_body_transl_z,
            "body_transl_delta": self._get_body_transl_delta,
            "body_transl_delta_pelv": self._get_body_transl_delta_pelv,
            "body_transl_delta_pelv_xy": self._get_body_transl_delta_pelv_xy,
            "body_transl_delta_pelv_xy_wo_z": self._get_body_transl_delta_pelv_xy_wo_z,
            "body_orient": self._get_body_orient,
            "body_orient_xy": self._get_body_orient_xy,
            "body_orient_delta": self._get_body_orient_delta,
            "z_orient_delta": self._get_z_orient_delta,
            "body_pose": self._get_body_pose,
            "body_pose_delta": self._get_body_pose_delta,

            "body_joints": self._get_body_joints,
            "body_joints_rel": self._get_body_joints_rel,
            "body_joints_local_wo_z_rot": self._get_body_joints_local_wo_z_rot,
            "body_joints_vel": self._get_body_joints_vel,
            "joint_global_oris": self._get_joint_global_orientations,
            "joint_ang_vel": self._get_joint_angular_velocity,
            "wrists_ang_vel": self._get_wrists_angular_velocity,
            "wrists_ang_vel_euler": self._get_wrists_angular_velocity_euler,
            
            "contact_masks": self._get_contact_masks,
            "obj_rotation": self._get_obj_rotation,
            "obj_translation": self._get_obj_translation,
        }

        # 0 amass 1 other datasets 
        self._meta_data_get_methods = {"framerate": self._get_framerate}
        
        self.nfeats = self.get_features_dimentionality()
 

    def get_features_dimentionality(self):
        """Get the dimentionality of the concatenated load_feats"""
   
        random_idx = np.random.choice(list(self.data.keys()), 1).item()
        item = self.__getitem__(random_idx)
 
        return [item[feat].shape[-1] for feat in self.load_feats
                if feat in self._feat_get_methods.keys()]

    def normalize_feats(self, feats, feats_name):
        
        if feats_name not in self.stats.keys():
            log.error(f"Tried to normalise {feats_name} but did not found stats \
                      for this feature. Try running calculate_statistics.py again.")
        
        if self.norm_type == "std":
            mean, std = self.stats[feats_name]['mean'].to(feats.device), self.stats[feats_name]['std'].to(feats.device)
            return (feats - mean) / (std + 1e-5)
        elif self.norm_type == "norm":
            max, min = self.stats[feats_name]['max'].to(feats.device), self.stats[feats_name]['min'].to(feats.device)
            
            if (feats - min) / (max - min + 1e-5) >= 1.05 and (feats - min) / (max - min + 1e-5) <= -0.01:
                print(f"Warning: Normalized feature {feats_name} has values outside [0,1]. Check if the stats are correct.")
            return (feats - min) / (max - min + 1e-5)

    def _get_body_joints(self, data):
        joints = to_tensor(data['joint_positions'][:, :self.n_body_joints, :])
        return rearrange(joints, '... joints dims -> ... (joints dims)')

    def _get_joint_global_orientations(self, data):
        body_pose = to_tensor(data['rots'][..., 3:self.n_body_joints * 3])  # drop pelvis orientation
        body_orient = to_tensor(data['rots'][..., :3])
        joint_glob_oris = local_to_global_orient(body_orient, body_pose,
                                                 self.body_chain,
                                                 input_format='aa',
                                                 output_format="rotmat")
        return rearrange(joint_glob_oris, '... j k d -> ... (j k d)')

    def _get_joint_angular_velocity(self, data):        
        pose = to_tensor(data['rots'][..., 3: self.n_body_joints*3])  # drop pelvis orientation
        # pose = rearrange(pose, '... (j c) -> ... j c', c=3)
        # pose = axis_angle_to_matrix(to_tensor(pose))
        pose = transform_body_pose(pose, "aa->rot")
        rot_diffs = torch.einsum('...ik,...jk->...ij', pose, pose.roll(1, 0))
        rot_diffs[0] = torch.eye(3).to(rot_diffs.device)  # suppose zero angular vel at first frame
        return rearrange(matrix_to_rotation_6d(rot_diffs), '... j c -> ... (j c)')

    def _get_wrists_angular_velocity(self, data):
        pose = to_tensor(data['rots'][..., 3: self.n_body_joints*3])  # drop pelvis orientation
        pose = axis_angle_to_matrix(to_tensor(pose[..., 19:21, :]))  
        pose = transform_body_pose(pose, "aa->rot")
        rot_diffs = torch.einsum('...ik,...jk->...ij', pose, pose.roll(1, 0))
        rot_diffs[0] = torch.eye(3).to(rot_diffs.device)  # suppose zero angular vel at first frame
        return rearrange(matrix_to_rotation_6d(rot_diffs), '... j c -> ... (j c)')

    def _get_wrists_angular_velocity_euler(self, data):
        pose = to_tensor(data['rots'][..., 3:3 + 3*21])  # drop pelvis orientation
        pose = rearrange(pose, '... (j c) -> ... j c', c=3)
        pose = transform_body_pose(to_tensor(pose[..., 19:21, :]), "aa->rot")
        rot_diffs = torch.einsum('...ik,...jk->...ij', pose, pose.roll(1, 0))
        rot_diffs[0] = torch.eye(3).to(rot_diffs.device)  # suppose zero angular vel at first frame
        return rearrange(matrix_to_euler_angles(rot_diffs, "XYZ"), '... j c -> ... (j c)')

    def _get_body_joints_vel(self, data):
        joints = to_tensor(data['joint_positions'][:, :self.n_body_joints, :])
        joint_vel = joints - joints.roll(1, 0)  # shift one right and subtract
        joint_vel[0] = 0
        return rearrange(joint_vel, '... j c -> ... (j c)')

    def _get_body_joints_local_wo_z_rot(self, data):
        """get body joint coordinates relative to the pelvis"""
        joints = to_tensor(data['joint_positions'][:, :self.n_body_joints, :])
        pelvis_transl = to_tensor(joints[:, 0, :])
        pelvis_orient = to_tensor(data['rots'][..., :3])

        pelvis_orient_z = get_z_rot(pelvis_orient, in_format="aa")

        rel_joints = torch.einsum('fdi,fjd->fji',
                                  pelvis_orient_z,
                                  joints - pelvis_transl[:, None, :])
 
        return rearrange(rel_joints, '... j c -> ... (j c)')

    def _get_body_joints_rel(self, data):
               
        """get body joint coordinates relative to the pelvis"""
        joints = to_tensor(data['joint_positions'][:, :self.n_body_joints, :])
        pelvis_transl = to_tensor(joints[:, 0, :])
        joints_glob = to_tensor(joints[:, :self.n_body_joints, :])
        pelvis_orient = to_tensor(data['rots'][..., :3])
        pelvis_orient = transform_body_pose(pelvis_orient, "aa->rot").float()
        # relative_joints = R.T @ (p_global - pelvis_translation)
        rel_joints = torch.einsum('fdi,fjd->fji', pelvis_orient, joints_glob - pelvis_transl[:, None, :])
        return rearrange(rel_joints, '... j c -> ... (j c)')

    def _get_contact_masks(self, data):
            
        return data['contacts_mask'][:, [SMPLX_JOINTS["left_ankle"], 
                                        SMPLX_JOINTS["right_ankle"], 
                                        SMPLX_JOINTS["left_foot"], 
                                        SMPLX_JOINTS["right_foot"]]]
                                
    @staticmethod
    def _get_framerate(data):
        """get framerate"""
        return torch.tensor([30])

    @staticmethod
    def _get_chunk_start(data):
        """get number of original sequence frames"""
        return torch.tensor([data['chunk_start']])

    @staticmethod
    def _get_num_frames(data):
        """get number of original sequence frames"""
        return torch.tensor([data['rots'].shape[0]])

    @staticmethod
    def _get_body_transl(data):
        """get body pelvis translation"""
        return to_tensor(data['trans'])
        # body.translation is NOT the same as the pelvis translation=
        # return to_tensor(data.body.params.transl)

    @staticmethod
    def _get_body_transl_z(data):
        """get body pelvis tranlation"""
        return to_tensor(data['joint_positions'])[:, 0, 2:] # only z

    @staticmethod
    def _get_body_transl_delta(data):
        """get body pelvis tranlation delta"""
        trans = to_tensor(data['trans'])
        trans_vel = trans - trans.roll(1, 0)  # shift one right and subtract
        trans_vel[0] = 0  # zero out velocity of first frame
        return trans_vel
  

    def _get_obj_rotation(self, data):
        """Return canonical object rotation (already canonicalized by _canonicalize_datum)."""
        obj_rot = to_tensor(data['obj_orient'])
        if self.rot_repr == "6d":
            obj_rot = transform_body_pose(obj_rot, "aa->6d")
        return obj_rot

    def _get_obj_translation(self, data):
        """Return canonical object translation (already canonicalized by _canonicalize_datum)."""
        return to_tensor(data['obj_trans'])
    
    
    def _get_body_transl_delta_pelv(self, data):
        """
        get body pelvis tranlation delta relative to pelvis coord.frame
        v_i = t_i - t_{i-1} relative to R_{i-1}
        """    
        
        trans = to_tensor(data['trans'])
        data['init_pelvis_trans'] = trans[0] 
            
        trans_vel = trans - trans.roll(1, 0)  # shift one right and subtract
 
        rot = to_tensor(data['rots'].reshape(-1, self.n_body_joints * 3)[..., :3]).float()
        
        pelvis_orient = transform_body_pose(rot, "aa->rot")
 
        trans_vel_pelv = change_for(trans_vel, pelvis_orient.roll(1, 0))
        trans_vel_pelv[0] = 0  # zero out velocity of first frame
 
        return trans_vel_pelv
    
    @staticmethod
    def _get_body_transl_delta_pelv_xy(data):
        """
        get body pelvis tranlation delta while removing the global z rotation of the pelvis
        v_i = t_i - t_{i-1} relative to R_{i-1}_xy
        """
        trans = to_tensor(data['trans'])
        trans_vel = trans - trans.roll(1, 0)  # shift one right and subtract
        pelvis_orient = to_tensor(data['rots'][..., :3])
        R_z = get_z_rot(pelvis_orient, in_format="aa")
        # rotate -R_z
        trans_vel_pelv = change_for(trans_vel, R_z.roll(1, 0), forward=True)
        trans_vel_pelv[0] = 0  # zero out velocity of first frame
        return trans_vel_pelv

    @staticmethod
    def _get_body_transl_delta_pelv_xy_wo_z(data):
        """
        get body pelvis tranlation delta while removing the global z rotation of the pelvis
        v_i = t_i - t_{i-1} relative to R_{i-1}_xy
        """
        trans = to_tensor(data['joint_positions'][:, 0, :])
        # trans = to_tensor(data['trans'])
        trans_vel = trans - trans.roll(1, 0)  # shift one right and subtract
        pelvis_orient = to_tensor(data['rots'][..., :3])
        R_z = get_z_rot(pelvis_orient, in_format="aa")
        # rotate -R_z
        trans_vel_pelv = change_for(trans_vel, R_z.roll(1, 0), forward=True)
        trans_vel_pelv[0] = 0  # zero out velocity of first frame
        return trans_vel_pelv[..., :2]

    def _get_body_orient(self, data):
        """get body global orientation"""
        # default is axis-angle representation
        pelvis_orient = to_tensor(data['rots'][..., :3])
        if self.rot_repr == "6d":
            # axis-angle to rotation matrix & drop last row
            pelvis_orient = transform_body_pose(pelvis_orient, "aa->6d")
        return pelvis_orient

    def _get_body_orient_xy(self, data):
        """get body global orientation"""
        # default is axis-angle representation
        pelvis_orient = to_tensor(data['rots'][..., :3])
        if self.rot_repr == "6d":
            # axis-angle to rotation matrix & drop last row
            pelvis_orient_xy = remove_z_rot(pelvis_orient, in_format="aa")
        return pelvis_orient_xy

    def _get_body_orient_delta(self, data):
        """get global body orientation delta"""
        # default is axis-angle representation
        pelvis_orient = to_tensor(data['rots'][..., :3])
        pelvis_orient_delta = rot_diff(pelvis_orient, in_format="aa",
                                       out_format=self.rot_repr)
        return pelvis_orient_delta

    def _get_z_orient_delta(self, data):
        """get global body orientation delta"""
        # default is axis-angle representation
        pelvis_orient = to_tensor(data['rots'][..., :3])
        pelvis_orient_z = get_z_rot(pelvis_orient, in_format="aa")
        pelvis_orient_z = transform_body_pose(pelvis_orient_z, "rot->aa")
        z_orient_delta = rot_diff(pelvis_orient_z, in_format="aa",
                                       out_format=self.rot_repr)
        
        data["init_pelvis_orient_z"] = to_tensor(pelvis_orient_z[0])
        
        return z_orient_delta

    def _get_body_pose(self, data):
        """get body pose"""
        
        # default is axis-angle representation: Frames x (Jx3) (J=21)
        pose = to_tensor(data['rots'][..., 3: self.n_body_joints * 3])  # drop pelvis orientation
        pose = transform_body_pose(pose, f"aa->{self.rot_repr}")
        return pose

    def _get_body_pose_delta(self, data):
        """get body pose rotational deltas"""
        # default is axis-angle representation: Frames x (Jx3) (J=21)
        pose = to_tensor(data['rots'][..., 3: self.n_body_joints * 3])  # drop pelvis orientation
        pose_diffs = rot_diff(pose, in_format="aa", out_format=self.rot_repr)
        return pose_diffs

    def _canonicalize_datum(self, data):
        """Canonicalize body (and object) to face-front orientation in-place.

        Calls _canonica_facefront on rots/trans (and obj_orient/obj_trans if
        present), then rotates joint_positions by the same R_change so all
        fields are consistent in the canonical frame.
        """
        rots = to_tensor(data['rots'])      # (T, J*3)
        trans = to_tensor(data['trans'])     # (T, 3)

        has_obj = 'obj_orient' in data and 'obj_trans' in data

        # Build body_mesh_fn for centroid correction when SMPLX model is available
        body_mesh_fn = None
        if has_obj and self.body_model_dict is not None:
            gender = data.get('gender', 'neutral')
            bm = self.body_model_dict.get(gender)
            if bm is not None:
                betas_raw = to_tensor(data['betas']).float()
                if betas_raw.dim() == 1:
                    betas_raw = betas_raw.unsqueeze(0).expand(rots.shape[0], -1)
                n_betas = bm.num_betas if hasattr(bm, 'num_betas') else 300
                if betas_raw.shape[-1] < n_betas:
                    pad = torch.zeros(betas_raw.shape[0], n_betas - betas_raw.shape[-1])
                    betas_raw = torch.cat([betas_raw, pad], dim=-1)
                dev = next(bm.parameters()).device
                betas_dev = betas_raw.to(dev)

                def body_mesh_fn(rots_aa, trans_in):
                    T = rots_aa.shape[0]
                    psd = {
                        'betas': betas_dev[:T],
                        'expression': torch.zeros((T, 10), device=dev),
                        'transl': to_tensor(trans_in).float().to(dev),
                        'global_orient': to_tensor(rots_aa[:, :3]).float().to(dev),
                        'body_pose': to_tensor(rots_aa[:, 3:66]).float().to(dev),
                        'jaw_pose': to_tensor(rots_aa[:, 66:69]).float().to(dev),
                        'leye_pose': to_tensor(rots_aa[:, 69:72]).float().to(dev),
                        'reye_pose': to_tensor(rots_aa[:, 72:75]).float().to(dev),
                        'left_hand_pose': to_tensor(rots_aa[:, 75:120]).float().to(dev),
                        'right_hand_pose': to_tensor(rots_aa[:, 120:]).float().to(dev),
                    }
                    with torch.no_grad():
                        out = bm(**psd)
                    return out.vertices.detach().cpu().numpy()

        if has_obj:
            obj_rot = to_tensor(data['obj_orient'])
            obj_trans = to_tensor(data['obj_trans'])
            rots_can, trans_can, obj_rot_can, obj_trans_can = \
                self._canonica_facefront(rots, trans,
                                         obj_rot=obj_rot, obj_trans=obj_trans,
                                         body_mesh_fn=body_mesh_fn)
            data['obj_orient'] = obj_rot_can
            data['obj_trans'] = obj_trans_can
        else:
            rots_can, trans_can = self._canonica_facefront(rots, trans)

        # Canonicalize joint positions with the same R_change
        if 'joint_positions' in data:
            joints = to_tensor(data['joint_positions'])  # (T, J, 3)
            orig_pelvis_rotmat = transform_body_pose(rots[:, :3], "aa->rot")
            can_pelvis_rotmat = transform_body_pose(rots_can[:, :3], "aa->rot")
            R_change = can_pelvis_rotmat @ orig_pelvis_rotmat.transpose(-1, -2)
            rel = joints - trans[:, None]
            rel_can = torch.einsum('fij,fkj->fki', R_change, rel)
            data['joint_positions'] = trans_can[:, None] + rel_can

        data['rots'] = rots_can
        data['trans'] = trans_can

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _canonica_facefront(rotations, translation,
                            obj_rot=None, obj_trans=None,
                            obj_com_local=None, obj_scale=None,
                            body_mesh_fn=None):
        """Canonicalize body (and optionally object) to face-front orientation.

        Args:
            rotations:     (T, J*3) body joint rotations in axis-angle.
            translation:   (T, 3)   body pelvis translation.
            obj_rot:       (T, 3)   object global rotation, axis-angle.
            obj_trans:     (T, 3)   object translation (mesh frame origin).
            obj_com_local: (3,)     template COM in local frame (ARCTIC/OMOMO).
                           If None, obj_trans is rotated directly (centered meshes).
            obj_scale:     scalar or (T,) mesh scale factor (OMOMO).
                           If None, treated as 1.0 (no scaling, e.g. ARCTIC).
            body_mesh_fn:  callable (rots_aa (T,J*3), trans (T,3)) -> (T,V,3)
                           numpy array of body mesh vertices.  Used to compute
                           the exact body-mesh centroid shift caused by SMPLX
                           LBS + pose blend shapes when root orientation changes.
                           If None, no centroid correction is applied.

        Returns:
            Without object args  : (rots_can, trans_can)
            With object args only: (rots_can, trans_can, obj_rot_can, obj_trans_can)
            With obj_com_local   : (rots_can, trans_can, obj_rot_can, obj_trans_can,
                                    obj_com_can)
        """
        rots_motion = rotations
        trans_motion = translation
        datum_len = rotations.shape[0]
        rots_motion_rotmat = transform_body_pose(rots_motion.reshape(datum_len,
                                                           -1, 3),
                                                           'aa->rot')
        # Clone before canonicalize_rotations so we can compute R_change later
        orig_pelvis_rotmat = rots_motion_rotmat[:, 0].clone()

        orient_R_can, trans_can = canonicalize_rotations(rots_motion_rotmat[:, 0],
                                                         trans_motion)
        rots_motion_rotmat_can = rots_motion_rotmat
        rots_motion_rotmat_can[:, 0] = orient_R_can
        # Only center XY — Z rotation never changes Z so Z values are preserved as-is.
        translation_can = trans_can.clone()
        translation_can[:, :2] = translation_can[:, :2] - trans_can[0, :2]
        rots_motion_aa_can = transform_body_pose(rots_motion_rotmat_can, 'rot->aa')
        rots_motion_aa_can = rearrange(rots_motion_aa_can, 'F J d -> F (J d)', d=3)

        if obj_rot is None or obj_trans is None:
            return rots_motion_aa_can, translation_can

        # R_change[t] = R_canon[t] @ R_orig[t]^T
        # canonicalize_rotations only modifies the Z angle, so R_change is a
        # pure per-frame Z rotation — safe to apply identically to the object.
        R_change = orient_R_can @ orig_pelvis_rotmat.transpose(-1, -2)  # (T,3,3)

        # --- object rotation ---
        obj_rot_rotmat = transform_body_pose(obj_rot, "aa->rot")          # (T,3,3)
        obj_rot_can_rotmat = torch.einsum('fij,fjk->fik', R_change, obj_rot_rotmat)
        obj_rot_can = transform_body_pose(obj_rot_can_rotmat, "rot->aa")  # (T,3)

        # --- body centroid correction ---
        # SMPLX LBS + pose blend shapes shift the body mesh centroid relative
        # to the pelvis when root orientation changes.  Compute the exact shift
        # from body vertices before/after canonicalization so the object tracks
        # the body centroid rather than just the pelvis.
        if body_mesh_fn is not None:
            body_verts_before = body_mesh_fn(rots_motion, trans_motion)
            body_verts_after  = body_mesh_fn(rots_motion_aa_can, translation_can)
            body_c_before = torch.tensor(body_verts_before.mean(1),
                                         dtype=torch.float32, device=R_change.device)
            body_c_after  = torch.tensor(body_verts_after.mean(1),
                                         dtype=torch.float32, device=R_change.device)
            # Difference between actual centroid offset and rigidly-rotated offset
            centroid_correction = (body_c_after - translation_can) - \
                torch.einsum('fij,fj->fi', R_change, body_c_before - trans_motion)
        else:
            centroid_correction = 0.0

        # --- object translation ---
        if obj_com_local is not None:
            com_local_dev = obj_com_local.to(obj_rot_rotmat.device)
            # Scale factor: OMOMO uses v = scale * R @ v + t, ARCTIC uses scale=1
            if obj_scale is not None:
                s = obj_scale.to(obj_rot_rotmat.device)
                if s.dim() >= 1:
                    s = s.view(-1, 1)  # (T,) → (T,1) for broadcasting
            else:
                s = 1.0
            # (T,3) world-space COM trajectory: scale * R @ com_local + transl
            rot_com = torch.einsum('fij,j->fi', obj_rot_rotmat, com_local_dev)
            com_world = s * rot_com + obj_trans
            # Rotate relative offset body→COM by R_change, add to canonical body pos
            rel = com_world - trans_motion
            rel_can = torch.einsum('fij,fj->fi', R_change, rel)
            obj_com_can = trans_can + rel_can + centroid_correction
            obj_com_can[:, :2] = obj_com_can[:, :2] - trans_can[0, :2]
            # Canonical mesh-frame-origin translation: transl = com - scale * R_can @ com_local
            rot_com_can = torch.einsum('fij,j->fi', obj_rot_can_rotmat, com_local_dev)
            obj_trans_can = obj_com_can - s * rot_com_can
            return rots_motion_aa_can, translation_can, obj_rot_can, obj_trans_can, obj_com_can

        # No COM: rotate body→object offset directly
        rel = obj_trans - trans_motion
        rel_can = torch.einsum('fij,fj->fi', R_change, rel)
        obj_trans_can = trans_can + rel_can + centroid_correction
        obj_trans_can[:, :2] = obj_trans_can[:, :2] - trans_can[0, :2]
        return rots_motion_aa_can, translation_can, obj_rot_can, obj_trans_can
    
    @staticmethod
    def load_object_geometry(object_name, obj_scale,
                             obj_trans, obj_rot,
                                obj_arti, datasetname,
                                obj_bottom_scale=None,
                                obj_bottom_trans=None,
                                obj_bottom_rot=None,
                                load_simplified=False,
                                reflect_x=False):
  
        if datasetname == 'GRAB':

            if load_simplified:
                obj_mesh_path = f'data/motion/Hand_Raw/GRAB/grab/tools/object_meshes/contact_meshes_simplified/{object_name}.ply'
            else:
                obj_mesh_path = f'data/motion/Hand_Raw/GRAB/grab/tools/object_meshes/contact_meshes/{object_name}.ply'

 
            obj_inp_dict = {'obj_mesh_path': obj_mesh_path,
                            'obj_scale': obj_scale,
                            'obj_rot': axis_angle_to_matrix(obj_rot),
                            'obj_trans': obj_trans,
                            'reflect_x': reflect_x}

            obj_mesh_verts, obj_mesh_faces = apply_transformation_to_obj_geometry(**obj_inp_dict)
            

            return {'vertices': obj_mesh_verts.detach().cpu().numpy(), 
                    'faces': obj_mesh_faces}

        elif datasetname == 'ARCTIC':
            T = obj_arti.shape[0]

            if reflect_x:
                # Augmented data stores conjugated rotation M@R@M and
                # reflected translation M@t.  ARCTIC uses an articulated
                # forward pass so we can't just flip template vertices.
                # Instead: undo augmentation, run forward with original
                # params, then reflect the output vertices.
                obj_rot_fwd = obj_rot.clone()
                obj_rot_fwd[:, 1:] *= -1
                obj_trans_fwd = obj_trans.clone()
                obj_trans_fwd[:, 0] *= -1
            else:
                obj_rot_fwd = obj_rot
                obj_trans_fwd = obj_trans

            obj_inp_dict = {'angles': obj_arti[:, None],
                            'global_orient': obj_rot_fwd,
                            'query_names': np.array([object_name] * T),
                            'transl': obj_trans_fwd,
                            'fwd_template': False}

            if load_simplified:
                obj_out = arctic_obj_forward_simplify(**obj_inp_dict)
            else:
                obj_out = arctic_obj_forward_raw(**obj_inp_dict)

            verts = obj_out['v'].detach().cpu().numpy()
            if reflect_x:
                verts[:, :, 0] *= -1

            return {'vertices': verts,
                    'faces': obj_out['f'][0].cpu().numpy().astype(np.uint32),
                    'top_com': obj_out['top_com'],
                    'bottom_com': obj_out['bottom_com'],
                    'merged_com': obj_out['merged_com'],
                    'bottom_idx': np.where(obj_out['parts_ids'][0] == 2)[0],
                    'top_idx': np.where(obj_out['parts_ids'][0] == 1)[0]}


        elif datasetname == 'OMOMO':

            obj_rot = transform_body_pose(obj_rot, 'aa->rot')

            if object_name in ['vacuum', 'mop'] :

                if load_simplified:
                    top_obj_mesh_path = f'data/motion/Body_Raw/OMOMO/captured_objects_simplified/{object_name}_cleaned_simplified_top.obj'
                    bottom_obj_mesh_path = f'data/motion/Body_Raw/OMOMO/captured_objects_simplified/{object_name}_cleaned_simplified_bottom.obj'
                else:
                    top_obj_mesh_path = f'data/motion/Body_Raw/OMOMO/captured_objects/{object_name}_cleaned_simplified_top.obj'
                    bottom_obj_mesh_path = f'data/motion/Body_Raw/OMOMO/captured_objects/{object_name}_cleaned_simplified_bottom.obj'
                
                obj_bottom_scale = obj_scale
                obj_bottom_rot = obj_rot
                obj_bottom_trans = obj_trans

        
                top_obj_mesh_verts, top_obj_mesh_faces = apply_transformation_to_obj_geometry(top_obj_mesh_path,
                    obj_scale, obj_rot, obj_trans, reflect_x=reflect_x)
                bottom_obj_mesh_verts, bottom_obj_mesh_faces = apply_transformation_to_obj_geometry(bottom_obj_mesh_path,
                    obj_bottom_scale, obj_bottom_rot, obj_bottom_trans, reflect_x=reflect_x)

                obj_mesh_verts, obj_mesh_faces = merge_two_parts([top_obj_mesh_verts, bottom_obj_mesh_verts],
                    [top_obj_mesh_faces, bottom_obj_mesh_faces])

            else:
                if load_simplified:
                    obj_mesh_path = f'data/motion/Body_Raw/OMOMO/captured_objects_simplified/{object_name}_cleaned_simplified.obj'
                else:
                    obj_mesh_path = f'data/motion/Body_Raw/OMOMO/captured_objects/{object_name}_cleaned_simplified.obj'

                obj_mesh_verts, obj_mesh_faces = apply_transformation_to_obj_geometry(obj_mesh_path,
                    obj_scale, obj_rot, obj_trans, reflect_x=reflect_x)
 

            return {'vertices': obj_mesh_verts.detach().cpu().numpy(), 
                    'faces': obj_mesh_faces}

           
     

    def __getitem__(self, idx):

        datum = dict(self.data[idx])  # shallow copy to avoid mutating self.data
        self._canonicalize_datum(datum)
        

        data_dict = {}
        for feat in self.load_feats:
            if 'precomputed_features' not in datum.keys():
                data_dict[feat] = self._feat_get_methods[feat](datum).to(self.device)
            elif feat not in datum['precomputed_features'].keys():
                data_dict[feat] = self._feat_get_methods[feat](datum).to(self.device)
            else:
                data_dict[feat] = datum['precomputed_features'][feat].to(self.device)

        for feat, method in self._meta_data_get_methods.items():
            data_dict[feat] = method(datum)

        data_dict['datum'] = datum
        data_dict['id'] = datum['id']
        data_dict['length'] = len(data_dict['body_pose'])
        data_dict['dataset_name'] = datum['body_dataset_name']

        # cast betas
        pad_shape = list(datum['betas'].shape)
        pad_shape[-1] = 300 - datum['betas'].shape[-1]
        datum['betas'] = torch.cat([datum['betas'], torch.zeros(pad_shape, dtype=datum['betas'].dtype, device=datum['betas'].device)], dim=-1)
                
        return DotDict(data_dict)

    def get_all_features(self, idx, load_feats=None):

        datum = dict(self.data[idx])  # shallow copy to avoid mutating self.data
        self._canonicalize_datum(datum)

        load_feats = self.load_feats if load_feats is None else load_feats

        data_dict_source = {f'{feat}': self._feat_get_methods[feat](datum)
                            for feat in load_feats}

        meta_data_dict = {feat: method(datum) for feat, method in self._meta_data_get_methods.items()}

        data_dict = {**data_dict_source, **meta_data_dict}

        return DotDict(data_dict)