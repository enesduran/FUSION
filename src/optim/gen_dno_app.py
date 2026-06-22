import os
import sys
import glob
import copy
import torch
import smplx
import joblib
import shutil
import trimesh
import argparse
import functools
import itertools
import subprocess
import numpy as np
from ema_pytorch import EMA
from psbody.mesh import Mesh
from omegaconf import OmegaConf
from bps_torch.bps import bps_torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import dno_helper
from utils.transforms import quat_fk
from optim.dno_adam import DNOOptions
from optim.dno_common import get_number_of_stages, configure_model, setup_output_folders, load_checkpoint
from condition import CondKeyLocationsLoss
from utils.genutils import seed_everything
from src.diffusion import create_diffusion
from src.data.amass_dataset import AmassDataset
from src.utils.metric_utils import save_metrics
from src.data.amass_wrapper import AmassWrapper
from src.model.base_diffusion import MotionDiffusion
from src.render.mesh_viz import RendererWrapper, colors_dict
from src.utils.transforms3d import transform_body_pose, rot_diff, get_z_rot, matrix_to_axis_angle
from src.utils.viz_utils import merge_videos, pack_to_render, color1, color2, color3, obj_color
from src.utils.optim_utils import prepare_grad_masks, prepare_lock_masks, ddim_loop_with_gradient, plot_loss
from src.utils.process_utils import BRANCH_NAME, SMPLX_JOINTS, GLOBAL_RHAND_TIPS_INDICES, GLOBAL_LHAND_TIPS_INDICES


class OptimizerWrapper:

    def __init__(self, ema_decay=0.995, ema_update_every=10):
        
        self.num_poses = 5
        optimization_cfg.use_ddim = True
        self.demo_mode = bool(optimization_cfg.demo_mode) if 'demo_mode' in optimization_cfg else False
        
        self.renderer = RendererWrapper(path2body_models=data_cfg.smplx_path)

        print("Loading dataset...")
        self.prep_dataloader()
        
        configure_model(model_cfg, data_cfg, diffusion_cfg, self.data_wrapper)
        self.NUMBER_OF_STAGES = get_number_of_stages(optimization_cfg)
        
        print("Creating model and diffusion...")

        self.num_ode_steps = 5
 
        self.motion_diff_model = MotionDiffusion(**model_cfg, **loss_cfg)
        self.ema = EMA(self.motion_diff_model.denoiser, beta=ema_decay, update_every=ema_update_every)
 
        ### Task selection ###
        if optimization_cfg.obj_contact_optimization:

            self.task = "object_guided_optimization"
            self.render_hand = True

            self.grabnet_bps = bps_torch(custom_basis = grabnet.bps)
            grabnet.refine_net.rhm_train = self.motion_diff_model.rot2xyz.hand_dict['right']

        elif optimization_cfg.use_clutch:
            
            self.task = "clutch_optimization"
            self.render_hand = False

            self.watertight_conversion_dict = joblib.load('data/body_models/watertight/conversion_dict.pkl')

            self.base_lhand_vertex_ids = self.watertight_conversion_dict['basemesh_lhand_vertex_ids']
            self.base_rhand_vertex_ids = self.watertight_conversion_dict['basemesh_rhand_vertex_ids']

            self.base_hand_vertex_ids = np.concatenate((self.base_lhand_vertex_ids, self.base_rhand_vertex_ids)) 

            
        elif optimization_cfg.self_contact_optimization:
            self.task = "self_contact_optimization"
            self.client = get_openai_client()
            self.render_hand = False


            self.interaction_dict_path = 'data/self_interaction/self_interaction_dict.p' 
            
            if os.path.exists(self.interaction_dict_path):
                self.interaction_dict = joblib.load(self.interaction_dict_path)
            else:
                self.interaction_dict = dict()

        else:
            raise ValueError(f"Unknown task: {self.task}")


    
        vis_res_suffix = "_".join((self.task.split("_")[:-1])) + "_vis_res"
        quant_res_suffix = "_".join((self.task.split("_")[:-1])) + "_quant_res"

        self.weights_folder, self.vis_folder, self.results_folder = setup_output_folders(
            optimization_cfg, vis_res_suffix, quant_res_suffix, opt.optimizer_file)

        self.add_fingertips_flag = True
        self.NUM_JOINTS = self.motion_diff_model.rot2xyz.bm_dict['neutral'].NUM_JOINTS + 1   # 55

        self.step = load_checkpoint(self.ema, self.motion_diff_model.denoiser,
                                    self.weights_folder, optimization_cfg.milestone,
                                    model_cfg.pretrained_path)
                 
        self.optimize()
        
    @staticmethod
    def load_body_vtemp(body_vtemp):
        
        if body_vtemp == 'no_vtemp':
            return None
        else:
            return torch.tensor(trimesh.load(file_obj=body_vtemp).vertices).to(device).float()

    
     
    def prep_dataloader(self):
        data_cfg.load_splits = ['test']
        data_cfg.device = device 
        
        data_cfg.train_batch_size = optimization_cfg.train_batch_size
        data_cfg.val_batch_size = optimization_cfg.val_batch_size
        data_cfg.demo_mode = optimization_cfg.demo_mode 
        
        if self.demo_mode:      
            data_cfg.motion_filepath = optimization_cfg.motion_filepath
            
            data_cfg.include_pose_augmentation = False
            data_cfg.include_time_augmentation = False
        
        self.data_wrapper = AmassWrapper(**data_cfg)

        if 'test' in data_cfg.load_splits:
            self.dl = self.data_wrapper.test_dataloader()
        if 'val' in data_cfg.load_splits:
            self.dl = self.data_wrapper.val_dataloader()
        if 'train' in data_cfg.load_splits:
            self.dl = self.data_wrapper.train_dataloader()
            


        ### Task selection ###
        if optimization_cfg.obj_contact_optimization:
            pass 
        
            if self.demo_mode:
                # filter the dataset to only include the specified object category for demo. 
                assert self.dl.dataset.data[0] == "GRAB", f"HOI method only works with GRAB objects!"
                

                # pop elements that do not belong to the category.
                self.dl.dataset.data = [self.dl.dataset.data[i] for i in range(total_length) if self.dl.dataset.data[i]['datum']['obj_name'] == object_category]
            

        elif optimization_cfg.use_clutch:
            total_length = len(clutch_dict_all)
            
            # pop elements after the length.
            self.dl.dataset.data = [self.dl.dataset.data[i] for i in range(total_length)]
                         
        # self contact optimization. Means that we are not bound to the dataset. 
        elif optimization_cfg.self_contact_optimization:
            # find total length 
            total_length = self_interaction_length

            # pop elements after the length.
            self.dl.dataset.data = [self.dl.dataset.data[i] for i in range(total_length)]

    # @staticmethod
    def find_closest_index_angle(self, grasp_dict, val_data_batch, idx):


        # either
        start_x = self.motion_diff_model.forward()(torch.zeros((1, optimization_cfg.window, model_cfg.nfeats)).to(device), 
                                         timestep=torch.tensor([300]).long().to(device)).detach()
        diffout_dict = self.motion_diff_model.diffout2motion(start_x)
        
        smplx_params = pack_to_render(trans=diffout_dict['full_motion_unnorm'][..., :3].cpu(), 
                                            rots=diffout_dict['full_motion_unnorm'][..., 3:].cpu(),
                                            pose_repr='6d',
                                            device=device)
        
        
        lrot_quat = torch.cat([smplx_params['global_orient'], 
                               smplx_params['body_pose'], 
                               smplx_params['jaw_pose'], 
                               smplx_params['leye_pose'], 
                               smplx_params['reye_pose'], 
                               smplx_params['left_hand_pose'], 
                               smplx_params['right_hand_pose']], dim=-1)[:, 0]
        lrot_quat = transform_body_pose(lrot_quat.reshape(1, 55, 3), 'aa->quat')
 
        global_rot, global_pos = quat_fk(lrot_quat.cpu(), 
                                        val_data_batch['datum']['pos_offset'][idx:idx+1], 
                                        self.motion_diff_model.rot2xyz.bm_dict['neutral'].parents)
    
        global_rot = transform_body_pose(global_rot, 'quat->rot')[0, SMPLX_JOINTS['right_wrist']]


    
        hand_rotmat_grabnet = transform_body_pose(grasp_dict['global_orient'], 'aa->rot')



        traces = np.trace(hand_rotmat_grabnet.transpose(2, 1).cpu() @ global_rot.numpy(), axis1=1, axis2=2)  # Shape: (N,)
    
        closest_index = np.argmax(traces)  # Maximum trace means closest rotation
        closest_matrix = hand_rotmat_grabnet[closest_index]
        max_trace = traces[closest_index]

        return closest_index
    

    def find_closest_index_distance(self, t_pos, rotated_targets):

        closest_index = torch.argmin(torch.linalg.norm(rotated_targets[:, 0, 0] - t_pos[SMPLX_JOINTS['right_wrist']].to(device), dim=-1)).item()
        
        return closest_index 
    
    def detect_object_movement_times(self, obj_rot, obj_trans):

        obj_vel = obj_trans[:, :-1] - obj_trans[:, 1:]
        obj_vel = torch.cat((torch.zeros_like(obj_vel[:, :1]), obj_vel), dim=1)

        # obj_vel_ = obj_trans - torch.roll(obj_trans, dims=1, shifts=1)
        rotmat_diff = rot_diff(obj_rot, in_format='aa', out_format='rot')
        aa_diff = rot_diff(obj_rot, in_format='aa', out_format='aa')


        def get_rotation_angle_magnitude(rotation_matrices: torch.Tensor) -> torch.Tensor:
            """
            Get the magnitude of rotation angle from rotation matrices
            
            Args:
                rotation_matrices: (N, T, 3, 3) or (N, 3, 3) tensor
                
            Returns:
                Rotation angle magnitudes in radians
            """
            # Use trace to compute angle: trace(R) = 1 + 2*cos(θ)
            trace = torch.diagonal(rotation_matrices, dim1=-2, dim2=-1).sum(-1)
            angle = torch.acos(torch.clamp(trace / 2, -1 + 1e-6, 1 - 1e-6))
            
            return angle



        rotmat_diff_angles = get_rotation_angle_magnitude(rotmat_diff)
        aa_diff_angles = torch.linalg.norm(aa_diff, ord=2, dim=-1)


        angle_time_idx = rotmat_diff_angles > 2e-1
        trans_time_idx = torch.linalg.norm(obj_vel, ord=2, dim=-1) > 8e-4
        

        movement_time_idx = torch.logical_or(angle_time_idx, trans_time_idx)

        return movement_time_idx
    

    def get_t_pose(self, *, val_data_batch):

        t_pose = self.motion_diff_model.rot2xyz.forward_joints(torch.zeros((val_data_batch['body_pose'].shape[0], 3)),
                                                              val_data_batch['datum']['pos_offset'])
                 
        t_pose = t_pose @ torch.tensor([[1, 0, 0],
                                        [0, 0, 1],
                                        [0, -1, 0]]).float()
            
        
        t_pose += (self.motion_diff_model.first_trans[:val_data_batch['body_pose'].shape[0]].cpu() + 
                   val_data_batch['datum']['root_offset'][:, None])
        
        return t_pose

    def forward_grabnet(self, obj_dict_list):

        grabnet_output = []
        
        for obj_dict in obj_dict_list:

            dorig = {'bps_object': [],
                    'verts_object': [],
                    'mesh_object': [],
                    'rotmat': [],
                    'batch_sizes': []}

            grabnet_bs = 3000 if obj_dict['obj_name'] in ['apple', 'toothpaste'] else 6000

            dorig['batch_sizes'].append(grabnet_bs)
            
            verts_obj = obj_dict['vertices'][0]
            bps_object = self.grabnet_bps.encode(torch.from_numpy(verts_obj), feature_type='dists')['dists']

            dorig['rotmat'].append(obj_dict['rotmat'][0])
            dorig['bps_object'].append(bps_object.to(grabnet.device).expand(grabnet_bs, -1)) 
            dorig['mesh_object'].append(Mesh(verts_obj, obj_dict['faces']))
            dorig['verts_object'] = torch.from_numpy(verts_obj.astype(np.float32)).unsqueeze(0).expand(grabnet_bs, -1, 3)
            

            MANO_TIP_IDS = {'thumb': 744, 'index': 320, 'middle': 443, 'ring': 554, 'pinky': 671}
             
            # we need to create MANO again beciase we need to set the batch size of the right hand model
            rh_model = smplx.create('data/body_models',
                                    model_type='mano',
                                    num_betas=10,
                                    gender='neutral',
                                    is_rhand=True,
                                    flat_hand_mean=True,
                                    vertex_ids=None,
                                    batch_size=grabnet_bs,
                                    use_pca=False).to(device)
            grabnet.refine_net.rhm_train = rh_model
 
            with torch.no_grad():

                bps_object = torch.cat(dorig['bps_object'])

                drec_cnet = grabnet.coarse_net.sample_poses(bps_object)
                rh_gen_cnet = grabnet.refine_net.rhm_train(**drec_cnet)
    
                verts_rh_gen_cnet = rh_gen_cnet.vertices
                joints_rh_gen_cnet = rh_gen_cnet.joints

                _, h2o, _ = point2point_signed(verts_rh_gen_cnet, dorig['verts_object'].to(device))
    
                # prepare refirnement net input
                drec_cnet['trans_rhand_f'] = drec_cnet['transl']
                drec_cnet['global_orient_rhand_rotmat_f'] = transform_body_pose(drec_cnet['global_orient'], 'aa->rot')
                drec_cnet['fpose_rhand_rotmat_f'] = transform_body_pose(drec_cnet['hand_pose'], 'aa->rot') 
                drec_cnet['verts_object'] = dorig['verts_object'].to(device)
                drec_cnet['h2o_dist']= h2o.abs()
                drec_cnet['joints'] = joints_rh_gen_cnet
                drec_cnet['vertices'] = verts_rh_gen_cnet
    
                drec_rnet = grabnet.refine_net(**drec_cnet)

                rh_gen_rnet = grabnet.refine_net.rhm_train(**drec_rnet)
                
                verts_rh_gen_rnet = rh_gen_rnet.vertices.cpu()
                joints_rh_gen_rnet = torch.cat([rh_gen_rnet.joints, rh_gen_rnet.vertices[:, list(MANO_TIP_IDS.values())]], dim=1) 
 
                grabnet_output.append({'joints': joints_rh_gen_rnet, 
                                       'vertices': verts_rh_gen_rnet,
                                       'global_orient': drec_rnet['global_orient'],
                                       'hand_pose': drec_cnet['hand_pose']})
            
  
        return grabnet_output
        
    def optimize(self):

        self.motion_diff_model.eval() 
        
        model_kwargs = {}

        all_motions, obs_list = [], []
        text_id = 0
        
        for val_data_batch in self.dl:
            
            val_data_batch = {k: v.to(device) if torch.is_tensor(v) else v
                for k, v in val_data_batch.items()}
      
            diffout = self.motion_diff_model.norm_and_cat(val_data_batch, model_cfg.input_feats).detach()
            
            gt_diffout_dict = self.motion_diff_model.diffout2motion(diffout)
       
            gt_motion_dict, gt_camera_dict = self.motion_diff_model.rot2xyz(gt_diffout_dict, 
                                                body_vtemp=val_data_batch["datum"]["body_vtemp"], 
                                                betas=val_data_batch["datum"]["betas"],
                                                cpu_flag=True,
                                                add_fingertips=True,
                                                gender_list=val_data_batch["datum"]["gender"])
        
             
            task_info = {"task": self.task, 
                         "device": device, 
                         "joint_num": self.NUM_JOINTS,
                         "gen_frames": optimization_cfg.window,
                         "gen_batch_size": diffout.shape[0],
                         "initial_motion": torch.from_numpy(gt_motion_dict['joints']).to(device)}
             

            t_pose = self.get_t_pose(val_data_batch = val_data_batch)

             
            if self.task == "self_contact_optimization":
 
                task_info['t_pose'] = t_pose
                task_info['client'] = self.client
                task_info['interaction_dict'] = self.interaction_dict
                task_info['text_id'] = np.arange(text_id, text_id + diffout.shape[0])
    
                text_id += diffout.shape[0]
                
            #### Prepare everything for the task ####
            target_opt_dict, target_mask_dict, _, is_noise_init, instruction_dict_list = dno_helper.prepare_task(task_info)
            
            if instruction_dict_list is None:
                print(text_id)
                continue

            

            #### Noise Optimization Config ####       
            noise_opt_list = []

             
            for stg_i in range(self.NUMBER_OF_STAGES):
            
                stg_i_grad_mask = prepare_grad_masks(eval(f"optimization_cfg.stg{stg_i+1}.parameter_grad_mask_str"),
                                                    shape=(diffout.shape[0], optimization_cfg.window, model_cfg.nfeats),
                                                    device=device)
                stg_i_lock_mask = prepare_lock_masks(eval(f"optimization_cfg.stg{stg_i+1}.parameter_lock_mask_str"), 
                                                          shape=(diffout.shape[0], optimization_cfg.window, model_cfg.nfeats),
                                                          device=device)
                
                noise_opt_list.append(DNOOptions(**eval(f"optimization_cfg.stg{stg_i+1}"), 
                                                parameter_grad_mask=stg_i_grad_mask,
                                                parameter_lock_mask=stg_i_lock_mask))
                
            # At this point, we need to have (1) target_opt_dict, (2) target_mask_dict, (3) kframes, (4, optional) initial motion     
            gen_shape = [diffout.shape[0], optimization_cfg.window, model_cfg.nfeats]
            cur_zt = torch.zeros(gen_shape).to(device)
           
            diffusion = create_diffusion(timestep_respacing=f"ddim{self.num_ode_steps}",
                                        learn_sigma=False,
                                        sigma_small=True,
                                        diffusion_steps=self.motion_diff_model.diff_params.num_train_timesteps,
                                        noise_schedule=self.motion_diff_model.diff_params.noise_schedule,
                                        predict_xstart=False if self.motion_diff_model.diff_params.predict_type == 'noise' else True,
                                        **loss_cfg)      
            
            model_kwargs['obj_dict_list'] = []
            model_kwargs['stats_dict'] = self.motion_diff_model.stats['concatenated_features']


            for _k_ in range(diffout.shape[0]):
                
                if self.task == "object_guided_optimization":

                    obj_trans = val_data_batch['datum']["obj_trans"][_k_]

                    R = get_z_rot(val_data_batch['datum']['rots'][_k_, 0:1, :3], in_format="aa")[0]

                    obj_trans = torch.einsum('...ij,...j->...i', R.T, obj_trans)
                    obj_trans_zeros = torch.zeros_like(obj_trans)
                    obj_rotmat_zeros = torch.zeros_like(val_data_batch['datum']["obj_orient"][_k_])

                    object_dict = AmassDataset.load_object_geometry(val_data_batch['datum']["obj_name"][_k_], 
                                        torch.tensor([1.0]),
                                        obj_trans_zeros, 
                                        obj_rotmat_zeros, 
                                        val_data_batch['datum']["obj_arti"][_k_], 
                                        val_data_batch['dataset_name'][_k_],
                                        load_simplified=True)
                                                    
                    object_dict['trans'] = obj_trans

                    object_dict['obj_name'] = val_data_batch['datum']["obj_name"][_k_]

                    object_dict['body_faces'] = self.motion_diff_model.rot2xyz.bm_dict['neutral'].faces
                    object_dict['rotmat'] = transform_body_pose(obj_rotmat_zeros, "aa->rot")
 
                    model_kwargs['obj_dict_list'].append(object_dict)
                    
                else:
                    object_dict = {}

     
            if self.task == "self_contact_optimization":

                # add faces and commands for the self-penetration loss. 
                model_kwargs['self_contact_dict'] = {'instruction_dict_list': instruction_dict_list, 
                                    'faces': self.motion_diff_model.rot2xyz.bm_dict['neutral'].faces}
                
                joblib.dump(task_info['interaction_dict'], self.interaction_dict_path)

                
      
            elif self.task == "object_guided_optimization":
                 
                grabnet_data_dict_list = self.forward_grabnet(model_kwargs['obj_dict_list'])

                movement_time_idx = self.detect_object_movement_times(obj_rot=val_data_batch['datum']['obj_orient'], 
                                                                    obj_trans=val_data_batch['datum']['obj_trans'])
                
               
                movement_time_idx = val_data_batch['datum']['obj_moving_frames'] 


                 
                repeated_joints_list, repeated_verts_list = [], []
 
                # find fingertips location
                R = get_z_rot(val_data_batch['datum']['rots'][:, 0, :3], in_format="aa")  # (B, 3, 3)

                new_obj_trans = torch.einsum('...ij, ...kj->...ki', R.transpose(1, 2), val_data_batch['datum']["obj_trans"])

                obj_rotmat_can = transform_body_pose(val_data_batch['datum']['obj_orient'], 'aa->rot').to(device)
                # Rotate canonical object orientation into the diffout frame
                R_z_inv = R.transpose(-1, -2)  # (B, 3, 3)
                
                obj_rotmat = torch.einsum('bij,bfjk->bfik', R_z_inv.to(device), obj_rotmat_can)
                
                closest_index_list = []
                target_mask_dict['joints'][...] = 0
                target_opt_dict['joints'][...] = 0 

                rotated_targets = [] 


                for _k_ in range(diffout.shape[0]):

                    repeated_joints_list = grabnet_data_dict_list[_k_]['joints'].unsqueeze(1).expand((-1, optimization_cfg.window, -1, -1)) # (N, T, J, 3)
                    repeated_verts_list = grabnet_data_dict_list[_k_]['vertices']   # (N, V, 3)
 
                    canonical_obj_rotmat_k = obj_rotmat[_k_].expand((repeated_joints_list.shape[0], optimization_cfg.window, 3, 3))
                
                    # ij normally 
                    rotated_targets_k_ = torch.einsum('btji,btki->btkj', canonical_obj_rotmat_k, repeated_joints_list) \
                                                        + new_obj_trans[None, _k_, :, None].to(device)
                    
                
                    # closest_index = self.find_closest_index_angle(grabnet_data_dict_list[_k_], val_data_batch, _k_)  
                    closest_index = self.find_closest_index_distance(t_pose[_k_], rotated_targets_k_)
                    closest_index_list.append(closest_index)
        
                    seq_id = str(val_data_batch['id'][_k_].item()).zfill(6)
                 
                    # save the grasp
                    trimesh.util.concatenate([
                        trimesh.Trimesh(vertices=repeated_verts_list[closest_index] @ canonical_obj_rotmat_k[closest_index][0].cpu(), 
                                        faces=self.motion_diff_model.rot2xyz.hand_dict['right'].faces),
                        trimesh.Trimesh(vertices=model_kwargs['obj_dict_list'][_k_]['vertices'][0] @ canonical_obj_rotmat_k[closest_index][0].cpu().numpy(), 
                                        faces=model_kwargs['obj_dict_list'][_k_]['faces'].astype(np.uint32))]) \
                                .export(os.path.join(self.vis_folder, f"{seq_id}_hand_object.obj"))


                    # save the grasp with the closest index
                    rotated_targets.append(rotated_targets_k_[closest_index:closest_index+1])

                # Get batch and time indices where movement is True
                b_idx, t_idx = torch.where(movement_time_idx)  # shapes: [N], [N]

                j_idx = torch.tensor(GLOBAL_RHAND_TIPS_INDICES, device=device).repeat(b_idx.size(0), 1).view(-1) # shape [N * J]

                g_idx = torch.arange(len(GLOBAL_RHAND_TIPS_INDICES), device=device).repeat(b_idx.size(0)) 
                b_idx = b_idx.repeat_interleave(len(GLOBAL_RHAND_TIPS_INDICES))      # shape [N * J]
                t_idx = t_idx.repeat_interleave(len(GLOBAL_RHAND_TIPS_INDICES))      # shape [N * J]

                target_mask_dict['joints'][b_idx, t_idx, j_idx, :] = 1
                rotated_targets = torch.cat(rotated_targets, dim=0)

                target_opt_dict['joints'][b_idx, t_idx, j_idx, :] = rotated_targets[b_idx, t_idx, g_idx, :]

            elif self.task == "clutch_optimization":

                
                target_mask_dict['joints'][...] = 0
                target_opt_dict['joints'][...] = 0

                for _k_ in range(diffout.shape[0]):

                    clutch_dict, _ = self.motion_diff_model.rot2xyz.forward_hand(clutch_dict_all[_k_], pos_offset=None, cpu_flag=False, **model_kwargs)
            
                    target_mask_dict['vertices'][_k_, :, self.base_hand_vertex_ids] = 1
                    
                    target_opt_dict['vertices'][_k_, :, self.base_hand_vertex_ids] = \
                        torch.cat([clutch_dict['left']['vertices'], 
                                   clutch_dict['right']['vertices']], dim=2)[:, :optimization_cfg.window, :]
            

            def criterion(x, latent_z):
                return CondKeyLocationsLoss(
                    obs_list=obs_list,
                    target=target_opt_dict,
                    target_mask=target_mask_dict,       
                    transform=functools.partial(self.motion_diff_model.rot2xyz,  
                                                    body_vtemp=val_data_batch["datum"]['body_vtemp'], 
                                                    betas=val_data_batch["datum"]['betas'],
                                                    add_fingertips=self.add_fingertips_flag),
                    
                    use_mse_loss=optimization_cfg.use_mse_loss,
                    inv_transform=self.motion_diff_model.diffout2motion,
                    penetration_dict=self.data_wrapper.watertight_conversion_dict,
                    ).__call__(x, latent_z, **model_kwargs) 
            

            def solver(z):
                return ddim_loop_with_gradient(
                    diffusion,
                    self.motion_diff_model.denoiser,
                    cur_zt.shape,
                    model_kwargs={'detach_condition': False},
                    noise=z,
                    clip_denoised=False)
                
            out_dict_list = []
                        
            ######## Main optimization loop #######
            for stg_i in range(self.NUMBER_OF_STAGES):  
                
                out_dict = DNO(model=solver, 
                               start_z=cur_zt, 
                               criterion=criterion, 
                               conf=noise_opt_list[stg_i])()
                
     
                out_dict_list.append({"hist": out_dict['hist'].copy(),
                                      "min_loss_x": copy.deepcopy(out_dict['min_loss_x']),
                                      "loss_coef_dict": copy.deepcopy(out_dict['loss_coef_dict']),
                                      "x": copy.deepcopy(out_dict['x'])})
                
                cur_zt = out_dict["min_loss_z"].detach().clone()
          
            final_out_x = out_dict["x"].detach().clone()
            final_out_min_loss_x = out_dict["min_loss_x"].detach().clone()
                
            gen_diffout_dict = self.motion_diff_model.diffout2motion(final_out_min_loss_x)
            # gen_diffout_dict_minloss = self.motion_diff_model.diffout2motion(out_dict_list[-2]["min_loss_x"])

            gen_motion_dict, gen_camera_dict = self.motion_diff_model.rot2xyz(gen_diffout_dict, 
                                                body_vtemp=val_data_batch["datum"]["body_vtemp"], 
                                                betas=val_data_batch["datum"]["betas"],
                                                cpu_flag=True, 
                                                add_fingertips=False,
                                                gender_list=val_data_batch["datum"]["gender"])
             
            
             
            for _k_ in range(diffout.shape[0]):
     
                seq_id = str(val_data_batch['id'][_k_].item()).zfill(6)
                
                print(f'Plotting losses for sequence {seq_id}')
                
                for stage_id in range(self.NUMBER_OF_STAGES):

                    plot_loss(out_dict_list[stage_id]["hist"][_k_], 
                              out_dict_list[stage_id]["loss_coef_dict"],
                              os.path.join(self.vis_folder, f"{seq_id}_{stage_id+1}.png"))
                    torch.save(out_dict_list[stage_id]["min_loss_x"], os.path.join(self.results_folder, f"{seq_id}_{stage_id+1}_optimized_z.pt"))
                    torch.save(out_dict_list[stage_id]["x"], os.path.join(self.results_folder, f"{seq_id}_{stage_id+1}_optimized_x.pt"))
                

                assert not val_data_batch['datum']["augment_flag"][_k_].item(), "Augmented data is not supported"

                target_vis_dict = {"target_location": target_opt_dict['joints'][_k_].detach().cpu().numpy(),
                                    "target_mask": target_mask_dict['joints'][_k_].detach().cpu().numpy()}
                
                gen_motion_dict_k = {'faces': gen_motion_dict['faces'],
                                     'vertices': gen_motion_dict['vertices'][_k_], 
                                     'joints': gen_motion_dict['joints'][_k_]}
              
                gen_camera_dict_k = {'camera_rot': gen_camera_dict['camera_rot'][_k_],
                                     'camera_transl': gen_camera_dict['camera_transl'][_k_],
                                     'coef': 1.9}

                if self.task == 'self_contact_optimization':
                    gen_camera_dict_k['rel_pos'] = (0, -2.0, gen_camera_dict['camera_transl'][_k_][:, 2].mean().item() + 0.1)


                gen_skeleton_dict_k = {'positions': gen_motion_dict['joints'][_k_],
                                       'contact_masks': gen_diffout_dict['contact_masks'][_k_].cpu().numpy()}
   
            
                object_dict = {}

                rendering_scale = 2.0 if self.task == 'object_guided_optimization' else 1.7

                text_for_vid = ''
                mano_skeleton_dict_k = {}
                min_self_contact_loss = None
                mesh_list = [gen_motion_dict_k]
                command_list = []


                if self.task == 'object_guided_optimization':

                    R = get_z_rot(val_data_batch['datum']['rots'][_k_, 0:1, :3], in_format="aa")[0]

                    obj_trans = torch.einsum('...ij,...j->...i', R.T, val_data_batch['datum']["obj_trans"][_k_])

                    obj_orient_can = val_data_batch['datum']["obj_orient"][_k_]
                    R_obj_can = transform_body_pose(obj_orient_can, "aa->rot")
                    obj_orient = matrix_to_axis_angle(torch.einsum('ij,tjk->tik', R.T, R_obj_can))

                    object_dict = AmassDataset.load_object_geometry(val_data_batch['datum']["obj_name"][_k_],
                                        torch.tensor([1.0]),
                                        obj_trans,
                                        obj_orient,
                                        val_data_batch['datum']["obj_arti"][_k_],
                                        val_data_batch["dataset_name"][_k_],
                                        load_simplified=True)

                    object_dict['pose'] = obj_orient
                    object_dict['trans'] = obj_trans
        
                    # target_mask is of shape (B, T, 21, 3)
                    mano_skeleton_mask = torch.where(target_mask_dict['joints'][_k_].sum(-2) == 0, 0, 1).unsqueeze(-2)
                    
                    mano_skeleton_dict_k = {'positions': rotated_targets[_k_].cpu().numpy(),
                                   'contact_masks': gen_diffout_dict['contact_masks'][_k_].cpu().numpy(),
                                   'skeleton_masks': mano_skeleton_mask.detach().cpu().numpy(), # (T, 1, 3)
                                   'render_skeleton': True}
                        
                elif self.task == 'clutch_optimization':

                    mesh_list.append({'vertices': target_opt_dict['vertices'][_k_, :, self.base_rhand_vertex_ids].detach().cpu().numpy(), 
                                      'faces': self.motion_diff_model.rot2xyz.hand_dict['right'].faces}) 
                    
                    mesh_list.append({'vertices': target_opt_dict['vertices'][_k_, :, self.base_lhand_vertex_ids].detach().cpu().numpy(), 
                                    'faces': self.motion_diff_model.rot2xyz.hand_dict['left'].faces})

                else:
 
                    text_for_vid = instruction_dict_list[_k_]['text']
                    command_list = instruction_dict_list[_k_]['command_list']
        
                    min_self_contact_loss = min(out_dict_list[stage_id]["hist"][_k_]['self_contact_loss'])
                            
                    target_vis_dict['target_mask'][...] = 0
                    target_vis_dict['use_different_colours'] = True
                    
                    # this is to make sure contact points are displayed.
                    for frame_list, vertex_list in command_list:

                        concat_vertex_idx_list = list(itertools.chain.from_iterable(vertex_list))
    
                        target_vis_dict['target_mask'][frame_list[0]:frame_list[1], :len(concat_vertex_idx_list)] = 1 
 
                        target_vis_dict['target_location'][frame_list[0]:frame_list[1], :len(concat_vertex_idx_list)] = \
                            gen_motion_dict_k['vertices'][frame_list[0]:frame_list[1], concat_vertex_idx_list] 

                # save metrics
                metric_dict_k = save_metrics(gen_motion_dict_k, 
                                            {'command_list': command_list, 'object_dict': object_dict}, 
                                            out_dict) 
                metric_dict_k['text_for_vid'] = text_for_vid  
         
                joblib.dump(metric_dict_k, os.path.join(self.results_folder, f"{seq_id}_metrics.p"))


                tarfile_path = os.path.join(self.results_folder, f"{seq_id}.tar")
 
                 
                # save text if exists 
                joblib.dump({'vertices': gen_motion_dict_k['vertices'],
                            'faces': gen_motion_dict_k['faces'],
                            'joints': gen_motion_dict_k['joints'],
                            'camera_transl': gen_camera_dict_k['camera_transl'],
                            'camera_rot': gen_camera_dict_k['camera_rot'], 
                            'text_for_vid': text_for_vid,
                            'self_penetration_loss': metric_dict_k['self_penetration_loss'],
                            'min_self_contact_loss': min_self_contact_loss,
                            'target_loc': target_vis_dict['target_location'],
                            'target_mask': target_vis_dict['target_mask'],
                            'object_dict': object_dict}, 
                            tarfile_path)
                        
                
             
                # render both video and images
                # video_cmd = f'python src/render/render_single_fusion.py dataset_path={tarfile_path} mode=video savedir={self.vis_folder}_blender'
                # subprocess.Popen(video_cmd, shell=True)
                # image_cmd = f'python src/render/render_single_fusion.py num={self.num_poses} dataset_path={tarfile_path} mode=sequence savedir={self.vis_folder}_blender'
                # subprocess.Popen(image_cmd, shell=True)

                print(f"saving visualizations to [{self.vis_folder}]...")
                self.renderer.render_motion(mesh_list, 
                                            color = [color1, color2, color3],
                                            target_dict=target_vis_dict,
                                            object_dict=object_dict,
                                            skeleton_dict=mano_skeleton_dict_k,
                                            camera_dict=gen_camera_dict_k,
                                            rendering_scale=rendering_scale,
                                            text_for_vid=text_for_vid,
                                            filename=f"{self.vis_folder}/{seq_id}_pred")
                
                if self.task == 'object_guided_optimization':

                    gen_camera_dict_k['coef'] = 0.0
                    gen_camera_dict_k['camera_transl'] = obj_trans
                    gen_camera_dict_k['lock2object'] = True
                     
                    self.renderer.render_motion(mesh_list, 
                                            color = [color1, color2, color3],
                                            target_dict={},
                                            object_dict=object_dict,
                                            skeleton_dict={},
                                            camera_dict=gen_camera_dict_k,
                                            rendering_scale=rendering_scale,
                                            filename=f"{self.vis_folder}/{seq_id}_pred_hand")
     
                    merge_videos([f"{self.vis_folder}/{seq_id}_pred.mp4", f"{self.vis_folder}/{seq_id}_pred_hand.mp4"],
                                 output_path=f"{self.vis_folder}/{seq_id}_pred_merged.mp4")
                    
                    
                # if not os.path.exists(f"{self.vis_folder}/{seq_id}_gt.mp4"):
                #     self.renderer.render_motion([gt_motion_dict_k], 
                #                                 color = [color1, color2, color3],
                #                                 target_dict={},
                #                                 object_dict=object_dict,
                #                                 skeleton_dict={},
                #                                 camera_dict=gt_camera_dict_k,
                #                                 rendering_scale=rendering_scale,
                #                                 filename=f"{self.vis_folder}/{seq_id}_gt")
              
 
                
                if val_data_batch["dataset_name"][_k_] in self.data_wrapper.object_dataset_list and self.render_hand: 
                    gen_camera_dict_k['coef'] = 0.0
                    gen_camera_dict_k['camera_transl'] = obj_trans
                    gen_camera_dict_k['lock2object'] = True
                     
                    self.renderer.render_motion([gen_motion_dict_k], 
                                            color = [color1, color2, color3],
                                            target_dict=target_dict,
                                            object_dict=object_dict,
                                            skeleton_dict={},
                                            camera_dict=gen_camera_dict_k,
                                            rendering_scale=rendering_scale,
                                            filename=f"{self.vis_folder}/{seq_id}_pred_hand")
                    
  
                    # render side by side. 
                    # merge_videos([f"{self.vis_folder}/{seq_id}_pred.mp4", f"{self.vis_folder}/{seq_id}_pred_hand.mp4"],
                    #              output_path=f"{self.vis_folder}/{seq_id}_pred_merged.mp4")

                all_motions.extend(gen_diffout_dict['full_motion_unnorm'])
   
        
        # [bs * num_dump_step, 1, 3, T]
        all_motions = torch.cat(all_motions, axis=0).detach().cpu().numpy()  
  
        
        np.save(os.path.join(self.vis_folder, "results.npy"), {"motion": all_motions})
        print(f"[Done] Results are at [{os.path.abspath(self.vis_folder)}]")
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer_file', type=str, required=True) 

    opt = parser.parse_args()
 
    optimization_cfg = OmegaConf.load(opt.optimizer_file)
    model_cfg = OmegaConf.load(optimization_cfg.model_cfg)
    data_cfg = OmegaConf.load(optimization_cfg.data_cfg)
    diffusion_cfg = OmegaConf.load(optimization_cfg.diffusion_cfg)
    loss_cfg = OmegaConf.load(optimization_cfg.loss_cfg)

    seed_everything(optimization_cfg.seed)
    
    device = f"cuda:{optimization_cfg.device}" if torch.cuda.is_available() else "cpu"


    assert optimization_cfg.optimizer in ['ADAMW', 'LBFGS']

    if optimization_cfg.optimizer == 'ADAMW':
        from optim.dno_adam import DNO
    else:
        from dno_lbfgs import DNO


    # requires gpt key 
    if optimization_cfg.obj_contact_optimization:
        # add grabnet 
        sys.path.append(os.path.join(os.getcwd(), 'external/GrabNet'))
        
        from grabnet.tests.tester import Tester
        from grabnet.train.trainer import Trainer
        from grabnet.tools.vis_tools import points_to_spheres
        from grabnet.tools.train_tools import point2point_signed
        from grabnet.tests.grab_new_objects import vis_results, load_obj_verts
        from grabnet.tools.cfg_parser import Config


        cwd = os.getcwd()

        work_dir = cwd + '/external/GrabNet/tests'
        bps_dir   = 'external/GrabNet/grabnet/configs/bps.npz'
        best_cnet = 'external/GrabNet/grabnet/models/coarsenet.pt'
        best_rnet = 'external/GrabNet/grabnet/models/refinenet.pt'
        cfg_path = 'external/GrabNet/grabnet/configs/grabnet_cfg.yaml'

        cfg = Config(default_cfg_path=cfg_path, 
                    **{'work_dir':work_dir,
                       'best_cnet': best_cnet,
                       'best_rnet': best_rnet,
                       'bps_dir': bps_dir})
        
        cfg.dataset_dir = 'external/GrabNet/data/grabnet_data'
        cfg.rhm_path = 'data/body_models/mano'

        # grabnet = Trainer(cfg=cfg, inference=True)
        grabnet = Tester(cfg=cfg)

        grabnet.coarse_net.eval()
        grabnet.refine_net.eval()

        # remove grabnet
        sys.path = [elem for elem in sys.path if 'external' not in elem]
    
    elif optimization_cfg.self_contact_optimization:
        from src.utils.llm_utils import self_interaction_length
        from src.llm.gpt_self_interaction import get_openai_client

    elif optimization_cfg.use_clutch:
        clutch_dict_all = joblib.load('data/clutch/clutch_dict_processed.p')

    OptimizerWrapper()
