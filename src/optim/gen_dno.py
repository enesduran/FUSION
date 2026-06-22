import os
import sys
import copy
import torch
import smplx
import joblib
import shutil
import trimesh
import argparse
import functools
import subprocess
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from ema_pytorch import EMA
from omegaconf import OmegaConf
from condition import CondKeyLocationsLoss
 

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


import dno_helper
from optim.dno_adam import DNOOptions
from utils.genutils import seed_everything
from src.diffusion import create_diffusion
from src.data.amass_dataset import AmassDataset
from src.utils.process_utils import BRANCH_NAME
from src.utils.metric_utils import save_metrics
from src.data.amass_wrapper import AmassWrapper
from src.render.mesh_viz import RendererWrapper
from src.model.base_diffusion import MotionDiffusion
from src.utils.transforms3d import transform_body_pose, get_z_rot, matrix_to_axis_angle
from src.utils.viz_utils import merge_videos, pack_to_render, color1, color2, color3, obj_color
from src.utils.metric_utils import calculate_traj_loc_error, calculate_skating_ratio, THRESH_LIST
from optim.dno_common import get_number_of_stages, configure_model, setup_output_folders, load_checkpoint
from src.utils.optim_utils import prepare_grad_masks, prepare_lock_masks, ddim_loop_with_gradient, ddim_invert, plot_loss


class OptimizerWrapper:

    def __init__(self, ema_decay=0.995,
                 ema_update_every=10,):
        
        ### Task selection ###
        self.task = optimization_cfg.task
        self.demo_mode = bool(optimization_cfg.demo_mode) if 'demo_mode' in optimization_cfg else False
  
        self.renderer = RendererWrapper(path2body_models=data_cfg.smplx_path)
        optimization_cfg.use_ddim = True
        self.render_hand = True
    
        print("Loading dataset...")
        self.prep_dataloader()
        
        configure_model(model_cfg, data_cfg, diffusion_cfg, self.data_wrapper)
        self.NUMBER_OF_STAGES = get_number_of_stages(optimization_cfg)
        
        print("Creating model and diffusion...")
 
        self.motion_diff_model = MotionDiffusion(**model_cfg, **loss_cfg)
        self.ema = EMA(self.motion_diff_model.denoiser, beta=ema_decay, update_every=ema_update_every)

        vis_res_suffix = f"{self.task}_vis_res"
        quant_res_suffix = f"{self.task}_quant_res"

        self.weights_folder, self.vis_folder, self.results_folder = setup_output_folders(
            optimization_cfg, vis_res_suffix, quant_res_suffix, opt.optimizer_file)

        self.NUM_JOINTS = self.motion_diff_model.rot2xyz.bm_dict['neutral'].NUM_JOINTS + 1   # 55
        self.add_fingertips_flag = True

        self.step = load_checkpoint(self.ema, self.motion_diff_model.denoiser,
                                    self.weights_folder, optimization_cfg.milestone,
                                    model_cfg.pretrained_path)
            
        self.num_ode_steps = 10
                
        self.optimize()
        
    @staticmethod
    def load_body_vtemp(body_vtemp):    
        return None if body_vtemp == 'default' else torch.tensor(trimesh.load(file_obj=body_vtemp).vertices).to(device).float()
    
     
    def prep_dataloader(self):
        data_cfg.load_splits = optimization_cfg.load_splits
        data_cfg.device = device 
        
        data_cfg.train_batch_size = optimization_cfg.train_batch_size
        data_cfg.val_batch_size = optimization_cfg.val_batch_size
        data_cfg.demo_mode = optimization_cfg.demo_mode 
        
        if self.demo_mode:      
            data_cfg.motion_filepath = optimization_cfg.motion_filepath
        
            data_cfg.include_pose_augmentation = False
            data_cfg.include_time_augmentation = False
                
        self.data_wrapper = AmassWrapper(**data_cfg)

        if 'val' in data_cfg.load_splits:
            self.dl = self.data_wrapper.val_dataloader()
        elif 'test' in data_cfg.load_splits:
            self.dl = self.data_wrapper.test_dataloader()
        else:
            self.dl = self.data_wrapper.train_dataloader()

    def optimize(self):
        self.motion_diff_model.eval() 
        
        model_kwargs = {}
        all_motions, obs_list = [], []

        #### Noise Optimization Config ####       
        noise_opt_list = []

        print('Batch size',  self.dl.batch_size)
     
        for stg_i in range(self.NUMBER_OF_STAGES):
                
            stg_i_grad_mask = prepare_grad_masks(eval(f"optimization_cfg.stg{stg_i+1}.parameter_grad_mask_str"),
                                                shape=(self.dl.batch_size, optimization_cfg.window, model_cfg.nfeats),
                                                device=device)
            stg_i_lock_mask = prepare_lock_masks(eval(f"optimization_cfg.stg{stg_i+1}.parameter_lock_mask_str"), 
                                                        shape=(self.dl.batch_size, optimization_cfg.window, model_cfg.nfeats),
                                                        device=device)
            
            noise_opt_list.append(DNOOptions(**eval(f"optimization_cfg.stg{stg_i+1}"), 
                                            parameter_grad_mask=stg_i_grad_mask,
                                            parameter_lock_mask=stg_i_lock_mask))
        
 

        for val_data_batch in self.dl:
            
            val_data_batch = {k: v.to(device) if torch.is_tensor(v) else v
                for k, v in val_data_batch.items()}

            # since SMPLX is not symmetric, object datasets wont be accurate for rendering. 
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
                         "gen_batch_size": self.dl.batch_size,
                         "initial_motion": torch.from_numpy(gt_motion_dict['joints']).to(device)}


            #### Prepare everything for the task ####
            target_opt_dict, target_mask_dict, keyframes, is_noise_init, obs_list = dno_helper.prepare_task(task_info)
   
            # Do inversion to get the initial noise for editing
            if not is_noise_init:
                diffusion_invert = create_diffusion(timestep_respacing=f"ddim{self.num_ode_steps}",
                                        learn_sigma=False,
                                        sigma_small=True,
                                        diffusion_steps=self.motion_diff_model.diff_params.num_train_timesteps,
                                        noise_schedule=self.motion_diff_model.diff_params.noise_schedule,
                                        predict_xstart=False if self.motion_diff_model.diff_params.predict_type == 'noise' else True, 
                                        **loss_cfg) # noise vs sample
                
                dump_steps = [0, 5, 10, 20, 30, 40, 49]
    
                motion_to_invert = diffout.clone()

                ######## DDIM inversion ########
                inv_noise, pred_x0_list = ddim_invert(
                    diffusion_invert,
                    self.motion_diff_model.denoiser,
                    motion_to_invert,
                    model_kwargs=model_kwargs,
                    dump_steps=dump_steps,
                    num_inference_steps=10,
                    clip_denoised=False)
                
                cur_zt = inv_noise.detach().clone()
                
            else:
                gen_shape = [diffout.shape[0], optimization_cfg.window, model_cfg.nfeats]
                cur_zt = torch.randn(gen_shape).to(device)
           
            diffusion = create_diffusion(timestep_respacing=f"ddim{self.num_ode_steps}",
                                        learn_sigma=False,
                                        sigma_small=True,
                                        diffusion_steps=self.motion_diff_model.diff_params.num_train_timesteps,
                                        noise_schedule=self.motion_diff_model.diff_params.noise_schedule,
                                        predict_xstart=False if self.motion_diff_model.diff_params.predict_type == 'noise' else True,
                                        **loss_cfg)              
  
            model_kwargs['obj_dict_list'] = []

            for _k_ in range(self.dl.batch_size):
                model_kwargs['obj_dict_list'].append({})
                
            def criterion(x, latent_z):
                return CondKeyLocationsLoss(
                    target=target_opt_dict,
                    target_mask=target_mask_dict,       
                    transform=functools.partial(self.motion_diff_model.rot2xyz,  
                                                    body_vtemp=val_data_batch["datum"]['body_vtemp'], 
                                                    betas=val_data_batch["datum"]['betas'],
                                                    add_fingertips=self.add_fingertips_flag),
                    
                    inv_transform=self.motion_diff_model.diffout2motion,
                    use_mse_loss=optimization_cfg.use_mse_loss,
                    obs_list=obs_list).__call__(x, latent_z, **model_kwargs) 
            

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
                
            gen_diffout_dict = self.motion_diff_model.diffout2motion(final_out_x)
            # gen_diffout_dict_minloss = self.motion_diff_model.diffout2motion(out_dict_list[-2]["min_loss_x"])

            gen_motion_dict, gen_camera_dict = self.motion_diff_model.rot2xyz(gen_diffout_dict, 
                                                body_vtemp=val_data_batch["datum"]["body_vtemp"], 
                                                betas=val_data_batch["datum"]["betas"],
                                                cpu_flag=True, 
                                                add_fingertips=True,
                                                gender_list=val_data_batch["datum"]["gender"])
            

            for _k_ in range(self.dl.batch_size):
    
                
                seq_id = str(val_data_batch['id'][_k_].item()).zfill(6)
                
                print(f'Plotting losses for sequence {seq_id}')
                
                for stage_id in range(self.NUMBER_OF_STAGES):
                    plot_loss(out_dict_list[stage_id]["hist"][_k_], 
                              out_dict_list[stage_id]["loss_coef_dict"],
                              os.path.join(self.vis_folder, f"{seq_id}_{stage_id+1}.png"))
                    torch.save(out_dict_list[stage_id]["min_loss_x"], os.path.join(self.results_folder, f"{seq_id}_{stage_id+1}_optimized_z.pt"))
                    torch.save(out_dict_list[stage_id]["x"], os.path.join(self.results_folder, f"{seq_id}_{stage_id+1}_optimized_x.pt"))
                
                target_vis_dict = {"target_location": target_opt_dict['joints'][_k_].detach().cpu().numpy(),
                                    "target_mask": target_mask_dict['joints'][_k_].detach().cpu().numpy()}

                
                gen_motion_dict_k = {'faces': gen_motion_dict['faces'],
                                     'vertices': gen_motion_dict['vertices'][_k_], 
                                     'joints': gen_motion_dict['joints'][_k_]}
                gt_motion_dict_k = {'faces': gt_motion_dict['faces'],
                                     'vertices': gt_motion_dict['vertices'][_k_], 
                                     'joints': gt_motion_dict['joints'][_k_]}
                gen_camera_dict_k = {'camera_rot': gen_camera_dict['camera_rot'][_k_],
                                     'camera_transl': gen_camera_dict['camera_transl'][_k_],
                                     'coef': 1.6}
                gt_camera_dict_k = {'camera_rot': gt_camera_dict['camera_rot'][_k_],
                                     'camera_transl': gt_camera_dict['camera_transl'][_k_],
                                     'coef': 1.6}
                
                gen_skeleton_dict_k = {'positions': gen_motion_dict['joints'][_k_][:, :self.NUM_JOINTS],
                                   'contact_masks': gen_diffout_dict['contact_masks'][_k_].cpu().numpy()}
                gt_skeleton_dict_k = {'positions': gt_motion_dict['joints'][_k_][:, :self.NUM_JOINTS],
                                    'contact_masks': gt_diffout_dict['contact_masks'][_k_].cpu().numpy()}

                object_dict = {}
                
                datasetname = val_data_batch['datum']['body_dataset_name'][_k_]
                
                if datasetname in self.data_wrapper.object_dataset_list \
                    and self.task == 'keypoint_tracking':

                    # diffout2motion integrates Z-orient from identity, so the
                    # reconstructed body lives in a frame rotated by
                    # R_z_can[0]^{-1} w.r.t. the canonical frame.  The per-frame
                    # offset is tiny (one frame of Z velocity), but the
                    # body-object translation error is
                    #   (I - R_z_inv) @ (trans_can[t] - trans_can[0])
                    # which grows with body displacement and becomes visible for
                    # hand-object interaction.  Rotate canonical object data into
                    # the diffout2motion frame (same approach as gen_dno_app.py).
                    
                    R_z_inv = get_z_rot(val_data_batch['datum']['rots'][_k_, 0:1, :3], in_format="aa")[0].T
            
                    obj_trans_can = val_data_batch['datum']["obj_trans"][_k_]  # (T, 3)
                    obj_trans = torch.einsum('ij,tj->ti', R_z_inv, obj_trans_can)

                    obj_orient_can = val_data_batch['datum']["obj_orient"][_k_]  # (T, 3) aa
                    R_obj_can = transform_body_pose(obj_orient_can, "aa->rot")  # (T, 3, 3)
                    obj_orient = matrix_to_axis_angle(
                        torch.einsum('ij,tjk->tik', R_z_inv, R_obj_can))  # (T, 3)

                    is_augmented = bool(val_data_batch['datum']['augment_flag'][_k_])

                    print(f"Dataset: {datasetname}, Augmented: {is_augmented}")

                    obj_bottom_trans_can = val_data_batch['datum']['obj_bottom_trans'][_k_]
                    obj_bottom_trans = torch.einsum('ij,tj->ti', R_z_inv, obj_bottom_trans_can)
                    obj_bottom_orient_can = val_data_batch['datum']["obj_bottom_orient"][_k_]
                    R_bot_can = transform_body_pose(obj_bottom_orient_can, "aa->rot")
                    obj_bottom_orient = matrix_to_axis_angle(
                        torch.einsum('ij,tjk->tik', R_z_inv, R_bot_can))

                    object_dict = AmassDataset.load_object_geometry(
                        val_data_batch['datum']["obj_name"][_k_],
                        obj_scale=val_data_batch['datum']["obj_scale"][_k_],
                        obj_trans=obj_trans,
                        obj_rot=obj_orient,
                        obj_arti=val_data_batch['datum']["obj_arti"][_k_],
                        datasetname=datasetname,
                        obj_bottom_scale=val_data_batch['datum']['obj_bottom_scale'][_k_],
                        obj_bottom_trans=obj_bottom_trans,
                        obj_bottom_rot=obj_bottom_orient,
                        load_simplified=True,
                        reflect_x=is_augmented)
                    

                # save metrics
                metric_dict_k = save_metrics(gen_motion_dict_k, 
                                            {'command_list': [], 'object_dict': object_dict}, 
                                            out_dict)

                l2_error = np.linalg.norm(target_vis_dict['target_location'] - gen_motion_dict_k['joints'], axis=-1)
                l2_error = l2_error[target_vis_dict['target_mask'][:, :, 0]].reshape(optimization_cfg.window, -1)

                traj_err = calculate_traj_loc_error(l2_error, target_vis_dict['target_mask'][:, :, 0], THRESH_LIST)
                skating_ratio, skate_vel = calculate_skating_ratio(gen_motion_dict_k['joints'].transpose(1, 2, 0)[None])

                metric_dict_k['traj_err'] = traj_err
                metric_dict_k['skating_ratio'] = skating_ratio
                metric_dict_k['skate_vel'] = skate_vel

                joblib.dump(metric_dict_k, os.path.join(self.results_folder, f"{seq_id}_metrics.p"))
 

                print(f"saving visualizations to [{self.vis_folder}]...")
                self.renderer.render_motion([gen_motion_dict_k], 
                                            color = [color1, color2, color3],
                                            target_dict=target_vis_dict,
                                            object_dict=object_dict,
                                            skeleton_dict=gen_skeleton_dict_k,
                                            camera_dict=gen_camera_dict_k,
                                            filename=f"{self.vis_folder}/{seq_id}_pred")

                if datasetname in self.data_wrapper.object_dataset_list and self.render_hand: 
                    gen_camera_dict_k['coef'] = 0.1
                    gen_camera_dict_k['camera_transl'] = obj_trans
                    gen_camera_dict_k['lock2object'] = True
                    
                    self.renderer.render_motion([gen_motion_dict_k], 
                                            color = [color1, color2, color3],
                                            target_dict=target_vis_dict,
                                            object_dict=object_dict,
                                            skeleton_dict=gen_skeleton_dict_k,
                                            camera_dict=gen_camera_dict_k,
                                            filename=f"{self.vis_folder}/{seq_id}_pred_hand")
                    
  
                    # render side by side. 
                    merge_videos([f"{self.vis_folder}/{seq_id}_pred.mp4", f"{self.vis_folder}/{seq_id}_pred_hand.mp4"],
                                 output_path=f"{self.vis_folder}/{seq_id}_pred_merged.mp4")
                
                    
                tarfile_path = os.path.join(self.results_folder, f"{seq_id}.tar")
     
                joblib.dump({'vertices': gen_motion_dict_k['vertices'],
                         'faces': gen_motion_dict_k['faces'],
                         'joints': gen_motion_dict_k['joints'],
                         'camera_transl': gen_camera_dict_k['camera_transl'], 
                         'camera_rot': gen_camera_dict_k['camera_rot'], 
                         'target_loc': target_opt_dict['joints'][_k_].cpu().numpy(),
                         'target_mask': target_mask_dict['joints'][_k_].cpu().numpy(),
                         'object_dict': object_dict}, 
                        tarfile_path)
   
    
                # subprocess.Popen(f'python src/render/render_single_fusion.py \
                #                  num={5} dataset_path={tarfile_path} \
                #                  mode=sequence savedir={self.vis_folder}_blender', shell=True)

                
                self.renderer.render_motion([gt_motion_dict_k], 
                                            color = [color1, color2, color3],
                                            target_dict=target_vis_dict,
                                            object_dict=object_dict,
                                            skeleton_dict=gt_skeleton_dict_k,
                                            camera_dict=gt_camera_dict_k,
                                            filename=f"{self.vis_folder}/{seq_id}_gt")

                if self.demo_mode:
                    merge_videos(
                        [f"{self.vis_folder}/{seq_id}_gt.mp4", f"{self.vis_folder}/{seq_id}_pred.mp4"],
                        output_path=f"{self.vis_folder}/{seq_id}_gt_pred_merged.mp4",
                        info_strs=['GT', 'Optimized'])
             
                all_motions.extend(gen_diffout_dict['full_motion_unnorm'])

        # [bs * num_dump_step, 1, 3, 120]
        all_motions = torch.cat(all_motions, axis=0).detach().cpu().numpy()  
  
        
        np.save(os.path.join(self.vis_folder, "results.npy"), {"motion": all_motions})
        print(f"[Done] Results are at [{os.path.abspath(self.vis_folder)}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--optimizer_file", type=str, required=True)
    opt = parser.parse_args()
 
    optimization_cfg = OmegaConf.load(opt.optimizer_file)
    model_cfg = OmegaConf.load(optimization_cfg.model_cfg)
    data_cfg = OmegaConf.load(optimization_cfg.data_cfg)
    diffusion_cfg = OmegaConf.load(optimization_cfg.diffusion_cfg)
    loss_cfg = OmegaConf.load(optimization_cfg.loss_cfg)

    if optimization_cfg.optimizer == 'ADAMW':
        from optim.dno_adam import DNO
    else:
        from dno_lbfgs import DNO


    seed_everything(optimization_cfg.seed)
    
    device = f"cuda:{optimization_cfg.device}" if torch.cuda.is_available() else "cpu"
    
    OptimizerWrapper()
   
# claude --resume e9b1773c-92ce-4778-8618-3c406567e59d