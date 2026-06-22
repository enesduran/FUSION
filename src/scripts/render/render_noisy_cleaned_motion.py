
import os
import sys
import torch
import shutil 
import joblib
import argparse
import subprocess
import numpy as np
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.data.amass_dataset import AmassDataset
from src.data.amass_wrapper import AmassWrapper
from src.render.mesh_viz import RendererWrapper
from src.utils.process_utils import BRANCH_NAME, CONTACT_INDICES, SMPLX_JOINTS

 
from src.utils.trainer_utils import cycle
from src.utils.genutils import seed_everything
from src.utils.viz_utils import pack_to_render
from src.model.base_diffusion import MotionDiffusion


color1 = (160 / 255, 160 / 255, 0 / 255, 1.0)
color2 = (0 / 255, 160 / 255, 160 / 255, 1.0)
obj_color = (1.0, 0.2, 0.2, 1.0)              

# omomo: Model has 10873295 trainable parameters and 154112 non-trainable parameters
# motionfix: Model has 19663055 trainable parameters and 0 non-trainable parameters
class Trainer(object):
    def __init__(self):
        
        super().__init__()
 

        data_cfg.device = device
        self.prep_dataloader()
        
        # idx for inputs
        idx_for_inputs = [data_cfg.load_feats.index(infeat) 
                        for infeat in self.data_wrapper.load_feats]
        
        # calculate total feature dimensions
        total_feats_dim = [self.data_wrapper.nfeats[i] for i in idx_for_inputs]
       
        model_cfg.nfeats = sum(total_feats_dim) 
 
        model_cfg.device = device
        model_cfg.dim_per_feat = total_feats_dim
        model_cfg.input_feats = self.data_wrapper.load_feats
        model_cfg.norm_type = self.data_wrapper.preproc.norm_type
        model_cfg.statistics_path = self.data_wrapper.preproc.stats_file
        model_cfg.diff_params = diffusion_cfg.diff_params
        model_cfg.smplx_path = data_cfg.smplx_path
        
        self.motion_diff_model = MotionDiffusion(**model_cfg, **loss_cfg)
        self.optimizer, self.scheduler = self.motion_diff_model.configure_optimizers()

        self.scaler = GradScaler(enabled=False)
  
        self.renderer = RendererWrapper(path2body_models=data_cfg.smplx_path)
        self.step = 0
         
         
        self.results_folder = os.path.join(trainer_cfg.save_dir, f'{BRANCH_NAME}/{trainer_cfg.exp_id}/weights') 
        self.vis_folder = os.path.join(trainer_cfg.save_dir, f'{BRANCH_NAME}/{trainer_cfg.exp_id}/vis_res')
        self.quant_folder = os.path.join(trainer_cfg.save_dir, f'{BRANCH_NAME}/{trainer_cfg.exp_id}/quant_res')
        self.generation_results_folder = os.path.join(trainer_cfg.save_dir, f'{BRANCH_NAME}/{trainer_cfg.exp_id}/generation_results')
        self.noise_denoise_steps = torch.tensor([[10], [30], [50], [100], [500]]).long().cuda()
       
     
        os.makedirs(self.results_folder, exist_ok=True)
        os.makedirs(self.vis_folder, exist_ok=True)
        os.makedirs(self.quant_folder, exist_ok=True)
        os.makedirs(self.generation_results_folder, exist_ok=True)
        
        # 0000 are for making it pop up at the start of the folder.
        shutil.copy(opt.trainer_file, os.path.join(self.vis_folder, "0000_trainer_cfg.yaml"))
        shutil.copy(trainer_cfg.loss_cfg, os.path.join(self.vis_folder, "0000_loss_cfg.yaml"))
        
 
  

    def prep_dataloader(self):
 
        data_cfg.load_splits = ['train', 'val'] if trainer_cfg.train_flag else ['val']       
        data_cfg.batch_size = trainer_cfg.batch_size   
 
        self.data_wrapper = AmassWrapper(**data_cfg)
        
        # for training we dont need to have it on whole dataset
        if trainer_cfg.train_flag:
            self.train_dl = cycle(self.data_wrapper.train_dataloader())
            self.val_dl = cycle(self.data_wrapper.val_dataloader()) 
            
        else:
            self.val_dl = self.data_wrapper.val_dataloader()
            
            
    def save(self, milestone):
        data = {'step': self.step,
                'model': self.motion_diff_model.denoiser.state_dict(),
                'ema': self.motion_diff_model.ema.state_dict(),
                'scaler': self.scaler.state_dict(),
                'optimizer': self.optimizer.state_dict()}

        milestone = f'{int(milestone):04d}' if milestone != 'best' else milestone
        torch.save(data, os.path.join(self.results_folder, f'model-{milestone}.pt'))

    def load(self, milestone, pretrained_path=None):

        if pretrained_path is None:
            milestone = f'{int(milestone):04d}' if milestone != 'best' else milestone
            pretrained_path = os.path.join(self.results_folder, f'model-{milestone}.pt')
          
        data = torch.load(pretrained_path)

        self.step = data['step']
        self.scaler.load_state_dict(data['scaler'])
        self.motion_diff_model.ema.load_state_dict(data['ema'], strict=False)
        self.motion_diff_model.denoiser.load_state_dict(data['model'], strict=False)
        print(f"Loaded weights from {pretrained_path}")
 
 
 
    def test(self):

        os.makedirs('fusion_runs/renders/motion_comparison', exist_ok=True)
        
        self.motion_diff_model.ema.ema_model.eval()
        self.motion_diff_model.eval()

        # visualize results
        for idx, val_data_dict in enumerate(self.val_dl):
            
            val_data_dict = {k: v.to(data_cfg.device) if torch.is_tensor(v) else v
                for k, v in val_data_dict.items()}
 
            x_start = self.motion_diff_model.norm_and_cat(val_data_dict, model_cfg.input_feats).detach()
 
            timesteps = self.motion_diff_model.sample_timesteps(samples=trainer_cfg.batch_size, sample_mode='uniform').long()
            timesteps = torch.tensor([100] * trainer_cfg.batch_size).long().to(data_cfg.device)

            noise = torch.randn_like(x_start)
            x_t = self.motion_diff_model.diffusion_process.q_sample(x_start, timesteps, noise)

            model_output = self.motion_diff_model.ema.ema_model.forward(x_t, timesteps)
            
            gt_motion_dict = self.motion_diff_model.diffout2motion(x_start)
            gen_motion_dict = self.motion_diff_model.diffout2motion(model_output)
            noisy_motion_dict = self.motion_diff_model.diffout2motion(x_t)

            gt_motion = gt_motion_dict['full_motion_unnorm']
            gen_motion = gen_motion_dict['full_motion_unnorm']
            
            if 'contact_masks' in model_cfg.input_feats:
                gt_contact_masks_ = gt_motion_dict['contact_masks']
                pred_contact_masks_ = gen_motion_dict['contact_masks']
                
                pred_contact_masks_[pred_contact_masks_ > 0.8] = 1
                pred_contact_masks_[pred_contact_masks_ < 0] = 0
                
    
            gt_dict, gt_camera_dict = self.motion_diff_model.rot2xyz(gt_motion_dict, 
                                                        body_vtemp=val_data_dict["datum"]['body_vtemp'], 
                                                        betas=val_data_dict["datum"]['betas'], 
                                                        gender_list=val_data_dict["datum"]['gender'],
                                                        cpu_flag=True)
                
            pred_dict, pred_camera_dict = self.motion_diff_model.rot2xyz(gen_motion_dict, 
                                                    body_vtemp=val_data_dict["datum"]['body_vtemp'], 
                                                    betas=val_data_dict["datum"]['betas'],
                                                    gender_list=val_data_dict["datum"]['gender'],
                                                    cpu_flag=True)
            
            noisy_dict, noisy_camera_dict = self.motion_diff_model.rot2xyz(noisy_motion_dict, 
                                                    body_vtemp=val_data_dict["datum"]['body_vtemp'], 
                                                    betas=val_data_dict["datum"]['betas'],
                                                    gender_list=val_data_dict["datum"]['gender'],
                                                    cpu_flag=True)



            for i in range(trainer_cfg.batch_size):
               
                object_dict = {}

                
                pred_contact_masks = pred_contact_masks_[i].detach().cpu().numpy()
                gt_contact_masks = gt_contact_masks_[i].detach().cpu().numpy()
                 
                # pred_skeleton_dict = {'positions': pred_dict['joints'][i, :, :len(SMPLX_JOINTS)],
                #                     'contact_masks': pred_contact_masks}            
                # gt_skeleton_dict = {'positions': gt_dict['joints'][i, :, :len(SMPLX_JOINTS)],
                #                     'contact_masks': gt_contact_masks}
                
                
                smpl_params_denoised = {"vertices": pred_dict['vertices'][i], 
                                        "faces": pred_dict['faces'],
                                        "joints": pred_dict['joints'][i]}
                smpl_params_gt = {"vertices": gt_dict['vertices'][i], 
                                        "faces": gt_dict['faces']}
                smpl_params_noisy = {"vertices": noisy_dict['vertices'][i], 
                                        "faces": noisy_dict['faces']}
                
                pred_camera_dict_i = {'camera_rot': pred_camera_dict['camera_rot'][i],
                               "camera_transl": pred_camera_dict['camera_transl'][i],
                               "coef": 2.0}
                gt_camera_dict_i = {'camera_rot': gt_camera_dict['camera_rot'][i],
                               "camera_transl": gt_camera_dict['camera_transl'][i],
                               "coef": 2.0}
                noisy_camera_dict_i = {'camera_rot': noisy_camera_dict['camera_rot'][i],
                               "camera_transl": noisy_camera_dict['camera_transl'][i],
                               "coef": 2.0}
                
                tarfile_denoised = f"fusion_runs/renders/motion_comparison/{str(val_data_dict['id'][i].item()).zfill(6)}_pred.tar"
                tarfile_gt = f"fusion_runs/renders/motion_comparison/{str(val_data_dict['id'][i].item()).zfill(6)}_gt.tar"
                tarfile_noisy = f"fusion_runs/renders/motion_comparison/{str(val_data_dict['id'][i].item()).zfill(6)}_noisy.tar"
                
                # tarfiles 
                joblib.dump({'camera_rot': pred_camera_dict_i['camera_rot'],
                            'camera_transl': pred_camera_dict_i['camera_transl'],
                            'vertices': smpl_params_denoised['vertices'],
                            'faces': smpl_params_denoised['faces'], 
                            'joints': smpl_params_denoised['joints']},
                            tarfile_denoised)
                joblib.dump({'camera_rot': gt_camera_dict_i['camera_rot'],
                            'camera_transl': gt_camera_dict_i['camera_transl'],
                            'vertices': smpl_params_gt['vertices'],
                            'faces': smpl_params_gt['faces'], 
                            # 'joints': smpl_params_gt['joints'],
                            },
                            tarfile_gt)
                joblib.dump({'camera_rot': noisy_camera_dict_i['camera_rot'],
                            'camera_transl': noisy_camera_dict_i['camera_transl'],
                            'vertices': smpl_params_noisy['vertices'],
                            'faces': smpl_params_noisy['faces'], 
                            # 'joints': smpl_params_noisy['joints']
                            },
                            tarfile_noisy)
                
                num_poses = 5
                floor_flag = 'false'
                 
                image_cmd = f'python src/render/render_single_fusion.py dataset_path={tarfile_denoised} mode=sequence \
                    num={num_poses} floor_flag={floor_flag} long_cam_flag=true savedir=fusion_runs/renders/motion_comparison/blender_denoised_{str(val_data_dict["id"][i].item()).zfill(6)}'
                subprocess.call(image_cmd, shell=True)
                
                image_cmd = f'python src/render/render_single_fusion.py dataset_path={tarfile_gt} mode=sequence \
                    num={num_poses} floor_flag={floor_flag} long_cam_flag=true savedir=fusion_runs/renders/motion_comparison/blender_gt_{str(val_data_dict["id"][i].item()).zfill(6)}'
                subprocess.call(image_cmd, shell=True)
                
                image_cmd = f'python src/render/render_single_fusion.py dataset_path={tarfile_noisy} mode=sequence \
                    num={num_poses} floor_flag={floor_flag} long_cam_flag=true savedir=fusion_runs/renders/motion_comparison/blender_noisy_{str(val_data_dict["id"][i].item()).zfill(6)}'
                subprocess.call(image_cmd, shell=True)
 

                self.renderer.render_motion([smpl_params_denoised],
                                            object_dict={}, 
                                            color = [color1, color2],
                                            skeleton_dict={},
                                            camera_dict=pred_camera_dict_i,
                                            filename=f"fusion_runs/renders/motion_comparison/{str(val_data_dict['id'][i].item()).zfill(6)}_denoised")
                self.renderer.render_motion([smpl_params_gt],
                                            object_dict={}, 
                                            color = [color1, color2],
                                            skeleton_dict={},
                                            camera_dict=pred_camera_dict_i,
                                            filename=f"fusion_runs/renders/motion_comparison/{str(val_data_dict['id'][i].item()).zfill(6)}_gt")
                self.renderer.render_motion([smpl_params_noisy],
                                            object_dict={}, 
                                            color = [color1, color2],
                                            skeleton_dict={},
                                            camera_dict=pred_camera_dict_i,
                                            filename=f"fusion_runs/renders/motion_comparison/{str(val_data_dict['id'][i].item()).zfill(6)}_noisy")
 
                    
         

        print('testing complete')

def main():
    trainer = Trainer()
    trainer.load(trainer_cfg.milestone)
    trainer.test()
    torch.cuda.empty_cache()
   

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainer-file', default='configs/trainer1_setting0.yaml') 
    return parser.parse_args()


if __name__ == "__main__":
    
    opt = parse_opt()
 
    trainer_cfg = OmegaConf.load(opt.trainer_file)

     
    model_cfg = OmegaConf.load(trainer_cfg.model_cfg)
    data_cfg = OmegaConf.load(trainer_cfg.data_cfg)
    diffusion_cfg = OmegaConf.load(trainer_cfg.diffusion_cfg)
    loss_cfg = OmegaConf.load(trainer_cfg.loss_cfg)
    
    seed_everything(trainer_cfg.seed)
    trainer_cfg.train_flag = False

    trainer_cfg.milestone = "best"
    device = f"cuda:{trainer_cfg.device}" if torch.cuda.is_available() else "cpu"

    main()
 
