import os 
import torch
import logging
import numpy as np
from typing import List
from torch import Tensor
from ema_pytorch import EMA
from einops import rearrange
from omegaconf import DictConfig
import torch.distributions as dist
from torch.nn import functional as F
from torch.nn.functional import l1_loss, mse_loss, smooth_l1_loss


from src.model.rotation2xyz import Rotation2xyz
from src.model.tmed_denoiser import TMED_denoiser
from src.model.omomo_denoiser import TransformerDiffusionModel


from src.diffusion import create_diffusion
from src.utils.genutils import cast_dict_to_tensors
from src.model.utils.lr_scheduler import CosineAnnealingLRWarmup
from src.utils.transforms3d import apply_rot_delta, get_z_rot, change_for
from src.utils.transforms3d import change_for, transform_body_pose, get_z_rot
 
from src.utils.optim_utils import GLOBAL_RHAND_INDICES, GLOBAL_LHAND_INDICES

log = logging.getLogger(__name__)


# from pytorch_lightning import LightningModule 
class MotionDiffusion:
    def __init__(self, 
                 denoiser_cfg: DictConfig,
                 losses: DictConfig,
                 diff_params: DictConfig,
                 hparams: DictConfig,
                 nfeats: int,
                 input_feats: List[str],
                 statistics_path: str,
                 dim_per_feat: List[int],
                 norm_type: str,
                 smplx_path: str,
                 loss_func_pos: str = 'mse', # l1 mse
                 loss_func_feats: str = 'mse', # l1 mse
                 device: str ='cuda:0',
                 ema_decay: float = 0.995,
                 ema_update_every: int = 1,
                 timestep_respacing = None,
                 **kwargs):
 
        
        self.nfeats = nfeats
        self.device = device
        self.norm_type = norm_type
        self.hparams = hparams
        self.input_feats = list(input_feats)   
        self.input_feats_dims = list(dim_per_feat)
        self.input_feats_dims_cumsum = np.cumsum([0] + self.input_feats_dims)

        # zero std indices [6,  11,  14, 339, 340, 341]
        ''' body_transl_delta_pelv 3, body_orient_xy 6,  z_orient_delta 6,
            body_pose 324, body_joints_local_wo_z_rot 165 contact_masks 4'''

        self.stats = self.load_norm_statistics(statistics_path, self.device)

 
        if set(["body_transl_delta_pelv_xy", "body_orient_delta",
                "body_pose_delta"]).issubset(self.input_feats):
            self.using_deltas = True
        else:
            self.using_deltas = False

        transl_feats = [x for x in self.input_feats if 'transl' in x]
        if set(transl_feats).issubset(["body_transl_delta", "body_transl_delta_pelv",
                                  "body_transl_delta_pelv_xy"]):
            self.using_deltas_transl = True
        else:
            self.using_deltas_transl = False
  
        self.input_feats = input_feats
        self.first_trans = None
        

        # If we want to overide it at testing time
        self.diff_params = diff_params
  
        # ARGUMENTS FOR DIFFUSION
        # timestep_respacing --> None just the default linear things
        # noise_schedule="linear", squaredcos_cap_v2
        # use_kl=False,
        # sigma_small=False,
        # predict_xstart=False,
        # learn_sigma=True,
        # rescale_learned_sigmas=False,
        self.diffusion_process = create_diffusion(timestep_respacing=timestep_respacing,
                                     learn_sigma=False,
                                     sigma_small=True,
                                     diffusion_steps=self.diff_params.num_train_timesteps,
                                     noise_schedule=self.diff_params.noise_schedule,
                                     predict_xstart=False if self.diff_params.predict_type == 'noise' else True,
                                     **kwargs) # noise vs sample
        
        self.rot2xyz = Rotation2xyz(device=device, smplx_path=smplx_path)
        
        if denoiser_cfg.name == 'omomo_architecture':
            denoiser_cfg.d_feats = nfeats
            denoiser_cfg.max_timesteps = self.diff_params.num_train_timesteps
        
        elif denoiser_cfg.name == 'motionfix_architecture':
            denoiser_cfg.d_feats = nfeats    
        else:
            raise NotImplementedError
        
        denoiser_class = TransformerDiffusionModel if denoiser_cfg.name == 'omomo_architecture' else TMED_denoiser
        
        # dynamically input network arguments 
        self.denoiser = denoiser_class(**denoiser_cfg).to(self.device)
        
        
        self.ema = EMA(self.denoiser, beta=ema_decay, update_every=ema_update_every)


        self.tsteps_distr = dist.Gamma(torch.tensor(2.0),
                                       torch.tensor(1.0))
        self.loss_params = losses
 
        if loss_func_feats == 'l1':
            self.loss_func_feats = l1_loss
        elif loss_func_feats in ['mse', 'l2']:
            self.loss_func_feats = mse_loss
        elif loss_func_feats in ['sl1']:
            self.loss_func_feats = smooth_l1_loss
 

        self.__post_init__()

    def train(self):
        self.denoiser.train()
    
    def eval(self):
        
        self.denoiser.eval()


    def load_norm_statistics(self, path, device):
        assert os.path.exists(path)
        stats = np.load(path, allow_pickle=True)[()]    
        return cast_dict_to_tensors(stats, device=device)

    def configure_optimizers(self):
 
        
        optimizer = torch.optim.AdamW(lr=self.hparams.optim.learning_rate,
                                      params=self.denoiser.parameters())
   
        if self.hparams.lr_scheduler is not None:

            return optimizer, CosineAnnealingLRWarmup(optimizer,
                                                T_max=self.hparams.optim.t_total,
                                                T_warmup=self.hparams.optim.t_warmup,
                                                lr_final=self.hparams.optim.lr_final,
                                                lr_initial=self.hparams.optim.learning_rate)
        else:
            return optimizer, None

    def cat_inputs(self, x_list: List[Tensor]):
        """
        cat the inputs to a unified vector and return their lengths in order
        to un-cat them later
        """
        return torch.cat(x_list, dim=-1), [x.shape[-1] for x in x_list]

    def __post_init__(self):

        trainable, nontrainable = 0, 0
        
        for p in self.denoiser.parameters():
            if p.requires_grad:
                trainable += np.prod(p.size())
            else:
                nontrainable += np.prod(p.size())

        self.hparams.n_params_trainable = int(trainable)
        self.hparams.n_params_nontrainable = int(nontrainable)
        print(f"Model has {self.hparams.n_params_trainable} trainable parameters and {self.hparams.n_params_nontrainable} non-trainable parameters")
        
 
    def forward(self):
        # Return callable that goes through self.denoiser.__call__
        # so that DDP's gradient synchronization hooks are triggered.
        return lambda *args, **kwargs: self.denoiser(*args, **kwargs)
 
    def _diffusion_reverse(self, 
                           inp_motion_mask_shape, 
                           init_vec=None, 
                           show_progress=True,
                           use_ddpm=False):
          
        if init_vec is None:
            initial_latents = torch.randn((inp_motion_mask_shape[0], inp_motion_mask_shape[1], self.nfeats), 
                device=self.device, dtype=torch.float)
        else:
            initial_latents = init_vec
 

        if use_ddpm:
            # Sample 
            final_diffout = self.diffusion_process.p_sample_loop(
                                                self.denoiser.forward,
                                                initial_latents.shape, 
                                                initial_latents, 
                                                clip_denoised=False, 
                                                model_kwargs=dict(),
                                                progress=show_progress,
                                                device=initial_latents.device)
            
        else:
            # Sample 
            final_diffout = self.diffusion_process.ddim_sample_loop(
                                                self.denoiser.forward,
                                                initial_latents.shape, 
                                                initial_latents, 
                                                clip_denoised=False, 
                                                model_kwargs=dict(),
                                                progress=show_progress,
                                                device=initial_latents.device)
        
     
        return initial_latents, final_diffout
         
    
    def sample_timesteps(self, samples: int, sample_mode=None):
        if sample_mode is None:
            if self.trainer.current_epoch / self.trainer.max_epochs > 0.5:

                gamma_samples = self.tsteps_distr.sample((samples,))
                lower_bound = 0
                upper_bound = self.diffusion_process.num_timesteps
                scaled_samples = upper_bound * (gamma_samples / gamma_samples.max()) 
                # Convert the samples to integers
                timesteps_sampled = scaled_samples.floor().int().to(self.device)
            else:
                timesteps_sampled = torch.randint(0,
                                    self.diffusion_process.num_timesteps,
                                     (samples, ),
                                    device=self.device)
        else:
            
            if sample_mode == 'uniform':
                timesteps_sampled = torch.randint(0,
                                        self.diffusion_process.num_timesteps,
                                        (samples, ),
                                        device=self.device)
        return timesteps_sampled

    def train_diffusion_forward(self, batch):
    
        # diffusion process return with noise and noise_pred
        input_motion_feats = batch['motion']

        bsz = input_motion_feats.shape[0]
        # Sample a random timestep for each motion
        timesteps = self.sample_timesteps(samples=bsz, sample_mode='uniform').long()
            
        model_kwargs = {'betas': batch['datum']['betas'], 
                        'body_vtemp': batch['datum']['body_vtemp'],
                        'pos_offset': batch['datum']['pos_offset'],
                        'root_offset': batch['datum']['root_offset'],
                        'gender': batch['datum']['gender'],
                        'joint_positions': batch['datum']['joint_positions']}
         
        return self.diffusion_process.training_losses(self,
                                                        input_motion_feats,
                                                        timesteps,
                                                        model_kwargs=model_kwargs)
    

    def unnorm_inputs(self, x_list: List[Tensor], names: List[str]):
        """
        Un-normalise inputs using the self.stats metrics
        """
        x_unnorm = []
        for x, name in zip(x_list, names):
            x_unnorm.append(self.unnorm(x, self.stats[name]))
        return x_unnorm
    
    def unnorm(self, x, stats):
        if self.norm_type == "standardize":
            mean = stats['mean'].to(self.device)
            std = stats['std'].to(self.device)
            return x * 2 * (std + 1e-5) + mean
        elif self.norm_type == "min_max":
            max = stats['max'].to(self.device)
            min = stats['min'].to(self.device)
            return x * (max - min + 1e-5) + min
    
    def norm(self, x, stats):

        if self.norm_type == "standardize":
            mean = stats['mean'].to(self.device)
            std = stats['std'].to(self.device)
            return (x - mean) / (2*(std + 1e-5))
        elif self.norm_type == "min_max":
            max = stats['max'].to(self.device)
            min = stats['min'].to(self.device)
            assert ((x - min) / (max - min + 1e-5)).min() >= 0
            assert ((x - min) / (max - min + 1e-5)).max() <= 1
            return (x - min) / (max - min + 1e-5)
        else:
            raise NotImplementedError


    def norm_and_cat(self, batch, features_types):
        """
        turn batch data into the format the forward() function expects
        """
        seq_first = lambda t: rearrange(t, 'b s ... -> s b ...')

        ## PREPARE INPUT ##
        list_of_feat_tensors = [seq_first(batch[f'{feat_type}'])
                                for feat_type in features_types if f'{feat_type}' in batch.keys()]

        # normalise and cat to a unified feature vector
        list_of_feat_tensors_normed = self.norm_inputs(list_of_feat_tensors,
                                                        features_types)


        x_norm, _ = self.cat_inputs(list_of_feat_tensors_normed)

        # Store per-sample canonicalized frame-0 translation
        if 'datum' in batch and 'trans' in batch['datum']:
            self.first_trans = batch['datum']['trans'][:, 0:1].to(self.device)

        return x_norm.permute(1, 0, 2)
    
    def norm_inputs(self, x_list: List[Tensor], names: List[str]):
        """
        Normalise inputs using the self.stats metrics
        """
        x_norm = []
 
        for x, name in zip(x_list, names):
            x_norm.append(self.norm(x, self.stats[name]))
        return x_norm
    

    def uncat_inputs(self, x: Tensor, lengths: List[int]):
        """
        split the unified feature vector back to its original parts
        """
        return torch.split(x, lengths, dim=-1)


    def training_step(self, batch, batch_idx):
        return self.allsplit_step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.allsplit_step("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.allsplit_step("test", batch, batch_idx)

    def compute_losses(self, out_dict, dataset_names):
 
        f_rg = np.cumsum([0] + self.input_feats_dims)

        all_losses_dict = {}
        motionfix_loss = torch.tensor(0.0, device=self.device)
        full_feature_loss = self.loss_func_feats(out_dict['target'],
                                            out_dict['model_output'],
                                            reduction='none')
    

        unique_datasets = list(set(dataset_names))

        # Per-sample masks over the batch axis, used to compute a true
        # per-sample mean loss for each dataset present in the batch.
        dataset_masks = {
            name: torch.tensor(
                [n == name for n in dataset_names],
                device=full_feature_loss.device, dtype=torch.bool,
            )
            for name in unique_datasets
        }
        dataset_losses = {
            name: torch.zeros((), device=full_feature_loss.device)
            for name in unique_datasets
        }
        # Sample counts per dataset in this batch -- needed downstream so
        # validation can weight each batch's per-dataset mean correctly.
        dataset_counts = {name: int(mask.sum().item()) for name, mask in dataset_masks.items()}

        # Main loss calculation loop
        for i, _ in enumerate(f_rg[:-1]):
            if 'delta' in self.input_feats[i]:
                # first element is zero for deltas, no need to include it.
                cur_feat_loss = full_feature_loss[:, 1:, f_rg[i]:f_rg[i+1]].mean(-1)
            else:
                cur_feat_loss = full_feature_loss[..., f_rg[i]:f_rg[i+1]].mean(-1)

            tot_feat_loss = cur_feat_loss.mean()

            # Update all_losses_dict with overall tot_feat_loss
            all_losses_dict.update({self.input_feats[i]: tot_feat_loss})
            motionfix_loss += tot_feat_loss  # Overall loss across datasets

            # Per-dataset: average only the rows belonging to this dataset.
            for name, mask in dataset_masks.items():
                dataset_losses[name] = dataset_losses[name] + cur_feat_loss[mask].mean()

        # Average across input features so the value is comparable to total_loss.
        for name in dataset_losses:
            dataset_losses[name] = dataset_losses[name].detach() / len(self.input_feats)

        motionfix_loss /= len(self.input_feats)
        
        all_losses_dict['total_loss'] = motionfix_loss
        all_losses_dict = all_losses_dict | dataset_losses
        
        
        # total_loss = out_dict['loss'].mean()
        # all_losses_dict['mse'] = out_dict['mse'].mean()  
        # all_losses_dict['vel_mse'] = out_dict['vel_mse'].mean()  
        all_losses_dict['rcxyz_mse'] = out_dict['rcxyz_mse'].mean()  
        # all_losses_dict['vel_xyz_mse'] = out_dict['vel_xyz_mse'].mean()  
        all_losses_dict['foot_skating'] = out_dict['foot_skating'].mean()  
        # all_losses_dict['contact_prediction'] = out_dict['contact_prediction'].mean()  

        motionfix_loss += out_dict['foot_skating'].mean()
        motionfix_loss += out_dict['rcxyz_mse'].mean()

        return motionfix_loss, all_losses_dict, dataset_counts
        
        # return total_loss, all_losses_dict 
        
        


    def check_nans(self):

        nan_cond = False
        total_norm = torch.tensor(0.0, device=self.device)
    
        for p in self.denoiser.parameters():
            if p.requires_grad:
                
                nan_cond = nan_cond or torch.isnan(p).any().item()
                total_norm += torch.norm(torch.norm(p.grad.detach(), 2.0))
                                  
        return nan_cond or torch.isnan(total_norm).item()


    def generate_motion(self,
                        motions_cond,
                        init_vec_method='noise', 
                        init_vec=None, 
                        show_progress=True,
                        use_ddpm=False):
        
        if init_vec_method == 'noise_prev':
            init_diff_rev = init_vec
        elif init_vec_method == 'source':
            init_diff_rev = motions_cond.permute(1, 0, 2)
        else:
            init_diff_rev = None
        

        with torch.no_grad():
            return self._diffusion_reverse(motions_cond.shape,
                                           init_vec=init_diff_rev,                       
                                           show_progress=show_progress,
                                           use_ddpm=use_ddpm)


          
    def diffout2motion(self, diffout):
        '''diffout is of shape B, T, D ''' 
         
        unnorm_dict = {}

        # it means only pose 
        if diffout.shape[1] == 1:
 
            rots_unnorm = self.cat_inputs(self.unnorm_inputs(
                                           self.uncat_inputs(diffout,
                                           self.input_feats_dims),
                                           self.input_feats))[0]
            unnorm_dict['full_motion_unnorm'] = rots_unnorm
        else:
            # - "body_transl_delta_pelv_xy_wo_z"
            # - "body_transl_z"
            # - "z_orient_delta"
            # - "body_orient_xy"
            # - "body_pose"
            # - "body_joints_local_wo_z_rot"
            feats_unnorm = self.cat_inputs(self.unnorm_inputs(
                                            self.uncat_inputs(diffout,
                                            self.input_feats_dims),
                                            self.input_feats))[0]
            
            
            if "contact_masks" in self.input_feats:
            
                idx = self.input_feats.index("contact_masks")                
                cum_idx1 = self.input_feats_dims_cumsum[idx]
                cum_idx2 = self.input_feats_dims_cumsum[idx+1]
                
                # prediction, pass through sigmoid 
                if torch.unique(feats_unnorm[..., cum_idx1:cum_idx2]).shape[0] > 2:
                    unnorm_contact_masks = feats_unnorm[..., cum_idx1:cum_idx2]
                    # unnorm_contact_masks = torch.sigmoid(feats_unnorm[..., cum_idx1:cum_idx2])
                else:
                    unnorm_contact_masks = feats_unnorm[..., cum_idx1:cum_idx2]
                
                unnorm_dict["contact_masks"] = unnorm_contact_masks 
                
          
            # joints are just for overparameterization, we don't need them in motion. Exclude.  
            if "body_joints_local_wo_z_rot" in self.input_feats:
                idx = self.input_feats.index("body_joints_local_wo_z_rot")     
                cum_idx1 = self.input_feats_dims_cumsum[idx]
                cum_idx2 = self.input_feats_dims_cumsum[idx+1]
                
                unnorm_dict["body_joints_local_wo_z_rot"] = feats_unnorm[..., cum_idx1:cum_idx2]
                
        
            first_trans = self.first_trans

            # get global translation depending on the input features.
            if 'z_orient_delta' in self.input_feats:

                first_orient_z = torch.eye(3, device=self.device).unsqueeze(0)  # Now the shape is (1, 1, 3, 3)
                first_orient_z = first_orient_z.repeat(feats_unnorm.shape[0], 1, 1)  # Now the shape is (B, 1, 3, 3)
                first_orient_z = transform_body_pose(first_orient_z, 'rot->6d')

                # find idx for z_orient_delta
                idx = self.input_feats.index("z_orient_delta")
                cum_idx1 = self.input_feats_dims_cumsum[idx]
                cum_idx2 = self.input_feats_dims_cumsum[idx+1]

                # --> first_orient_z convert to 6d
                # integrate z orient delta --> z component tof orientation
                z_orient_delta = feats_unnorm[..., cum_idx1:cum_idx2]

                prev_z = first_orient_z
                full_z_angle = [first_orient_z[:, None]]

                for i in range(1, z_orient_delta.shape[1]):
                    curr_z = apply_rot_delta(prev_z, z_orient_delta[:, i])
                    prev_z = curr_z.clone()
                    full_z_angle.append(curr_z[:,None])

                full_z_angle = torch.cat(full_z_angle, dim=1)

                # find the change of angle around z axis
                full_z_angle_rotmat = get_z_rot(full_z_angle)

                # find idx for z_orient
                idx = self.input_feats.index("body_orient_xy")
                cum_idx1 = self.input_feats_dims_cumsum[idx]
                cum_idx2 = self.input_feats_dims_cumsum[idx+1]


                xy_orient = feats_unnorm[..., cum_idx1:cum_idx2]
                xy_orient_rotmat = transform_body_pose(xy_orient, '6d->rot')


                full_global_orient_rotmat = full_z_angle_rotmat @ xy_orient_rotmat
                full_global_orient = transform_body_pose(full_global_orient_rotmat,
                                                         'rot->6d')

                unnorm_dict['unnorm_first_trans'] = first_trans
                
                # apply deltas
                assert 'body_transl_delta_pelv' in self.input_feats
                
                # find idx for body_transl_delta_pelv
                idx = self.input_feats.index("body_transl_delta_pelv")                
                cum_idx1 = self.input_feats_dims_cumsum[idx]
                cum_idx2 = self.input_feats_dims_cumsum[idx+1]


                pelvis_delta = feats_unnorm[..., cum_idx1:cum_idx2]

                trans_vel_pelv = change_for(pelvis_delta[:, 1:],
                                            full_global_orient_rotmat[:, :-1],
                                            forward=False)

                # new_state_pos = prev_trans_norm.squeeze() + trans_vel_pelv
                full_trans = torch.cumsum(trans_vel_pelv, dim=1) + first_trans
                full_trans = torch.cat([first_trans, full_trans], dim=1)

                # find idx for z_orient
                idx = self.input_feats.index("body_pose")                
                cum_idx1 = self.input_feats_dims_cumsum[idx]
                cum_idx2 = self.input_feats_dims_cumsum[idx+1]
              
                full_rots = torch.cat([full_global_orient, feats_unnorm[..., cum_idx1:cum_idx2]], dim=-1)
                unnorm_dict['full_motion_unnorm'] = torch.cat([full_trans, full_rots], dim=-1)

            elif "body_orient_delta" in self.input_feats:
                delta_trans = diffout[..., 6:9]
                pelv_orient = diffout[..., 9:15]

                # for i in range(1, delta_trans.shape[1]):
                full_trans_unnorm = self.integrate_translation(pelv_orient[:, :-1],
                                                            first_trans,
                                                            delta_trans[:, 1:])
                rots_unnorm = self.cat_inputs(self.unnorm_inputs(self.uncat_inputs(
                                                                diffout[..., 9:],
                                                        self.input_feats_dims[2:]),
                                                self.input_feats[2:])
                                                )[0]
                unnorm_dict['full_motion_unnorm'] = torch.cat([full_trans_unnorm,
                                                                rots_unnorm], dim=-1)
                    
            else:
      
                idx = self.input_feats.index("body_transl_delta")
                cum_idx1 = self.input_feats_dims_cumsum[idx]
                cum_idx2 = self.input_feats_dims_cumsum[idx+1]
                
                delta_trans = diffout[..., cum_idx1:cum_idx2]
                
                
                
                
                pelv_orient = diffout[..., 3:9]
                # for i in range(1, delta_trans.shape[1]):
                full_trans_unnorm = self.integrate_translation(pelv_orient[:, :-1],
                                                            first_trans,
                                                            delta_trans[:, 1:])
                rots_unnorm = self.cat_inputs(self.unnorm_inputs(self.uncat_inputs(
                                                                diffout[..., 3:],
                                                        self.input_feats_dims[1:]),
                                                self.input_feats[1:])
                                                )[0]
                unnorm_dict['full_motion_unnorm'] = torch.cat([full_trans_unnorm,
                                                        rots_unnorm], dim=-1)
        return unnorm_dict
    

    def motion2hand(self, unnorm_dict):

        smplx_motion_unnorm = unnorm_dict['full_motion_unnorm'][..., 3:]
 

        l_ind = torch.stack([torch.arange(i, i+6) for i in 6*(GLOBAL_LHAND_INDICES[1:])]).flatten()
        r_ind = torch.stack([torch.arange(i, i+6) for i in 6*(GLOBAL_RHAND_INDICES[1:])]).flatten()

        hand_motion_dict = {'right': smplx_motion_unnorm[..., r_ind],
                            'left': smplx_motion_unnorm[..., l_ind]}


        return hand_motion_dict


    def allsplit_step(self, split: str, batch: dict, generate_motions: bool = False):
         
        batch['motion'] = self.norm_and_cat(batch, self.input_feats)
        
        self.batch_size = len(batch['length'])
        dif_dict = self.train_diffusion_forward(batch)
        
        # rs_set Bx(S+1)xN --> first pose included
        total_loss, loss_dict, dataset_counts = self.compute_losses(dif_dict, batch['dataset_name'])

        loss_dict = {f'total_losses/{split}/{k}' if k not in self.input_feats
            else f'feature_losses/{split}/{k}': v.item() for k, v in loss_dict.items()}

        # Per-key sample counts -- only populated for the per-dataset entries.
        # Validation uses these to compute a true per-sample mean across batches.
        loss_counts = {
            f'total_losses/{split}/{name}': cnt for name, cnt in dataset_counts.items()
        }

        loss_dict_to_log = {'loss_dict': loss_dict, 'total_loss': total_loss, 'loss_counts': loss_counts}
         
        if split == 'val' and generate_motions:
            _, diffout = self.generate_motion(batch['motion'],
                                            init_vec_method='noise',
                                            show_progress=False,
                                            use_ddpm=False)
            
            loss_dict_to_log['predicted_motions'] = self.diffout2motion(diffout)['full_motion_unnorm']

        return loss_dict_to_log
