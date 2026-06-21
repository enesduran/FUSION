import os
import wandb
import torch
import shutil 
import argparse
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from src.utils.trainer_utils import cycle
from src.utils.genutils import seed_everything
from src.data.amass_dataset import AmassDataset
from src.data.amass_wrapper import AmassWrapper
from src.render.mesh_viz import RendererWrapper
from src.model.base_diffusion import MotionDiffusion
from src.utils.transforms3d import transform_body_pose, matrix_to_axis_angle, get_z_rot
from src.utils.process_utils import BRANCH_NAME, CONTACT_INDICES, SMPLX_JOINTS 
from src.utils.viz_utils import pack_to_render, color1, color2, color3, obj_color



def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


# of trainable parameters
# omomo model: 10873295 motionfix model: 19663055
class Trainer(object):
    def __init__(self):
        
        super().__init__()

        self.use_ddp = dist.is_initialized()
        self.rank = dist.get_rank() if self.use_ddp else 0
        self.world_size = dist.get_world_size() if self.use_ddp else 1
 
        # dont use it in test time as well 
        if trainer_cfg.wandb.use_wandb and trainer_cfg.train_flag and is_main_process():

            wandb.init(config=opt,
                       project=trainer_cfg.wandb.wandb_pj_name,
                       entity=trainer_cfg.wandb.entity,
                       name=trainer_cfg.wandb.exp_name,
                       dir=os.path.join(trainer_cfg.save_dir, BRANCH_NAME))

            wandb.save(opt.trainer_file, policy='now')
            wandb.save(trainer_cfg.loss_cfg, policy='now')

 
        data_cfg.device = device

        if trainer_cfg.train_flag:
            self.prep_dataloader()
            idx_for_inputs = [data_cfg.load_feats.index(infeat)
                            for infeat in self.data_wrapper.load_feats]
            total_feats_dim = [self.data_wrapper.nfeats[i] for i in idx_for_inputs]
            model_cfg.input_feats = self.data_wrapper.load_feats
        else:
            # Derive feature dims from stats file without loading any dataset
            stats = np.load(data_cfg.preproc.stats_file, allow_pickle=True)[()]
            total_feats_dim = [stats[feat]['mean'].shape[-1] for feat in data_cfg.load_feats
                               if feat in stats]
            model_cfg.input_feats = list(data_cfg.load_feats)

        model_cfg.nfeats = sum(total_feats_dim)
        model_cfg.device = device
        model_cfg.dim_per_feat = total_feats_dim
        model_cfg.norm_type = data_cfg.preproc.norm_type
        model_cfg.statistics_path = data_cfg.preproc.stats_file
        model_cfg.diff_params = diffusion_cfg.diff_params
        model_cfg.smplx_path = data_cfg.smplx_path
        
        self.motion_diff_model = MotionDiffusion(**model_cfg, **loss_cfg)

        # Wrap the denoiser in DDP for multi-GPU training
        if self.use_ddp:
            self.motion_diff_model.denoiser = DDP(
                self.motion_diff_model.denoiser, 
                device_ids=[self.rank],
                output_device=self.rank,
                find_unused_parameters=False
            )

        self.optimizer, self.scheduler = self.motion_diff_model.configure_optimizers()

        self.scaler = GradScaler(enabled=False)
        self.gradient_accumulate_every = getattr(trainer_cfg, 'gradient_accumulate_every', 1)
  
        self.renderer = RendererWrapper(path2body_models=data_cfg.smplx_path)
        self.step = 0
         
         
        self.results_folder = os.path.join(trainer_cfg.save_dir, f'{BRANCH_NAME}/{trainer_cfg.exp_id}/weights') 
        self.vis_folder = os.path.join(trainer_cfg.save_dir, f'{BRANCH_NAME}/{trainer_cfg.exp_id}/vis_res')
        self.quant_folder = os.path.join(trainer_cfg.save_dir, f'{BRANCH_NAME}/{trainer_cfg.exp_id}/quant_res')
        self.generation_results_folder = os.path.join(trainer_cfg.save_dir, f'{BRANCH_NAME}/{trainer_cfg.exp_id}/generation_results')
       
        os.makedirs(self.results_folder, exist_ok=True)
        os.makedirs(self.vis_folder, exist_ok=True)
        os.makedirs(self.quant_folder, exist_ok=True)
        os.makedirs(self.generation_results_folder, exist_ok=True)
        
        # 0000 are for making it pop up at the start of the folder.
        shutil.copy(opt.trainer_file, os.path.join(self.vis_folder, "0000_trainer_cfg.yaml"))
        shutil.copy(trainer_cfg.loss_cfg, os.path.join(self.vis_folder, "0000_loss_cfg.yaml"))
        
  

    def prep_dataloader(self):
        data_cfg.load_splits = ['train', 'val', 'test'] if trainer_cfg.train_flag else ['test']    
        
        data_cfg.train_batch_size = trainer_cfg.train_batch_size
        data_cfg.val_batch_size = trainer_cfg.val_batch_size
        data_cfg.demo_mode = trainer_cfg.demo_mode
         
        self.data_wrapper = AmassWrapper(**data_cfg)

        # for training we dont need to have it on whole dataset
        if trainer_cfg.train_flag:
            
            
            if self.use_ddp:
                self.train_sampler = DistributedSampler(
                    self.data_wrapper.dataset['train'],
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=True
                )
                self.val_sampler = DistributedSampler(
                    self.data_wrapper.dataset['val'],
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=False
                )
                train_dl = DataLoader(
                    self.data_wrapper.dataset['train'],
                    sampler=self.train_sampler,
                    **self.data_wrapper.train_dataloader_options
                )
                val_dl = DataLoader(
                    self.data_wrapper.dataset['val'],
                    sampler=self.val_sampler,
                    **self.data_wrapper.val_dataloader_options
                )
            else:
                self.train_sampler = None
                self.val_sampler = None
                train_dl = self.data_wrapper.train_dataloader()
                val_dl = self.data_wrapper.val_dataloader()

            self.train_dl = cycle(train_dl)
            self.val_dl = val_dl
            
            
    def save(self, milestone):
        # Only save on the main process to avoid file conflicts
        if not is_main_process():
            return
        # Unwrap DDP module for clean state_dict keys
        denoiser = self.motion_diff_model.denoiser
        if isinstance(denoiser, DDP):
            denoiser = denoiser.module
        data = {'step': self.step,
                'model': denoiser.state_dict(),
                'ema': self.motion_diff_model.ema.state_dict(),
                'scaler': self.scaler.state_dict(),
                'optimizer': self.optimizer.state_dict()}

        milestone = f'{int(milestone):04d}' if milestone != 'best' else milestone
        torch.save(data, os.path.join(self.results_folder, f'model-{milestone}.pt'))

            
    def load(self, milestone, pretrained_path=None):

        if pretrained_path is None:
            milestone = f'{int(milestone):04d}' if milestone != 'best' else milestone
            pretrained_path = os.path.join(self.results_folder, f'model-{milestone}.pt')
        else:
            pretrained_path = model_cfg.pretrained_path
            
        data = torch.load(pretrained_path)

        self.step = data['step']
        self.scaler.load_state_dict(data['scaler']) 
        self.motion_diff_model.ema.load_state_dict(data['ema'], strict=False)

        # Handle loading into DDP-wrapped or unwrapped denoiser
        denoiser = self.motion_diff_model.denoiser
        if isinstance(denoiser, DDP):
            denoiser.module.load_state_dict(data['model'], strict=False)
        else:
            denoiser.load_state_dict(data['model'], strict=False)
        if is_main_process():
            print(f"Loaded weights from {pretrained_path}")

  
 
    def train(self):
        if trainer_cfg.load_weights:
            try: 
                weights = os.listdir(os.path.join(self.results_folder))
                weights_paths = [os.path.join(self.results_folder, weight) for weight in weights]
                weight_path = max(weights_paths, key=os.path.getctime)
        
                print(f"Loading weight: {weight_path}")

                milestone = weight_path.split("/")[-1].split("-")[-1].replace(".pt", "")
                self.load(milestone)
            except:
                print('No weights found. Training from scratch...')

        init_step = self.step
        self.min_val_loss = np.inf

        # Foot-skating loss dominates early training and collapses the right knee;
        # ramp it in linearly over [t1, t2] so the body pose stabilizes first.
        fc_warmup_t1 = 0 # 4000
        fc_warmup_t2 = 2 # 10000
        fc_lambda_target = self.motion_diff_model.diffusion_process.lambda_fc

        # Track pseudo-epoch for DistributedSampler reshuffling
        self._ddp_epoch = 0
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self._ddp_epoch)

        for idx in range(init_step, trainer_cfg.train_num_steps):
             
            self.motion_diff_model.train()

            # Linearly ramp the foot-skating term from 0 to its target over [t1, t2].
            fc_frac = (self.step - fc_warmup_t1) / max(fc_warmup_t2 - fc_warmup_t1, 1)
            fc_frac = min(max(fc_frac, 0.0), 1.0)
            self.motion_diff_model.diffusion_process.lambda_fc = fc_lambda_target * fc_frac

            self.optimizer.zero_grad()
             
            data_dict = next(self.train_dl)
            train_loss_dict_to_log = self.motion_diff_model.allsplit_step('train', data_dict)

            total_loss = train_loss_dict_to_log['total_loss']
            total_loss.backward()

        
            if torch.isnan(total_loss).item() or self.motion_diff_model.check_nans():
                if is_main_process():
                    print('WARNING: NaN loss. Skipping to next data...')
                torch.cuda.empty_cache()
                continue
 
            if trainer_cfg.wandb.use_wandb and is_main_process(): 
                wandb_dict = train_loss_dict_to_log['loss_dict']
                wandb_dict['lr'] = self.optimizer.param_groups[0]['lr']
                wandb.log(wandb_dict)

            if idx % 100 == 0 and is_main_process():
                print("Step: {0} Loss: {1}".format(idx, total_loss))
           
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.motion_diff_model.ema.update()

            # Advance the sampler epoch periodically so each GPU sees different data
            if self.train_sampler is not None and (idx + 1) % 1000 == 0:
                self._ddp_epoch += 1
                self.train_sampler.set_epoch(self._ddp_epoch)
 
            if self.step != 0 and self.step % trainer_cfg.save_and_sample_every == 0:
                self.validate()
        
            self.step += 1

        if is_main_process():
            print('training complete')

        if trainer_cfg.wandb.use_wandb and is_main_process():
            wandb.run.finish()


    def validate(self):
        self.motion_diff_model.ema.ema_model.eval()
        self.motion_diff_model.eval()

        total_val_loss = 0.0
        num_val_batches = 0
        val_loss_accum = {}
        # Per-key denominators: sample counts for per-dataset keys (truly
        # unbiased per-sample mean), batch counts for everything else.
        val_loss_denom = {}

        with torch.no_grad():
            for val_data_batch in tqdm(self.val_dl):
                val_loss_dict_to_log = self.motion_diff_model.allsplit_step('val', val_data_batch, generate_motions=False)
                batch_loss = val_loss_dict_to_log['total_loss'].detach().item()
                total_val_loss += batch_loss
                num_val_batches += 1

                loss_counts = val_loss_dict_to_log.get('loss_counts', {})
                for k, v in val_loss_dict_to_log['loss_dict'].items():
                    val_v = v if isinstance(v, (int, float)) else v.detach().item()
                    if k in loss_counts:
                        # Weighted by per-batch sample count -> true per-sample mean.
                        c = loss_counts[k]
                        val_loss_accum[k] = val_loss_accum.get(k, 0.0) + val_v * c
                        val_loss_denom[k] = val_loss_denom.get(k, 0) + c
                    else:
                        val_loss_accum[k] = val_loss_accum.get(k, 0.0) + val_v
                        val_loss_denom[k] = val_loss_denom.get(k, 0) + 1

            avg_val_loss = total_val_loss / num_val_batches
            avg_loss_dict = {
                k: (v / val_loss_denom[k]) for k, v in val_loss_accum.items() if val_loss_denom[k] > 0
            }
            print(f'Validation Loss: {avg_val_loss}')

            if trainer_cfg.wandb.use_wandb and is_main_process():
                wandb.log(avg_loss_dict)

            if avg_val_loss < self.min_val_loss:
                self.min_val_loss = avg_val_loss
                self.save("best")

            milestone = self.step // trainer_cfg.save_and_sample_every
            self.save(milestone)

        # Synchronize all processes after validation/checkpointing
        if self.use_ddp:
            dist.barrier()


    def test(self, n_samples=500):
        self.motion_diff_model.ema.ema_model.eval()
        self.motion_diff_model.eval()

        for sample_idx in range(n_samples):

            self.motion_diff_model.first_trans = torch.zeros(1, 1, 3, device=data_cfg.device)
            motion_shape = (1, trainer_cfg.window, self.motion_diff_model.nfeats)

            init_noise = torch.randn(motion_shape, device=data_cfg.device)
            _, pure_noise_diffout = self.motion_diff_model.generate_motion(
                torch.zeros(motion_shape, device=data_cfg.device),
                init_vec_method='noise_prev',
                init_vec=init_noise,
                show_progress=False)

            gen_motion_dict = self.motion_diff_model.diffout2motion(pure_noise_diffout)
            gen_motion = gen_motion_dict['full_motion_unnorm']

            if 'contact_masks' in model_cfg.input_feats:
                pred_contact_masks_ = (gen_motion_dict['contact_masks'] > 0.8).float()

            pred_dict, pred_camera_dict = self.motion_diff_model.rot2xyz(gen_motion_dict,
                                                    cpu_flag=True)

            np.save(f"{self.vis_folder}/{sample_idx:06d}.npy",
                    {'pose': gen_motion[0].cpu().numpy()})

            if 'contact_masks' in model_cfg.input_feats:
                pred_contact_masks = pred_contact_masks_[0].detach().cpu().numpy()
            else:
                pred_contact_masks = np.zeros((trainer_cfg.window, len(CONTACT_INDICES)))

            pred_skeleton_dict = {'positions': pred_dict['joints'][0, :, :len(SMPLX_JOINTS)],
                                'contact_masks': pred_contact_masks}
            smpl_params_pred = {"vertices": pred_dict['vertices'][0],
                                    "faces": pred_dict['faces'],
                                    "joints": pred_dict['joints'][0]}
            pred_camera_dict_i = {'camera_rot': pred_camera_dict['camera_rot'][0],
                           "camera_transl": pred_camera_dict['camera_transl'][0],
                           "coef": 2.0}
            self.renderer.render_motion(mesh_list=[smpl_params_pred],
                                        object_dict={},
                                        color = [color1, color2],
                                        skeleton_dict=pred_skeleton_dict,
                                        camera_dict=pred_camera_dict_i,
                                        filename=f"{self.vis_folder}/{sample_idx:06d}")

        print('testing complete')
 
 
 
 

def setup_ddp():
    """Initialize DDP if launched via torchrun / torch.distributed.launch."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return local_rank
    return None


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def run_train():
    trainer = Trainer()
    trainer.train()
    torch.cuda.empty_cache()
    cleanup_ddp()

def run_test():
    trainer = Trainer()
    
    trainer.load(milestone=trainer_cfg.milestone, 
                  pretrained_path=model_cfg.pretrained_path)
    
    trainer.test()
    torch.cuda.empty_cache()
   

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainer-file', default='configs/trainer.yaml') 
    return parser.parse_args()


if __name__ == "__main__":
    
    opt = parse_opt()
 
    trainer_cfg = OmegaConf.load(opt.trainer_file)
 
    model_cfg = OmegaConf.load(trainer_cfg.model_cfg)
    data_cfg = OmegaConf.load(trainer_cfg.data_cfg)
    diffusion_cfg = OmegaConf.load(trainer_cfg.diffusion_cfg)
    loss_cfg = OmegaConf.load(trainer_cfg.loss_cfg)
    
    seed_everything(trainer_cfg.seed)

    # Setup DDP if launched with torchrun; otherwise fall back to single-GPU
    local_rank = setup_ddp()
    if local_rank is not None:
        device = f"cuda:{local_rank}"
    else:
        device = f"cuda:{trainer_cfg.device}" if torch.cuda.is_available() else "cpu"

    if trainer_cfg.train_flag:
        run_train()
    else:
        trainer_cfg.milestone = "best" if trainer_cfg.milestone is None else trainer_cfg.milestone 
        run_test()
    
 

# from tqdm import tqdm; import glob, joblib, torch
# for ii in ['data/motion/precomputed/val', 'data/motion/precomputed/test', 'data/motion/precomputed/train']: 
#     for file in tqdm(glob.glob(f'{ii}/*.p')):
#         data = joblib.load(file)
        
        
        
        