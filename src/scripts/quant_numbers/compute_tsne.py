
import os
import sys
import wandb
import torch
import shutil 
import joblib
import argparse
import numpy as np
import matplotlib.pyplot as plt
 

from omegaconf import OmegaConf
from sklearn.manifold import TSNE
from torch.cuda.amp import GradScaler
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import MultiStepLR

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))



from src.data.amass_dataset import AmassDataset
from src.data.amass_wrapper import AmassWrapper
from src.render.mesh_viz import RendererWrapper
from src.utils.process_utils import BRANCH_NAME, CONTACT_INDICES, SMPLX_JOINTS

from torch.utils.data import DataLoader
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
      
        # self.train_dl = DataLoader(self.data_wrapper.dataset['train'],
        #         shuffle=False,
        #         batch_size=trainer_cfg.batch_size,
        #         num_workers=trainer_cfg.num_workers,
        #         drop_last=True)

        
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

        self.motion_diff_model.ema.ema_model.eval()
        self.motion_diff_model.eval()

        gen_motion_feats = []
        # gt_motion_feats = {dataset_name: [] for dataset_name in data_cfg.hand_art_dataset_list}
        gt_motion_feats = {dataset_name: [] for dataset_name in ['GRAB', 'ARCTIC', 'EMBODY3D']}

         
        # visualize results
        for idx, val_data_dict in enumerate(self.val_dl):
            val_data_dict = {k: v.to(data_cfg.device) if torch.is_tensor(v) else v
                for k, v in val_data_dict.items()}

            K = val_data_dict['length'].shape[0]

            tsne_indices = [_i_ for _i_ in range(K) if val_data_dict['datum']['body_dataset_name'][_i_] in data_cfg.hand_art_dataset_list]
        
            if len(tsne_indices) == 0:
                continue
 
            val_data_dict['motion'] = self.motion_diff_model.norm_and_cat(val_data_dict, model_cfg.input_feats).detach()
  

            if mode == "denoise":
                denoised_diffout = self.motion_diff_model.train_diffusion_forward(val_data_dict)['model_output'].detach()
            
            else:
                # pure_noise_diffout actually means 
                _, denoised_diffout = self.motion_diff_model.generate_motion(val_data_dict['motion'],
                                            init_vec_method='noise',
                                            init_vec=None, 
                                            show_progress=False)

        

            gen_motion_dict = self.motion_diff_model.diffout2motion(denoised_diffout)
            gt_motion_dict = self.motion_diff_model.diffout2motion(val_data_dict['motion'])

            pred_dict, pred_camera_dict = self.motion_diff_model.rot2xyz(gen_motion_dict, 
                                                    body_vtemp=val_data_dict["datum"]['body_vtemp'], 
                                                    betas=val_data_dict["datum"]['betas'],
                                                    gender_list=val_data_dict["datum"]['gender'],
                                                    cpu_flag=False)

            gt_dict, gt_camera_dict = self.motion_diff_model.rot2xyz(gt_motion_dict, 
                                                    body_vtemp=val_data_dict["datum"]['body_vtemp'], 
                                                    betas=val_data_dict["datum"]['betas'], 
                                                    gender_list=val_data_dict["datum"]['gender'],
                                                    cpu_flag=False)

             
            dataset_indices = {dataset_name: [] for dataset_name in gt_motion_feats.keys()}

            for dataset_name in gt_motion_feats.keys():

                dataset_indices[dataset_name] = [_i_ for _i_ in range(K) if val_data_dict['datum']['body_dataset_name'][_i_] == dataset_name]

                if setting == "feature":
                    gt_motion_feats[dataset_name].append(val_data_dict['motion'][dataset_indices[dataset_name]])
                    gen_motion_feats.append(denoised_diffout[dataset_indices[dataset_name]])

                elif setting == "joints":
                    gt_motion_feats[dataset_name].append(gt_dict['joints'].reshape(K, trainer_cfg.window, -1)[dataset_indices[dataset_name]])
                    gen_motion_feats.append(pred_dict['joints'].reshape(K, trainer_cfg.window, -1)[dataset_indices[dataset_name]])

                elif setting == "rotation":
                    gt_motion_feats[dataset_name].append(gt_motion_dict['full_motion_unnorm'][:, :, 3:].reshape(K, trainer_cfg.window, -1)[dataset_indices[dataset_name]])  
                    gen_motion_feats.append(gen_motion_dict['full_motion_unnorm'][:, :, 3:].reshape(K, trainer_cfg.window, -1)[dataset_indices[dataset_name]])


        for dataset_name in gt_motion_feats.keys():
            gt_motion_feats[dataset_name] = torch.cat(gt_motion_feats[dataset_name], dim=0)
        
        dim_list = [gt_motion_feats[dataset_name].shape[0] for dataset_name in gt_motion_feats.keys()]
        gen_motion_feats = torch.cat(gen_motion_feats, dim=0)
        dim_list.append(gen_motion_feats.shape[0])

        embeddings = torch.cat([gt_motion_feats[dataset_name] for dataset_name in gt_motion_feats.keys()] + [gen_motion_feats], dim=0).cpu().numpy()

  
        vis_tsne(embeddings, dim_list=dim_list, labels=list(gt_motion_feats.keys()) + ['Generated'])
 
        print('testing complete')
 
 

def vis_tsne(combined_data, dim_list, labels):

     
    scaler = StandardScaler()
    # Reshape and scale: (N_samples, N_features)
    flat_data = combined_data.reshape(sum(dim_list), -1)
    

    embedded = tsne.fit_transform(flat_data)

    # Visualize
    plt.figure(figsize=(12, 8))
 
    dim_sum = 0

    edgecolors = ['darkblue', 'darkred', 'darkgreen', 'yellow']
    c_list = ['blue', 'red', 'green', 'yellow']

    for _i_, _dim_ in enumerate(dim_list):

        embedded_i = embedded[dim_sum:dim_sum+_dim_]

        plt.scatter(embedded_i[:, 0], embedded_i[:, 1], 
           c=c_list[_i_], alpha=0.7, s=50, label=f'{labels[_i_]} (N={_dim_})', 
           edgecolors=c_list[_i_], linewidth=0.5)

        dim_sum += _dim_


    plt.title('t-SNE Visualization of GT/Generated Samples', fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"fusion_runs/renders/correlation_analysis/tsne_{mode}_{setting}.png")
    

def run_train():
    trainer = Trainer()
    trainer.train()
    torch.cuda.empty_cache()

def run_test():
    trainer = Trainer()
    trainer.load(trainer_cfg.milestone)
    trainer.test()
    torch.cuda.empty_cache()
   

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainer-file', default='configs/tester2_setting0.yaml') 
    return parser.parse_args()


if __name__ == "__main__":

    tsne = TSNE(n_components=2, 
                random_state=42, 
                perplexity=2,       
                init='pca',             
                max_iter=1000)
    opt = parse_opt()

    # setting = "feature"
    # setting = "rotation"
    setting = "joints"

    # mode = "denoise"
    mode = "generate" 
 
    trainer_cfg = OmegaConf.load(opt.trainer_file)
 
    model_cfg = OmegaConf.load(trainer_cfg.model_cfg)
    data_cfg = OmegaConf.load(trainer_cfg.data_cfg)
    diffusion_cfg = OmegaConf.load(trainer_cfg.diffusion_cfg)
    loss_cfg = OmegaConf.load(trainer_cfg.loss_cfg)
    
    seed_everything(trainer_cfg.seed)
    
    device = f"cuda:{trainer_cfg.device}" if torch.cuda.is_available() else "cpu"

      
    trainer_cfg.milestone = "best" if trainer_cfg.milestone is None else trainer_cfg.milestone 
    # run_test()
    

    import glob, joblib, numpy as np, itertools


    gt_filenames = glob.glob('fusion_runs/main/0/vis_res/*_gt.mp4')

    # sample big enough 
    gt_filenames_sampled = np.random.choice(gt_filenames, 60, replace=False)

    os.makedirs('perceptual_study/Raw_Dataset', exist_ok=True)
    os.makedirs('perceptual_study/Merged_Dataset', exist_ok=True)
    os.makedirs('perceptual_study/FUSION_Generation', exist_ok=True)


    HARD_COUNT = 18
    gt_hand_count, gt_merged_count, generation_count = 0, 0, 0

    for _i_, _gt_ in enumerate(gt_filenames_sampled):

        if any([elem in _gt_ for elem in data_cfg.hand_art_dataset_list]) and gt_hand_count < HARD_COUNT:
            gt_hand_count += 1
            foldername = 'perceptual_study/Raw_Dataset'

        elif gt_merged_count < HARD_COUNT:
            gt_merged_count += 1
            foldername = 'perceptual_study/Merged_Dataset'
        else:
            continue
 
        generated_correspondence = os.path.join(os.path.dirname(_gt_), os.path.basename(_gt_).split("_")[0] + '.mp4')
    
        # copy to perceptual_study folder
        shutil.copy(_gt_, f"{foldername}/{os.path.basename(_gt_)}")
        
        if generation_count < HARD_COUNT:
            shutil.copy(generated_correspondence, f"perceptual_study/FUSION_Generation/{os.path.basename(generated_correspondence)}")
            generation_count += 1


        # shutil.copy(_gt_.replace("_gt.mp4", "_pred.mp4"), f"perceptual_study/{os.path.basename(_gt_.replace("_gt.mp4", "_pred.mp4"))}")


        
        



    
    
    