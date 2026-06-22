import os
import shutil
import torch

from src.utils.process_utils import BRANCH_NAME


def get_number_of_stages(optimization_cfg):
    num_stages = 0
    while True:
        try:
            opt_cfg = getattr(optimization_cfg, f'stg{num_stages + 1}')
            assert opt_cfg.num_opt_steps > 0
        except:
            break
        num_stages += 1
    return num_stages


def configure_model(model_cfg, data_cfg, diffusion_cfg, data_wrapper):
    list_of_all_feats = data_wrapper.nfeats
    idx_for_inputs = [data_cfg.load_feats.index(infeat)
                      for infeat in data_wrapper.load_feats]
    total_feats_dim = [list_of_all_feats[i] for i in idx_for_inputs]

    model_cfg.nfeats = sum(total_feats_dim)
    model_cfg.dim_per_feat = total_feats_dim
    model_cfg.input_feats = data_wrapper.load_feats
    model_cfg.norm_type = data_wrapper.preproc.norm_type
    model_cfg.statistics_path = data_wrapper.preproc.stats_file
    model_cfg.diff_params = diffusion_cfg.diff_params
    model_cfg.timestep_respacing = None
    model_cfg.smplx_path = data_cfg.smplx_path
    model_cfg.denoiser_cfg.dropout = 0.0


def setup_output_folders(optimization_cfg, vis_suffix, quant_suffix, config_file_path):
    weights_folder = os.path.join(optimization_cfg.save_dir,
                                  f'{BRANCH_NAME}/{optimization_cfg.exp_id}/weights')
    vis_folder = os.path.join(optimization_cfg.save_dir,
                              f'{BRANCH_NAME}/{optimization_cfg.exp_id}/{vis_suffix}')
    results_folder = os.path.join(optimization_cfg.save_dir,
                                  f'{BRANCH_NAME}/{optimization_cfg.exp_id}/{quant_suffix}')

    os.makedirs(vis_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)

    shutil.copy(config_file_path, os.path.join(vis_folder, "0000_optimization_cfg.yaml"))

    return weights_folder, vis_folder, results_folder


def load_checkpoint(ema, denoiser, weights_folder, milestone, pretrained_path):
    if milestone is not None:
        milestone_str = f'{int(milestone):04d}' if milestone != 'best' else milestone
        pretrained_path = os.path.join(weights_folder, f'model-{milestone_str}.pt')

    data = torch.load(pretrained_path)

    step = data['step']
    ema.load_state_dict(data['ema'], strict=False)
    denoiser.load_state_dict(data['model'], strict=False)
    print(f"Loaded weights from {pretrained_path}")

    return step
