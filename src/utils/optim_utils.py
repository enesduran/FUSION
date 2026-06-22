import os 
import sys
import torch 
import numpy as np 
from tqdm import tqdm
import matplotlib.pyplot as plt 

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.utils.process_utils import GLOBAL_BODY_INDICES, \
            GLOBAL_LEFT_LEG_INDICES, GLOBAL_RIGHT_LEG_INDICES, \
            GLOBAL_RHAND_INDICES, GLOBAL_LHAND_INDICES, GLOBAL_RHAND_ARM_INDICES, GLOBAL_LHAND_ARM_INDICES

# indicate which body parts can be optimized
BODY_PARTS_MASK_DICT = {'all':         [1, 1, 1, 1, 1, 1, 1, 1], 
                        'global_only': [1, 0, 0, 0, 0, 0, 0, 0], 
                        'global_foot': [1, 1, 1, 0, 0, 0, 0, 0],
                        'body_only':   [1, 1, 1, 1, 1, 1, 0, 0],
                        'hands_arms':  [0, 0, 0, 0, 1, 1, 1, 1],
                        'hands_only':  [0, 0, 0, 0, 0, 0, 1, 1],
                        }
                        
 
# 0 global translation and rotation
# 1 left leg
# 2 right leg
# 3 left arm 
# 4 right arm
# 5 left hand
# 6 right hand

LLM_INSTRUCTION_DICT = {
             0: {'text': 'scratching belly with right thumb', 
                'contact_ids': [8079, 5939], 
                'timesteps': [40, 80], 
                'body_parts': [0, 0, 0, 1, 0, 1, 0]},
             
             1: {'text': 'scratching belly with left hand', 
                'contact_ids': [5361, 4933, 5058, 5169, 5286, 5939], 
                'timesteps': [40, 80], 
                'body_parts': [0, 0, 0, 1, 0, 1, 0]},

             2: {'text': 'left hand touching left shoulder',
                'timesteps': [40, 80], 
                'contact_ids': [5361, 4933, 5058, 5169, 5286, 4439],
                'body_parts': [0, 0, 0, 1, 0, 1, 0]},

             3:{'text': 'touch your noise with a finger',
                'timesteps': [30, 60],
                'contact_ids': [7669, 9120],
                'body_parts': [1, 0, 0, 0, 1, 0, 1]}
              
                }
 

def prepare_lock_masks(mask_type, shape, device):
        
        one_hot_mask = BODY_PARTS_MASK_DICT[mask_type]

        mask = torch.zeros(shape).to(device)

        assert len(one_hot_mask) == 8
        
        # any parameter can change
        if sum(one_hot_mask) == 8:
             mask[:, :, :] = 1
        else: 
             
            # only optimize global rotation and translation, body_transl_delta_pelv, body_orient_xy, z_orient_delta can change
            if one_hot_mask[0]:
                mask[:, :, :3 + 6 + 6] = 1
                
            # left leg
            if one_hot_mask[1]:
                idx = torch.stack([torch.arange(i, i+6) for i in 6*(GLOBAL_LEFT_LEG_INDICES - 1)]).flatten()
                mask[:, :, idx+15] = 1
                

            # right leg
            if one_hot_mask[2]:
                idx = torch.stack([torch.arange(i, i+6) for i in 6*(GLOBAL_RIGHT_LEG_INDICES - 1)]).flatten()
                mask[:, :, idx+15] = 1

            # body only
            if one_hot_mask[3]:          
                idx = torch.stack([torch.arange(i, i+6) for i in 6*(GLOBAL_BODY_INDICES - 1)]).flatten()
                mask[:, :, idx+15] = 1
            
            # left arm
            if one_hot_mask[4]:
                idx = torch.stack([torch.arange(i, i+6) for i in 6*(GLOBAL_LHAND_ARM_INDICES - 1)]).flatten()
                mask[:, :, idx+15] = 1

            # right arm
            if one_hot_mask[5]:
                idx = torch.stack([torch.arange(i, i+6) for i in 6*(GLOBAL_RHAND_ARM_INDICES - 1)]).flatten()
                mask[:, :, idx+15] = 1
   
            # left hand
            if one_hot_mask[6]:
                idx = torch.stack([torch.arange(i, i+6) for i in 6*(GLOBAL_LHAND_INDICES - 1)]).flatten()
                mask[:, :, idx+15] = 1
                

            # right hand
            if one_hot_mask[7]:
                idx = torch.stack([torch.arange(i, i+6) for i in 6*(GLOBAL_RHAND_INDICES - 1)]).flatten()
                mask[:, :, idx+15] = 1
            
        return mask

prepare_grad_masks = prepare_lock_masks
 

def ddim_loop_with_gradient(
    diffusion, 
    model,
    shape,
    noise=None,
    clip_denoised=False,
    denoised_fn=None,
    cond_fn=None,
    model_kwargs=None,
    device=None,
    eta=0.0,
    detach_condition=False,
):
    if device is None:
        device = next(model.parameters()).device
    assert isinstance(shape, (tuple, list))
 

    if noise is not None:
        img = noise
    else:
        img = torch.randn(*shape, device=device)

    for i in list(range(diffusion.num_timesteps))[::-1]:

        t = torch.tensor([i] * shape[0], device=device)

        out = diffusion.ddim_sample(
            model,
            img,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            eta=eta)
        
        img = out["sample"]
    return img
        
@torch.no_grad()
def ddim_invert(
    diffusion,
    model,
    motion,   
    model_kwargs,  
    dump_steps=[],
    num_inference_steps=99,
    eta=0.0,
    clip_denoised=False,
    **kwds,
):
    """
    invert a real motion into noise map with determinisc DDIM inversion
    """
    latents = motion
    print("latents shape: ", latents.shape)
    xt_list = [latents]
    pred_x0_list = [latents]
    indices = list(range(num_inference_steps))  # start_t #  - skip_timesteps))

    for i, t in enumerate(tqdm(indices, desc="DDIM Inversion")):
        t = torch.tensor([t] * latents.shape[0], device=latents.device)
        out = diffusion.ddim_reverse_sample(
            model,
            latents,
            t,
            model_kwargs=model_kwargs,
            eta=eta,
            clip_denoised=clip_denoised,
        )
        latents, pred_x0 = out["sample"], out["pred_xstart"]
        xt_list.append(latents)
        pred_x0_list.append(pred_x0)

    if len(dump_steps) > 0:
        pred_x0_list_out = []
        for ss in reversed(dump_steps):
            print("save step: ", ss)
            pred_x0_list_out.append(pred_x0_list[ss])
        return latents, pred_x0_list_out

    return latents


LOSS_KEYS = ["closeness", "contact_v", "contact_h", "likelihood", "object_attendance", 
             "object_penetration", "self_contact", "self_penetration", "batch_smoothness", "decorrelate"]
 
# "perturb_scale", "loss_diff", "diff_norm", "grad_norm",
PLOT_LOSS_KEYS = ["loss_sum", "lr"] + [elem+"_loss" for elem in LOSS_KEYS]


COLUMN_SIZE = 4
ROW_SIZE = (len(PLOT_LOSS_KEYS) + COLUMN_SIZE - 1) // COLUMN_SIZE

def plot_loss(hist, loss_coefs=None, plot_out_path=''):
    
    plt.figure(figsize=(16, 7), dpi=300)

 
    # Plot loss
    for key_idx, key in enumerate(PLOT_LOSS_KEYS):
        
        plt.subplot(ROW_SIZE, COLUMN_SIZE, key_idx + 1)

        if key in ["loss_sum"]: 
            plt.semilogy(hist["step"], hist[key])
        
            # Plot horizontal red line at lowest point of loss function
            min_loss = min(hist[key])
            plt.axhline(y=min_loss, color="r")
            plt.text(0, min_loss, f"Min Loss: {min_loss:.4f}", color="r")
        else:

            if loss_coefs is not None:

                if key[:-5] in loss_coefs.keys():
                    plt.plot(hist["step"], np.array(hist[key]) * loss_coefs[key[:-5]])
                else:
                    plt.plot(hist["step"], hist[key])
                    
            else:
                plt.plot(hist["step"], hist[key])

            if key in ["closeness_loss", "self_contact_loss"]:
                min_loss = min(hist[key])
                plt.axhline(y=min_loss, color="r")
                plt.text(0, min_loss, f"Min Loss: {min_loss:.4f}", color="r")

        
        # Automatically format y-axis labels
        ax = plt.gca()
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3g}'))

        # Rotate y-axis labels if needed
        plt.setp(ax.get_yticklabels(), rotation=0, ha='right')
 
        plt.legend([key])

    plt.tight_layout(pad=2.0)  # Add padding between subplots     

    if not plot_out_path == '':
        plt.savefig(plot_out_path)

    plt.close()