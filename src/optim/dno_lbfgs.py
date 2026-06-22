import os
import sys
import math
import copy
import torch
import functools
import numpy as np 
from tqdm import tqdm
from dataclasses import dataclass, field

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
 
from utils.optim_utils import LOSS_KEYS
from utils.process_utils import mask_grad

@dataclass
class DNOOptions:
    
    num_opt_steps: int = field(default=500, metadata={"help": "Number of optimization steps"})
    
    lr: float = field(default=3e-1, metadata={"help": "Learning rate"})

    perturb_scale: float = field(default=0, metadata={"help": "scale of the noise perturbation"})

    diff_penalty_scale: float = field(default=0, metadata={"help": "penalty for the difference between the final z and the initial z"})
    
    lr_warm_up_steps: int = field(default=0, metadata={"help": "Number of warm-up steps for the learning rate"})    
    lr_decay_steps: int = field(default=None, metadata={"help": "Number of decay steps (if None, then set to num_opt_steps)"})


    decorrelate_scale: float = field(default=1, metadata={"help": "penalty for the decorrelation of the noise"})
    decorrelate_dim: int = field(default=1, metadata={"help": "dimension to decorrelate (we usually decorrelate time dimension)"})


    likelihood_scale: float = field(default=0, metadata={"help": "penalty for the decorrelation of the noise"})
    contact_v_scale: float = field(default=30, metadata={"help": "penalty for the foot skating"})
    contact_h_scale: float = field(default=0, metadata={"help": "penalty for the foot height"})
    hips2wrist_scale: float = field(default=0, metadata={"help": "penalty for the foot height"})
    closeness_scale: float = field(default=0, metadata={"help": "penalty for the foot height"})
    object_attendance_scale: float = field(default=0, metadata={"help": "penalty for the foot height"})
    object_penetration_scale: float = field(default=0, metadata={"help": "penalty for the human object penetration"})
    self_penetration_scale: float = field(default=0, metadata={"help": "penalty for the human self penetration"})
    self_contact_scale: float = field(default=0, metadata={"help": "penalty for the foot height"})
    
    parameter_grad_mask: torch.Tensor = field(default=1, metadata={"help": "penalty for the foot height"})
    parameter_lock_mask: torch.Tensor = field(default=1, metadata={"help": "penalty for the foot height"})
    parameter_grad_mask_str: str = field(default="all", metadata={"help": "penalty for the foot height"})
    parameter_lock_mask_str: str = field(default="all", metadata={"help": "penalty for the foot height"})
 
    def __post_init__(self):
        # if lr_decay_steps is not set, then set it to num_opt_steps
        if self.lr_decay_steps is None:
            self.lr_decay_steps = self.num_opt_steps


class DNO:
   
    def __init__(
        self,
        model,
        criterion,
        start_z,
        conf: DNOOptions):
        
        self.model = model
        self.criterion = criterion
        
        # for diff penalty
        self.start_z = start_z.detach()
        self.conf = conf

        self.current_z = self.start_z.clone().requires_grad_(True)

        # excluding the first dimension (batch size)
        self.dims = list(range(1, len(self.start_z.shape)))

        self.optimizer = torch.optim.LBFGS([self.current_z], lr=conf.lr)

        self.lr_scheduler = []

        if conf.lr_warm_up_steps > 0:
            self.lr_scheduler.append(
                lambda step: warmup_scheduler(step, conf.lr_warm_up_steps))
            
        scheduler = lambda step: cosine_decay_scheduler(
            step, conf.lr_decay_steps, conf.num_opt_steps, decay_first=False)
            
        self.lr_scheduler.append(scheduler)

        self.step_count = 0
        self.eps = 1e-10
       
        self.hist = []

    def __call__(self, num_steps: int = None, start_x=None):
        if num_steps is None:
            num_steps = self.conf.num_opt_steps

        min_loss = float("inf")
        min_loss_x, min_loss_z = None, None
        batch_size = self.start_z.shape[0]
        loss_coef_dict = {}

        # torch.autograd.set_detect_anomaly(True)
            
        with tqdm(range(num_steps)) as prog:
            for i in prog:
                info = {"step": [self.step_count] * batch_size}

                # learning rate scheduler
                lr_frac = 1
                if len(self.lr_scheduler) > 0:
                    for scheduler in self.lr_scheduler:
                        lr_frac *= scheduler(self.step_count)
                    self.set_lr(self.conf.lr * lr_frac)
                info["lr"] = [self.conf.lr * lr_frac] * batch_size
 
                # Define closure function for LBFGS
                def closure():
                    self.optimizer.zero_grad()
                    
                    # criterion
                    x = self.model(self.current_z)

                    if self.conf.parameter_lock_mask is not None:
                        x = torch.logical_not(self.conf.parameter_lock_mask) * start_x.clone().detach() + \
                              self.conf.parameter_lock_mask * x
     
                    # update hook here
                    if self.conf.parameter_grad_mask is not None:
                        x.register_hook(functools.partial(mask_grad, mask=self.conf.parameter_grad_mask))
                         
                    loss_dict = self.criterion(x, self.current_z)

                    loss = torch.zeros(batch_size,).to(self.current_z.device)                
                
                    for key in LOSS_KEYS:

                        if self.conf.__getattribute__(key + "_scale") > 0:

                            if key == "decorrelate":
                               loss_dict[key + "_loss"] = noise_regularize_1d(self.current_z, dim=self.conf.decorrelate_dim)
                         
                            temp_loss = self.conf.__getattribute__(key + "_scale") * loss_dict[key + "_loss"]
                            
                            assert temp_loss.shape == (batch_size,), key
                     
                            loss += temp_loss
                        
                    loss_mean = loss.mean()
                    loss_mean.backward()
                    
                    return loss_mean
                
                # Store x for tracking purposes (computed outside closure)
                x = self.model(self.current_z)

                if i == 0 and start_x is None:
                    start_x = x.detach().clone()

                if self.conf.parameter_lock_mask is not None:
                    x = torch.logical_not(self.conf.parameter_lock_mask) * start_x.clone().detach() + \
                          self.conf.parameter_lock_mask * x

                # update hook here
                if self.conf.parameter_grad_mask is not None:
                    x.register_hook(functools.partial(mask_grad, mask=self.conf.parameter_grad_mask))
                     
                loss_dict = self.criterion(x, self.current_z)
                
                loss = torch.zeros(batch_size,).to(self.current_z.device)                
            
                for key in LOSS_KEYS:

                    loss_coef_dict[key] = self.conf.__getattribute__(key + "_scale")

                    if loss_coef_dict[key] > 0:

                        if key == "decorrelate":
                            loss_dict[key + "_loss"] = noise_regularize_1d(self.current_z, dim=self.conf.decorrelate_dim)
                            
                         
                        temp_loss = loss_coef_dict[key] * loss_dict[key + "_loss"]
                        
                        assert temp_loss.shape == (batch_size,), key
                 
                        loss += temp_loss
                        info[key + "_loss"] = loss_dict[key + "_loss"].detach().cpu().numpy()
                    else:
                        info[key + "_loss"] = np.zeros(batch_size)


                info["loss_diff"] = torch.zeros(batch_size) 
                info["loss_decorrelate"] = torch.zeros(batch_size) 
       
                info["loss_sum"] = loss.detach().cpu()
                
                loss_mean = loss.mean()
  
                if loss_mean.item() < min_loss:
                    min_loss = loss_mean.item()
                    min_loss_x = x.detach()
                    
                    min_loss_z = copy.deepcopy(self.current_z.detach())

                # LBFGS optimization step
                self.optimizer.step(closure)
                
                # Calculate gradient norms for logging (after optimization step)
                if self.current_z.grad is not None:
                    grad_norms = self.current_z.grad.norm(p=2, dim=self.dims, keepdim=True)
                    info["grad_norm"] = (grad_norms.squeeze((self.dims)).detach().cpu())
                else:
                    info["grad_norm"] = torch.zeros(batch_size)

                # noise perturbation, match the noise fraction to the learning rate fraction
                noise_frac = lr_frac
                info["perturb_scale"] = [self.conf.perturb_scale * noise_frac] * batch_size

                noise = torch.randn_like(self.current_z)
                self.current_z.data += noise * self.conf.perturb_scale * noise_frac

                # log the norm(z - start_z)
                info["diff_norm"] = ((self.current_z - self.start_z).norm(p=2, dim=self.dims).detach().cpu())

                # log current z
                info["z"] = self.current_z.detach().cpu()
                info["x"] = x.detach().cpu()
                

                self.step_count += 1
                self.hist.append(info)

                if torch.isnan(loss_mean).item():
                    break
    
                
                prog.set_postfix({"loss": info["loss_sum"].mean().item()})

        
            # output is a list (over batch) of dict (over keys) of lists (over steps)
            hist = []
            for i in range(batch_size):
                hist.append({})
                for k in self.hist[0].keys():
                    hist[-1][k] = [info[k][i] for info in self.hist]
            

            return {"z": self.current_z.detach(),  # last step's z
                    "x": x.detach(),               # previous steps' x
                    "min_loss": min_loss,          # best loss
                    "min_loss_x": min_loss_x,      # best x
                    "min_loss_z": min_loss_z,      # best z
                    "loss_coef_dict": loss_coef_dict,
                    "hist": hist}
    
    def set_lr(self, lr):
        """Helper method to set learning rate"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


def warmup_scheduler(step, warmup_steps):
    if step < warmup_steps:
        return step / warmup_steps
    return 1


def cosine_decay_scheduler(step, decay_steps, total_steps, decay_first=True):
    # decay the last "decay_steps" steps from 1 to 0 using cosine decay
    # if decay_first is True, then the first "decay_steps" steps will be decayed from 1 to 0
    # if decay_first is False, then the last "decay_steps" steps will be decayed from 1 to 0
    if step >= total_steps:
        return 0
    if decay_first:
        if step >= decay_steps:
            return 0
        return (math.cos((step) / decay_steps * math.pi) + 1) / 2
    else:
        if step < total_steps - decay_steps:
            return 1
        return (
            math.cos((step - (total_steps - decay_steps)) / decay_steps * math.pi) + 1
        ) / 2


def noise_regularize_1d(noise, stop_at=2, dim=3):
    """
    Args:
        noise (torch.Tensor): (N, C, 1, size)
        stop_at (int): stop decorrelating when size is less than or equal to stop_at
        dim (int): the dimension to decorrelate
    """
    all_dims = set(range(len(noise.shape)))
    loss = 0
    size = noise.shape[dim]

    # pad noise in the size dimention so that it is the power of 2
    if size != 2 ** int(math.log2(size)):
        new_size = 2 ** int(math.log2(size) + 1)
        pad = new_size - size
        pad_shape = list(noise.shape)
        pad_shape[dim] = pad
        pad_noise = torch.randn(*pad_shape).to(noise.device)

        noise = torch.cat([noise, pad_noise], dim=dim)
        size = noise.shape[dim]

    while True:
        # this loss penalizes spatially correlated noise
        # the noise is rolled in the size direction and the dot product is taken
        # (bs, )
        loss = loss + (noise * torch.roll(noise, shifts=1, dims=dim)).mean(
            # average over all dimensions except 0 (batch)
            dim=list(all_dims - {0})
        ).pow(2)

        # stop when size is 8
        if size <= stop_at:
            break

        # (N, C, 1, size) -> (N, C, 1, size // 2, 2)
        noise_shape = list(noise.shape)
        noise_shape[dim] = size // 2
        noise_shape.insert(dim + 1, 2)
        noise = noise.reshape(noise_shape)
        # average pool over (2,) window
        noise = noise.mean([dim + 1])
        size //= 2

    return loss
