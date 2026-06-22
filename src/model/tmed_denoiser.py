import torch
import numpy as np
import torch.nn as nn


class TimestepEmbedderMDM(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
 
        time_embed_dim = self.latent_dim
        self.sequence_pos_encoder = PositionalEncoding(d_model=self.latent_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1,
                 max_len=5000, batch_first=False, negative=False):
        super().__init__()
        self.batch_first = batch_first

        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        
        self.negative = negative
        
        if negative:
            pe = torch.zeros(2*max_len, d_model)
            position = torch.arange(-max_len, max_len, dtype=torch.float).unsqueeze(1)
        else:
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)            

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)    # (max_len, 1, d_model)

        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x, hist_frames=0):
        if not self.negative:
            center = 0
            assert hist_frames == 0
            first = 0
        else:
            center = self.max_len
            first = center-hist_frames
        if self.batch_first:
            last = first + x.shape[1]
            x = x + self.pe.permute(1, 0, 2)[:, first:last, :]
        else:
            last = first + x.shape[0]
            x = x + self.pe[first:last, :]

        return self.dropout(x)

class TMED_denoiser(nn.Module):

    def __init__(self,
                 d_feats: int = 263,
                 latent_dim: list = [1, 256],
                 ff_size: int = 1024,
                 num_layers: int = 9,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 activation: str = "gelu",
                 **kwargs) -> None:

        super().__init__()
        self.latent_dim = latent_dim
        
        # self.feat_comb_coeff = nn.Parameter(torch.tensor([1.0]))
        
        self.pose_proj_in = nn.Linear(d_feats, self.latent_dim)
        self.pose_proj_out = nn.Linear(self.latent_dim, d_feats)
        self.embed_timestep = TimestepEmbedderMDM(self.latent_dim)
    
        self.query_pos = PositionalEncoding(self.latent_dim, dropout)
      
  
        # use torch transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation)
         
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self,
                noised_motion,
                timestep,
                **kwargs):
        
        # 0.  dimension matching
        # noised_motion [latent_dim[0], batch_size, latent_dim] <= [batch_size, latent_dim[0], latent_dim[1]]
        
        bs = noised_motion.shape[0]
        noised_motion = noised_motion.permute(1, 0, 2)
        # 0. check lengths for no vae (diffusion only)
        # if lengths not in [None, []]:


        # 1. time_embedding
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timestep.expand(noised_motion.shape[1]).clone()
        time_emb = self.embed_timestep(timesteps).to(dtype=noised_motion.dtype)
            
        # 4. transformer
        proj_noised_motion = self.pose_proj_in(noised_motion)
 
        xseq = self.query_pos(torch.cat((time_emb, proj_noised_motion), axis=0))        
        tokens = self.encoder(xseq)
        
        # discard time_embedding
        denoised_motion_proj = tokens[1:]  # denoised_motion_proj = tokens[time_emb.shape[0]:]
        denoised_motion = self.pose_proj_out(denoised_motion_proj)
        # denoised_motion_proj (T, B, 512) denoised_motion (T, B, 508)
 
        # 5. [batch_size, latent_dim[0], latent_dim[1]] <= [latent_dim[0], batch_size, latent_dim[1]]
        return denoised_motion.permute(1, 0, 2)