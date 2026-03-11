import math
import torch
from torch import nn
from sklearn.model_selection import KFold
import torch.nn.functional as F
from torch.nn.utils.parametrize import register_parametrization
from torch.nn.utils.parametrizations import orthogonal



class LatentModulatedSIRENLayer_v3(nn.Module):
    def __init__(self, in_size, out_size, latent_modulation_dim: 512, latent_v_dim: 512,w0=30.,
                 modulate_shift=True, modulate_scale=False, is_first=False, is_last=False, k=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.latent_modulation_dim = latent_modulation_dim
        self.latent_v_dim = latent_v_dim
        self.w0 = w0
        self.modulate_shift = modulate_shift
        self.modulate_scale = modulate_scale
        self.is_first = is_first
        self.is_last = is_last
        self.k=k

        self.linear = nn.Linear(in_size, out_size)

        if modulate_shift:
            self.modulate_shift_layer = nn.Linear(latent_modulation_dim, out_size)
            self.v_shift_layer = nn.Linear(latent_v_dim, out_size)
        if modulate_scale:
            self.modulate_scale_layer = nn.Linear(latent_modulation_dim, out_size)


        self._init(w0, is_first)

    def _init(self, w0, is_first):
        dim_in = self.linear.weight.size(1)
        w_std = 1/dim_in if is_first else (math.sqrt(6.0/dim_in)/w0)
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        nn.init.uniform_(self.linear.bias, -w_std, -w_std)

    def forward(self, x, latent, v):
       
        x = self.linear(x)

        if not self.is_first and not self.is_last:
            shift = 0.0 if not self.modulate_shift else self.modulate_shift_layer(latent)
            if self.latent_v_dim > 0:
                shift_v =  0.0 if not self.modulate_shift else self.v_shift_layer(v)

            scale = 1.0 if not self.modulate_scale else self.modulate_scale_layer(latent)
            if self.modulate_shift:
                if len(shift.shape) == 2:
                    shift = shift.unsqueeze(dim=1)
                    if self.latent_v_dim > 0:
                        shift_v = shift_v.unsqueeze(dim=1)
            if self.modulate_scale:
                if len(scale.shape) == 2:
                    scale = scale.unsqueeze(dim=1)
            try:
                if self.latent_v_dim > 0:
                    x = scale * x + shift + shift_v
                else: 
                    x = scale * x + shift

            except:
                print('x', x.shape, shift.shape, shift_v.shape)

            if not self.is_last:
                x = torch.sin(self.w0 * x)
        return x



class LatentModulatedSIRENLayer_ortho(nn.Module):
    def __init__(self, in_size, out_size, latent_modulation_dim: 512, latent_v_dim: 512, B, w0=30.,
                 modulate_shift=True, modulate_scale=False, is_first=False, is_last=False, k=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.latent_modulation_dim = latent_modulation_dim
        self.latent_v_dim = latent_v_dim
        self.w0 = w0
        self.modulate_shift = modulate_shift
        self.modulate_scale = modulate_scale
        self.is_first = is_first
        self.is_last = is_last
        self.B=B
        self.k=k
        print('size', in_size, out_size)
        self.linear = nn.Linear(in_size, out_size)
        if modulate_shift and not is_first and not is_last:
            self.modulate_shift_layer = nn.Linear(latent_modulation_dim, out_size)
            self.v_shift_layer = nn.Linear(latent_v_dim, out_size)

        if modulate_scale:
            self.modulate_scale_layer = nn.Linear(latent_modulation_dim, out_size)

         
        self._init(w0, is_first)

    def _init(self, w0, is_first):
        print('wo init',w0 )
        dim_in = self.linear.weight.size(1)
        w_std = 1/dim_in if is_first else (math.sqrt(6.0/dim_in)/w0)
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        nn.init.uniform_(self.linear.bias, -w_std, -w_std)

    def forward(self, x, latent, v):
       
        x = self.linear(x)

        combined =  latent@ self.B.T+ v 
       

        if not self.is_first and not self.is_last:
            shift = 0.0 if not self.modulate_shift else self.v_shift_layer(combined)
          
            scale = 1.0 if not self.modulate_scale else self.modulate_scale_layer(latent)

            
            if self.modulate_shift:
                if len(shift.shape) == 2:
                    shift = shift.unsqueeze(dim=1)
                  
            if self.modulate_scale:
                if len(scale.shape) == 2:
                    scale = scale.unsqueeze(dim=1)
            try:
                x = scale * x + shift 
            except:
                print('x', x.shape, shift.shape, shift_v.shape)

            if not self.is_last:
                x = torch.sin(self.w0 * x)
        return x

