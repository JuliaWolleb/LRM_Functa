import torch
from torch import nn
import torch.nn.functional as F
import math

from models.layers import LatentModulatedSIRENLayer_v3, LatentModulatedSIRENLayer_ortho
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class LatentModulatedSIRENLCB_separate(nn.Module):
    print('got MedFuncta/VidFuncta Model')

    def __init__(self, in_size, out_size, w0s, min_hidden_size=128, max_hidden_size=512, num_layers=5,
                 latent_modulation_dim=512,  latent_v_dim= 512, modulate_shift=True, modulate_scale=False, enable_skip_connections=False,
                 progression_type='linear', guidance = False):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_sizes = self._calculate_progressive_sizes(
            min_hidden_size, max_hidden_size, num_layers, progression_type
        )
        self.enable_skip_connections = enable_skip_connections
        self.guidance = guidance
        print('hidden sizes', self.hidden_sizes)

        print(f"Progressive layer widths: {self.hidden_sizes}")
        layers = []
        for i in range(num_layers - 1):
            is_first = i == 0
            layer_in_size = in_size if is_first else self.hidden_sizes[i - 1]
            layer_out_size = self.hidden_sizes[i]
            layers.append(LatentModulatedSIRENLayer_v3(in_size=layer_in_size, out_size=layer_out_size,
                                                    latent_modulation_dim=latent_modulation_dim,  latent_v_dim=  latent_v_dim  ,     w0=w0s[i],
                                                    modulate_shift=modulate_shift, modulate_scale=modulate_scale,
                                                    is_first=is_first))

        print('vdim', latent_v_dim) 



        self.layers = nn.ModuleList(layers)
        self.last_layer = LatentModulatedSIRENLayer_v3(in_size=self.hidden_sizes[-1], out_size=out_size,
                                                    latent_modulation_dim=latent_modulation_dim ,  latent_v_dim=  latent_v_dim, w0=w0s[-1],
                                                    modulate_shift=modulate_shift, modulate_scale=modulate_scale,
                                                    is_last=True)
        self.modulations = torch.zeros(size=[latent_modulation_dim], requires_grad=True)#.to(device)
        self.vdim = torch.zeros(size=[1,latent_v_dim], requires_grad=True)
        self.encoder = PositionalEncoding2D(input_dim=2, output_dim=128)

    def reset_modulations(self, device):
        self.modulations = self.modulations.detach() * 0
        self.modulations.requires_grad = True
    def reset_vdim(self):
        self.vdim = self.vdim.detach() * 0
        self.vdim.requires_grad = True
    def forward(self, x, fast_params=None,  get_features=False):
     
        x = self.layers[0](x, self.modulations, self.vdim)
 
        for layer in self.layers[1:]:
            y = layer(x, self.modulations, self.vdim)
            if self.enable_skip_connections:
                x = x + y
            else:
                x = y

        features = x
        out = self.last_layer(features, self.modulations, self.vdim) + 0.5
        if get_features:
            return out, features
        else:
            return out
   

    def _calculate_progressive_sizes(self, min_size, max_size, num_layers, progression_type='linear'):
        """Calculate progressive hidden layer sizes."""
        if num_layers <= 1:
            return [min_size]

        # We have num_layers-1 hidden layers (excluding output layer)
        n_hidden = num_layers - 1

        if progression_type == 'linear':
            # Linear interpolation
            sizes = np.linspace(min_size, max_size, n_hidden)
        elif progression_type == 'exponential':
            # Exponential growth
            log_min = np.log(min_size)
            log_max = np.log(max_size)
            log_sizes = np.linspace(log_min, log_max, n_hidden)
            sizes = np.exp(log_sizes)
        elif progression_type == 'cosine':
            # Cosine schedule (slower at beginning and end)
            t = np.linspace(0, 1, n_hidden)
            cosine_factor = (1 - np.cos(t * np.pi)) / 2
            sizes = min_size + (max_size - min_size) * cosine_factor
        else:
            raise ValueError(f"Unknown progression_type: {progression_type}")

        # Round to nearest multiple of 8 for efficiency (optional)
        sizes = [int(8 * round(size / 8)) for size in sizes]

        return sizes


class LatentModulatedSIRENLCB_basic(nn.Module):
    print('got  basic LRM-Functa Model')

    def __init__(self, in_size, out_size, w0s, min_hidden_size=256, max_hidden_size=256, num_layers=5,
                 latent_modulation_dim=512,  latent_v_dim= 512, modulate_shift=True, modulate_scale=False, enable_skip_connections=False,
                 progression_type='linear', ortho = False):
        super().__init__()
        self.num_layers = num_layers
   
        self.hidden_sizes = self._calculate_progressive_sizes(
            min_hidden_size, max_hidden_size, num_layers, progression_type
        )
        self.enable_skip_connections = enable_skip_connections
        print('hidden sizes', self.hidden_sizes)

        w_std =  math.sqrt(6.0 / latent_v_dim) / 50 
       
        self.B = nn.Parameter(torch.empty(latent_v_dim, latent_modulation_dim))
        nn.init.uniform_(self.B, -w_std, w_std)
       

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('device', device)

        print(f"Progressive layer widths: {self.hidden_sizes}")
        layers = []
        for i in range(num_layers - 1):
            is_first = i == 0
            layer_in_size = in_size if is_first else self.hidden_sizes[i - 1]
            layer_out_size = self.hidden_sizes[i]
   
            layers.append(LatentModulatedSIRENLayer_ortho(in_size=layer_in_size, out_size=layer_out_size,
                                                    latent_modulation_dim=latent_modulation_dim,  latent_v_dim=  latent_v_dim  , B=self.B  ,  w0=w0s[i],
                                                    modulate_shift=modulate_shift, modulate_scale=modulate_scale,
                                                    is_first=is_first))

   
        self.layers = nn.ModuleList(layers)
        self.last_layer = LatentModulatedSIRENLayer_ortho(in_size=self.hidden_sizes[-1], out_size=out_size,
                                                    latent_modulation_dim=latent_modulation_dim ,  latent_v_dim=  latent_v_dim, B=self.B  , w0=w0s[-1],
                                                    modulate_shift=modulate_shift, modulate_scale=modulate_scale,
                                                    is_last=True)
        self.modulations = torch.zeros(size=[latent_modulation_dim], requires_grad=True)#.to(device)
        self.vdim = torch.zeros(size=[1,latent_v_dim], requires_grad=True)#.to(device)

    def reset_modulations(self, device):
        self.modulations = self.modulations.detach() * 0
        self.modulations.requires_grad = True
    def reset_vdim(self):
        self.vdim = self.vdim.detach() * 0
        self.vdim.requires_grad = True
    def forward(self, x, fast_params=None,  get_features=False):


        x = self.layers[0](x, self.modulations, self.vdim)
       
        for layer in self.layers[1:]:
            y = layer(x, self.modulations, self.vdim)
            if self.enable_skip_connections:
                x = x + y
            else:
                x = y

        features = x
        out = self.last_layer(features, self.modulations, self.vdim) + 0.5
        if get_features:
            return out, features
        else:
            return out
   

    def _calculate_progressive_sizes(self, min_size, max_size, num_layers, progression_type='linear'):
        """Calculate progressive hidden layer sizes."""
        if num_layers <= 1:
            return [min_size]

        n_hidden = num_layers - 1

        if progression_type == 'linear':
            sizes = np.linspace(min_size, max_size, n_hidden)
        elif progression_type == 'exponential':
            # Exponential growth
            log_min = np.log(min_size)
            log_max = np.log(max_size)
            log_sizes = np.linspace(log_min, log_max, n_hidden)
            sizes = np.exp(log_sizes)
        elif progression_type == 'cosine':
            # Cosine schedule (slower at beginning and end)
            t = np.linspace(0, 1, n_hidden)
            cosine_factor = (1 - np.cos(t * np.pi)) / 2
            sizes = min_size + (max_size - min_size) * cosine_factor
        else:
            raise ValueError(f"Unknown progression_type: {progression_type}")
        sizes = [int(8 * round(size / 8)) for size in sizes]

        return sizes



class LatentModulatedSIRENLCB_ortho(nn.Module):
    print('got orthogonal LRM-Functa model')

    def __init__(self, in_size, out_size, w0s, min_hidden_size=128, max_hidden_size=512, num_layers=5,
                 latent_modulation_dim=512,  latent_v_dim= 512, modulate_shift=True, modulate_scale=False, enable_skip_connections=False,
                 progression_type='linear'):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_sizes = self._calculate_progressive_sizes(
            min_hidden_size, max_hidden_size, num_layers, progression_type
        )
        self.enable_skip_connections = enable_skip_connections
        print('hidden sizes', self.hidden_sizes)


        self.register_buffer("B", torch.empty(latent_v_dim, latent_modulation_dim))   #not learnable, but will be stored
        nn.init.orthogonal_(self.B)   # ensures columns are orthogonal
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.B = self.B.to(device)


        print(f"Progressive layer widths: {self.hidden_sizes}")
        layers = []
        for i in range(num_layers - 1):
            is_first = i == 0
            layer_in_size = in_size if is_first else self.hidden_sizes[i - 1]
            layer_out_size = self.hidden_sizes[i]
            layers.append(LatentModulatedSIRENLayer_ortho(in_size=layer_in_size, out_size=layer_out_size,
                                                    latent_modulation_dim=latent_modulation_dim,  latent_v_dim=  latent_v_dim , B=self.B ,  w0=w0s[i],
                                                    modulate_shift=modulate_shift, modulate_scale=modulate_scale,
                                                    is_first=is_first))

        self.layers = nn.ModuleList(layers)
        self.last_layer = LatentModulatedSIRENLayer_ortho(in_size=self.hidden_sizes[-1], out_size=out_size,
                                                    latent_modulation_dim=latent_modulation_dim ,  latent_v_dim=  latent_v_dim, B=self.B  , w0=w0s[-1],
                                                    modulate_shift=modulate_shift, modulate_scale=modulate_scale,
                                                    is_last=True)
        self.modulations = torch.zeros(size=[latent_modulation_dim], requires_grad=True)#.to(device)
        self.vdim = torch.zeros(size=[1,latent_v_dim], requires_grad=True)#.to(device)

    def reset_modulations(self, device):
        self.modulations = self.modulations.detach() * 0
        self.modulations.requires_grad = True
    def reset_vdim(self):
        self.vdim = self.vdim.detach() * 0
        self.vdim.requires_grad = True
    def forward(self, x, fast_params=None,  get_features=False):
   
        x = self.layers[0](x, self.modulations, self.vdim)
       
        for layer in self.layers[1:]:
            y = layer(x, self.modulations, self.vdim)
            if self.enable_skip_connections:
                x = x + y
            else:
                x = y

        features = x
        out = self.last_layer(features, self.modulations, self.vdim) + 0.5
        if get_features:
            return out, features
        else:
            return out
   

    def _calculate_progressive_sizes(self, min_size, max_size, num_layers, progression_type='linear'):
        """Calculate progressive hidden layer sizes."""
        if num_layers <= 1:
            return [min_size]

        # We have num_layers-1 hidden layers (excluding output layer)
        n_hidden = num_layers - 1

        if progression_type == 'linear':
            # Linear interpolation
            sizes = np.linspace(min_size, max_size, n_hidden)
        elif progression_type == 'exponential':
            # Exponential growth
            log_min = np.log(min_size)
            log_max = np.log(max_size)
            log_sizes = np.linspace(log_min, log_max, n_hidden)
            sizes = np.exp(log_sizes)
        elif progression_type == 'cosine':
            # Cosine schedule (slower at beginning and end)
            t = np.linspace(0, 1, n_hidden)
            cosine_factor = (1 - np.cos(t * np.pi)) / 2
            sizes = min_size + (max_size - min_size) * cosine_factor
        else:
            raise ValueError(f"Unknown progression_type: {progression_type}")

        # Round to nearest multiple of 8 for efficiency (optional)
        sizes = [int(8 * round(size / 8)) for size in sizes]

        return sizes









