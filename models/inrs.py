import torch
from torch import nn
import torch.nn.functional as F
import math

from downstream_tasks.transformer import ViT
from downstream_tasks.hemant_MLP import PhaseMLP
from models.layers import LatentModulatedSIRENLayer, LoRALayer, LatentModulatedSIRENLayer_v3, LatentModulatedSIRENLayer_ortho,LatentModulatedSIRENLayer_LORA, LatentModulatedSIRENLayer_v5, LatentModulatedSIRENLayerCLB, LatentModulatedSIRENLayerCLB_v3
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FourierFeatures(nn.Module):
    """Fourier feature mapping for positional encoding."""

    def __init__(self, in_dim: int, out_dim: int, scale: float = 10.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Random Fourier features
        self.B = nn.Parameter(torch.randn(in_dim, out_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        # x: [batch_size, in_dim]
        x_proj = x @ self.B  # [batch_size, out_dim // 2]
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)  # [batch_size, out_dim]

class SharedLORA(nn.Module):
    """Shared Implicit Neural Representation with LoRA conditioning."""

    def __init__(self,
                 coord_dim: int = 2,
                 fourier_dim: int = 256,
                 hidden_dim: int = 256,
                 num_layers: int = 10,
                 lora_rank: int = 16,
                 fourier_scale: float = 10.0):
        super().__init__()

        self.fourier_features = FourierFeatures(coord_dim, fourier_dim, fourier_scale)

        # Shared network weights (meta-learned)
        self.shared_weights = nn.ParameterList()
        self.shared_biases = nn.ParameterList()

        dims = [fourier_dim] + [hidden_dim] * (num_layers - 1) + [1]

        for i in range(len(dims) - 1):
            self.shared_weights.append(nn.Parameter(torch.randn(dims[i + 1], dims[i]) * 0.02))
            self.shared_biases.append(nn.Parameter(torch.zeros(dims[i + 1])))

        # LoRA layers for conditioning
        self.lora_layers = nn.ModuleList([
            LoRALayer(dims[i], dims[i + 1], rank=lora_rank)
            for i in range(len(dims) - 1)
        ])

        self.num_layers = len(self.lora_layers)

    def forward(self, coords):
        """Forward pass through the network.
        coords: [batch_size, 2] - normalized pixel coordinates in [-1, 1]
        """
        x = self.fourier_features(coords)

        for i in range(self.num_layers):
            # Use F.linear with weight and bias directly
            base_out = F.linear(x, self.shared_weights[i], self.shared_biases[i])
            # Add LoRA adaptation
            lora_out = (x @ self.lora_layers[i].lora_A) @ self.lora_layers[i].lora_B * self.lora_layers[i].scaling
            x = base_out + lora_out

            # ReLU for all but last layer
            if i < self.num_layers - 1:
                x = F.relu(x)

        # Sigmoid to output pixel values in [0, 1]
        return torch.sigmoid(x)

    def forward_with_params(self, coords, lora_params):
        """Forward pass with specific LoRA parameters."""
        x = self.fourier_features(coords)

        # Split lora_params into A and B for each layer
        param_idx = 0
        for i in range(self.num_layers):
            lora_A = lora_params[param_idx]
            lora_B = lora_params[param_idx + 1]
            param_idx += 2

            # Compute layer output with LoRA
            # Note: shared_weights[i] is already [out_features, in_features], so use it directly
            base_out = F.linear(x, self.shared_weights[i], self.shared_biases[i])
            lora_out = (x @ lora_A) @ lora_B * self.lora_layers[i].scaling
            x = base_out + lora_out

            # ReLU for all but last layer
            if i < self.num_layers - 1:
                x = F.relu(x)

        return torch.sigmoid(x)

    def get_lora_params(self):
        """Get all LoRA parameters for inner loop optimization."""
        params = []
        for lora in self.lora_layers:
            params.extend([lora.lora_A, lora.lora_B])
        return params

    def get_shared_params(self):
        """Get shared network parameters for outer loop optimization."""
        params = list(self.shared_weights) + list(self.shared_biases)
        return params

    def reset_modulations(self, device):
        """Reset LoRA parameters to initial values."""
        for lora in self.lora_layers:
            nn.init.normal_(lora.lora_A, std=0.01)
            nn.init.normal_(lora.lora_B, mean=0., std=1e-6)
        #    nn.init.zeros_(lora.lora_B)


class PositionalEncoding2D(nn.Module):
    def __init__(self, input_dim=2, output_dim=256):
        super().__init__()
        assert output_dim % (2 * input_dim) == 0, "Output dim must be divisible by 2 * input_dim"
        self.input_dim = input_dim
        self.L = output_dim // (2 * input_dim)  # number of frequency bands per dim
        
        # frequencies = [2^0, 2^1, ..., 2^(L-1)]
        self.freq_bands = 2.0 ** torch.arange(self.L).float()

    def forward(self, x):
        """
        x: (batch, 2) coordinates
        returns: (batch, output_dim) encoded features
        """
        # (batch, 2, L)
        x_expanded = x.unsqueeze(-1) * self.freq_bands.to(x.device)  
        
        # apply sin and cos
        sin_enc = torch.sin(x_expanded)
        cos_enc = torch.cos(x_expanded)
        
        # concatenate along last dim -> (batch, 2, 2L)
        enc = torch.cat([sin_enc, cos_enc], dim=-1)
                # flatten -> (B, N, 4L)
        enc = enc.view(x.shape[0], x.shape[1], -1)
        
       
        
        return enc



class LatentModulatedSIREN(nn.Module):
    def __init__(self,
                 in_size,
                 out_size,
                 hidden_size=256,
                 num_layers=5,
                 latent_modulation_dim=512,
                 w0=30.,
                 w0_increments=0.,
                 modulate_shift=True,
                 modulate_scale=False,
                 enable_skip_connections=True):
        super().__init__()
        layers = []
        print('w0s', w0)
        for i in range(num_layers-1):
            is_first = i == 0
            layer_in_size = in_size if is_first else hidden_size
         #   layer_in_size = hidden_size if is_first else hidden_size
            layers.append(LatentModulatedSIRENLayer(in_size=layer_in_size, out_size=hidden_size,
                                                    latent_modulation_dim=latent_modulation_dim, w0=w0[i],
                                                    modulate_shift=modulate_shift, modulate_scale=modulate_scale,
                                                    is_first=is_first)) #, k=i))
           # w0 += w0_increments  # Allows for layer adaptive w0s
        self.layers = nn.ModuleList(layers)
        self.last_layer = LatentModulatedSIRENLayer(in_size=hidden_size, out_size=out_size,
                                                    latent_modulation_dim=latent_modulation_dim, w0=w0[-1],
                                                    modulate_shift=modulate_shift, modulate_scale=modulate_scale,
                                                    is_last=True)
        self.enable_skip_connections = enable_skip_connections
        self.modulations = torch.zeros(size=[ latent_modulation_dim], requires_grad=True).to(device)

    def reset_modulations(self, device):
        self.modulations = self.modulations.detach() * 0
        self.modulations.requires_grad = True
        

    def forward(self, x, get_features=False):
        
        x = self.layers[0](x, self.modulations)
        for layer in self.layers[1:]:
            y = layer(x, self.modulations)
            if self.enable_skip_connections:
                x = x + y
            else:
                x = y
        features = x
        out = self.last_layer(features, self.modulations) + 0.5

        if get_features:
            return out, features
        else:
            return out



class LORA(nn.Module):
    def __init__(self, in_size, out_size,  min_hidden_size=256, max_hidden_size=256, num_layers=10, w0=20.,
                 w0_increments=10., rank=16 ):

        super().__init__()
        self.num_layers = num_layers
        self.rank = rank
        self.hidden_sizes = self._calculate_progressive_sizes(
            min_hidden_size, max_hidden_size, num_layers, progression_type = 'exponential'
        )
        print(f"Progressive layer widths: {self.hidden_sizes}")
        layers = []
        self.lora_params = []

        
      

        for i in range(num_layers - 1):
            w0 += w0_increments
            is_first = i == 0
            layer_in_size = 256 if is_first else self.hidden_sizes[i - 1]
            layer_out_size = self.hidden_sizes[i]
            layers.append(LoRALayer(in_size=layer_in_size, out_size=layer_out_size, rank = self.rank, w0=w0,is_last=False, is_First=is_first))
           
        w0 += w0_increments
        self.last_layer = LoRALayer(in_size=layer_in_size, out_size=out_size, rank = self.rank, w0=w0, is_last = True, is_First = False)
        layers.append(self.last_layer)
        print('got last layer', out_size)
        self.layers = nn.ModuleList(layers)
        self.encoder = PositionalEncoding2D(input_dim=2, output_dim=256)
        
     #   self.lora_params = nn.ParameterList([
         #   p for layer in self.layers
        #    for p in layer.lora_params_layer
      #  ])
    def lora_params(self):
        # Flatten all layer LoRA parameters into a single list
        params = []
        for layer in self.layers:
            params.append(layer.lora_A)
            params.append(layer.lora_B)
        return params

    def get_lora_param_dict(self):
        """
        Returns a dict of all LoRA params with keys suitable for functional_call
        """
        params = {}
        for i, layer in enumerate(self.layers):
            params[f'layers.{i}.lora_A'] = layer.lora_A
            params[f'layers.{i}.lora_B'] = layer.lora_B
        return params

   # def reset_modulations(self, device):
      #  for layer in self.layers:
       #     layer.reset_parameters(device)

    def reset_modulations(self, device):
        """Reset LoRA parameters to initial values."""
        for lora in self.layers:
            nn.init.normal_(lora.lora_A, std=0.01)
            nn.init.zeros_(lora.lora_B)

    def get_lora_params(self):
        # Return a list of dicts for each layer
        return [{'lora_A': layer.lora_A, 'lora_B': layer.lora_B} for layer in self.layers]
    def lora_params(self):
        return [p for layer in self.layers for p in (layer.lora_A, layer.lora_B)]
   

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


    def forward(self, x, fast_params=None, get_features=False):
        x=self.encoder(x)
      #  x = self.layers[0](x)
        
        for i, layer in enumerate(self.layers):
            if fast_params is None:
                x = layer(x)
            else:
                x = layer(
                    x,
                    lora_A=fast_params[f'layers.{i}.lora_A'],
                    lora_B=fast_params[f'layers.{i}.lora_B']
                )
           
        out = x
       # out = self.last_layer(features) + 0.5
        if get_features:
            return out, out
        else:
            return out

class LatentModulatedSIREN_LORA(nn.Module):
    def __init__(self, in_size, out_size,w0s, low_rank_dim = 16,  min_hidden_size=256, max_hidden_size=512, num_layers=5,
                 latent_modulation_dim=512, modulate_shift=True, modulate_scale=False, enable_skip_connections=True,
                 progression_type='linear'):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_sizes = self._calculate_progressive_sizes(
            min_hidden_size, max_hidden_size, num_layers, progression_type
        )
        print(f"Progressive layer widths: {self.hidden_sizes}")
        layers = []
        for i in range(num_layers - 1):
            is_first = i == 0
            layer_in_size = in_size if is_first else self.hidden_sizes[i - 1]
            layer_out_size = self.hidden_sizes[i]
            layers.append(LatentModulatedSIRENLayer_LORA(in_size=layer_in_size, out_size=layer_out_size, low_rank_dim = low_rank_dim,
                                                    latent_modulation_dim=latent_modulation_dim, w0=w0s[i],
                                                    modulate_shift=modulate_shift, modulate_scale=modulate_scale,
                                                    is_first=is_first))
        self.layers = nn.ModuleList(layers)
        self.last_layer = LatentModulatedSIRENLayer_LORA(in_size=self.hidden_sizes[-1], out_size=out_size, low_rank_dim = low_rank_dim,
                                                    latent_modulation_dim=latent_modulation_dim, w0=w0s[-1],
                                                    modulate_shift=modulate_shift, modulate_scale=modulate_scale,
                                                    is_last=True)
        self.modulations = torch.zeros(size=[latent_modulation_dim], requires_grad=True).to(device)

    def reset_modulations(self, device):
        
        self.modulations = self.modulations.detach() * 0
        self.modulations.requires_grad = True

    def forward(self, x, get_features=False):
        x = self.layers[0](x, self.modulations)
        for layer in self.layers[1:]:
            x = layer(x, self.modulations)
        features = x
        out = self.last_layer(features, self.modulations) + 0.5
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


class LatentModulatedSIRENLCB(nn.Module):
    def __init__(self, in_size, out_size, w0s, min_hidden_size=256, max_hidden_size=256, num_layers=5,
                 latent_modulation_dim=512, modulate_shift=True, modulate_scale=False, enable_skip_connections=True,
                 progression_type='linear'):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_sizes = self._calculate_progressive_sizes(
            min_hidden_size, max_hidden_size, num_layers, progression_type
        )
        print(f"Progressive layer widths: {self.hidden_sizes}")
        print('got clb')
        layers = []
        for i in range(num_layers - 1):
            is_first = i == 0
            layer_in_size = in_size if is_first else self.hidden_sizes[i - 1]
            layer_out_size = self.hidden_sizes[i]
            layers.append(LatentModulatedSIRENLayerCLB(in_size=layer_in_size, out_size=layer_out_size,
                                                    latent_modulation_dim=latent_modulation_dim, w0=w0s[i],
                                                    modulate_shift=modulate_shift, modulate_scale=modulate_scale,
                                                    is_first=is_first))
        self.layers = nn.ModuleList(layers)
        self.last_layer = LatentModulatedSIRENLayerCLB(in_size=self.hidden_sizes[-1], out_size=out_size,
                                                    latent_modulation_dim=latent_modulation_dim, w0=w0s[-1],
                                                    modulate_shift=modulate_shift, modulate_scale=modulate_scale,
                                                    is_last=True)
        self.modulations = torch.zeros(size=[latent_modulation_dim], requires_grad=True).to(device)

    def reset_modulations(self, device):
        self.modulations = self.modulations.detach() * 0
        self.modulations.requires_grad = True

    def forward(self, x, get_features=False):
        x = self.layers[0](x, self.modulations)
        for layer in self.layers[1:]:
            x = layer(x, self.modulations)
        features = x
        out = self.last_layer(features, self.modulations) + 0.5
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


class LatentModulatedSIREN_v2(nn.Module):
    def __init__(self,
                 in_size,
                 out_size,
                 hidden_size=256,
                 num_layers=5,
                 latent_modulation_dim=512,
                 v_dim = 512,
                 w0=30.,
                 w0_increments=0.,
                 modulate_shift=True,
                 modulate_scale=False,
                 enable_skip_connections=True,
                 mode = 'concat'):
        super().__init__()
        layers = []
        print('vdim2', v_dim)
        self.vdim2 = v_dim
        self.mode = mode
        for i in range(num_layers-1):
            is_first = i == 0
            layer_in_size = in_size if is_first else hidden_size
            if v_dim >0:
                if self.mode == 'additive':
                    latentsize = latent_modulation_dim
                else:
                     latentsize = latent_modulation_dim + v_dim   #concatenate vector
                

            else:
                latentsize = latent_modulation_dim
                self.vdim = 0
              
            layers.append(LatentModulatedSIRENLayer(in_size=layer_in_size, out_size=hidden_size,
                                                    latent_modulation_dim=latentsize, w0=w0,
                                                    modulate_shift=modulate_shift, modulate_scale=modulate_scale,
                                                    is_first=is_first))
            w0 += w0_increments  # Allows for layer adaptive w0s
        self.layers = nn.ModuleList(layers)
        self.last_layer = LatentModulatedSIRENLayer(in_size=hidden_size, out_size=out_size,
                                                   latent_modulation_dim=latentsize, w0=w0,
                                                    modulate_shift=modulate_shift, modulate_scale=modulate_scale,
                                                    is_last=True)
        self.enable_skip_connections = enable_skip_connections
        self.modulations = torch.zeros(size=[latent_modulation_dim], requires_grad=True).to(device)
        self.vdim = torch.zeros(size=[1,latent_modulation_dim], requires_grad=True).to(device)

        

    def reset_modulations(self):
        self.modulations = self.modulations.detach() * 0
        self.modulations.requires_grad = True
    
    def reset_vdim(self):
        self.vdim = self.vdim.detach() * 0
        self.vdim.requires_grad = True

    def forward(self, x, get_features=False):
        
        
        if self.vdim2 == 0:

            concat = self.modulations
            
        else: 
            if self.mode =='concat':
                  concat =  torch.cat((self.modulations, self.vdim.repeat(self.modulations.shape[0],1)), dim=1)
                  
            else:
                concat = self.modulations + self.vdim
        
        x = self.layers[0](x, concat)
        for layer in self.layers[1:]:
            y = layer(x, concat)
            if self.enable_skip_connections:
                x = x + y
            else:
                x = y
        features = x
        out = self.last_layer(features, concat) + 0.5   #why is there a +0.5?
        
        if get_features:
            return out, features
        else:
            return out


class LatentModulatedSIRENLCB_v3(nn.Module):
    print('got  LCB v3 mdel')

    def __init__(self, in_size, out_size, w0s, min_hidden_size=128, max_hidden_size=512, num_layers=5,
                 latent_modulation_dim=512,  latent_v_dim= 512, modulate_shift=True, modulate_scale=False, enable_skip_connections=True,
                 progression_type='linear'):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_sizes = self._calculate_progressive_sizes(
            min_hidden_size, max_hidden_size, num_layers, progression_type
        )
        self.enable_skip_connections = enable_skip_connections
        print('hidden sizes', self.hidden_sizes)
     #   B = torch.empty(latent_v_dim, latent_modulation_dim)
      #  nn.init.orthogonal_(B)
      #  self.B = nn.Parameter(B)



        print(f"Progressive layer widths: {self.hidden_sizes}")
        layers = []
        for i in range(num_layers - 1):
            is_first = i == 0
        #    layer_in_size = in_size if is_first else self.hidden_sizes[i - 1]
            layer_in_size = 128 if is_first else self.hidden_sizes[i - 1]
            layer_out_size = self.hidden_sizes[i]
            layers.append(LatentModulatedSIRENLayerCLB_v3(in_size=layer_in_size, out_size=layer_out_size,
                                                    latent_modulation_dim=latent_modulation_dim,  latent_v_dim=  latent_v_dim  ,     w0=w0s[i],
                                                    modulate_shift=modulate_shift, modulate_scale=modulate_scale,
                                                    is_first=is_first))

        print('vdim', latent_v_dim)                                    
        self.layers = nn.ModuleList(layers)
        self.last_layer = LatentModulatedSIRENLayerCLB_v3(in_size=self.hidden_sizes[-1], out_size=out_size,
                                                    latent_modulation_dim=latent_modulation_dim ,  latent_v_dim=  latent_v_dim, w0=w0s[-1],
                                                    modulate_shift=modulate_shift, modulate_scale=modulate_scale,
                                                    is_last=True)
        self.modulations = torch.zeros(size=[latent_modulation_dim], requires_grad=True)#.to(device)
        self.vdim = torch.zeros(size=[1,latent_v_dim], requires_grad=True)#.to(device)
        self.encoder = PositionalEncoding2D(input_dim=2, output_dim=128)

    def reset_modulations(self, device):
        self.modulations = self.modulations.detach() * 0
        self.modulations.requires_grad = True
    def reset_vdim(self):
        self.vdim = self.vdim.detach() * 0
        self.vdim.requires_grad = True
    def forward(self, x, fast_params=None,  get_features=False):
      #  print('x0', x.shape)
        
        x = self.encoder(x)
      #  print('encoded', x.shape)   #hash encoding

        x = self.layers[0](x, self.modulations, self.vdim)
       
       
       
       
        for layer in self.layers[1:]:
            x = layer(x, self.modulations, self.vdim)



      #  for layer in self.layers[1:]:
         #           y = layer(x,  self.modulations, self.vdim)
                 #   if self.enable_skip_connections:
                #        x = x + y
                    #  last = last + x
               #     else:
               #         x = y

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



class LatentModulatedSIRENLCB_separate(nn.Module):
    print('got  LCB separate mdel')

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
          #  layer_in_size = 128 if is_first else self.hidden_sizes[i - 1]
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
        self.vdim = torch.zeros(size=[1,latent_v_dim], requires_grad=True)#.to(device)
        self.encoder = PositionalEncoding2D(input_dim=2, output_dim=128)
       # self.simplecls = PhaseMLP()

    def reset_modulations(self, device):
        self.modulations = self.modulations.detach() * 0
        self.modulations.requires_grad = True
    def reset_vdim(self):
        self.vdim = self.vdim.detach() * 0
        self.vdim.requires_grad = True
    def forward(self, x, fast_params=None,  get_features=False):
      #  print('x0', x.shape)
        
     #   x = self.encoder(x)
      #  print('encoded', x.shape)   #hash encoding

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


class LatentModulatedSIRENLCB_nonortho(nn.Module):
    print('got  LCB nonortho mdel')

    def __init__(self, in_size, out_size, w0s, min_hidden_size=128, max_hidden_size=512, num_layers=5,
                 latent_modulation_dim=512,  latent_v_dim= 512, modulate_shift=True, modulate_scale=False, enable_skip_connections=False,
                 progression_type='linear', ortho = True):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_sizes = self._calculate_progressive_sizes(
            min_hidden_size, max_hidden_size, num_layers, progression_type
        )
        self.enable_skip_connections = enable_skip_connections
        print('hidden sizes', self.hidden_sizes)


        #B not leranable
     #   B = torch.randn(latent_v_dim, latent_modulation_dim)
        w_std =  math.sqrt(6.0 / latent_v_dim) / 50 #.item()
       
        self.B = nn.Parameter(torch.empty(latent_v_dim, latent_modulation_dim))
        nn.init.uniform_(self.B, -w_std, w_std)
       
        if ortho:
            nn.init.orthogonal_(self.B)   #this was not on before
        
        print('B', self.B.shape)


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('device', device)

            # Move tensors to the same device
       # self.B = self.B.to(device)


        print(f"Progressive layer widths: {self.hidden_sizes}")
        layers = []
        for i in range(num_layers - 1):
            is_first = i == 0
            layer_in_size = in_size if is_first else self.hidden_sizes[i - 1]
          #  layer_in_size = 128 if is_first else self.hidden_sizes[i - 1]
            layer_out_size = self.hidden_sizes[i]
            layers.append(LatentModulatedSIRENLayer_ortho(in_size=layer_in_size, out_size=layer_out_size,
                                                    latent_modulation_dim=latent_modulation_dim,  latent_v_dim=  latent_v_dim  , B=self.B  ,  w0=w0s[i],
                                                    modulate_shift=modulate_shift, modulate_scale=modulate_scale,
                                                    is_first=is_first))

        print('vdim', latent_v_dim) 
        self.layers = nn.ModuleList(layers)
        self.last_layer = LatentModulatedSIRENLayer_ortho(in_size=self.hidden_sizes[-1], out_size=out_size,
                                                    latent_modulation_dim=latent_modulation_dim ,  latent_v_dim=  latent_v_dim, B=self.B  , w0=w0s[-1],
                                                    modulate_shift=modulate_shift, modulate_scale=modulate_scale,
                                                    is_last=True)
        self.modulations = torch.zeros(size=[latent_modulation_dim], requires_grad=True)#.to(device)
        self.vdim = torch.zeros(size=[1,latent_v_dim], requires_grad=True)#.to(device)
        self.encoder = PositionalEncoding2D(input_dim=2, output_dim=128)

    def reset_modulations(self, device):
        self.modulations = self.modulations.detach() * 0
        self.modulations.requires_grad = True
    def reset_vdim(self):
        self.vdim = self.vdim.detach() * 0
        self.vdim.requires_grad = True
    def forward(self, x, fast_params=None,  get_features=False):
      #  print('x0', x.shape)
        
      #  x = self.encoder(x)
      #  print('encoded', x.shape)   #hash encoding

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



class LatentModulatedSIRENLCB_ortho(nn.Module):
    print('got  LCB ortho mdel')

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


        #B not leranable
        self.register_buffer("B", torch.empty(latent_v_dim, latent_modulation_dim))   #not learnable, but will be stored
        nn.init.orthogonal_(self.B)   # ensures columns are orthogonal
        print('B', self.B.shape)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('device', device)

            # Move tensors to the same device
        self.B = self.B.to(device)


        print(f"Progressive layer widths: {self.hidden_sizes}")
        layers = []
        for i in range(num_layers - 1):
            is_first = i == 0
            layer_in_size = in_size if is_first else self.hidden_sizes[i - 1]
          #  layer_in_size = 128 if is_first else self.hidden_sizes[i - 1]
            layer_out_size = self.hidden_sizes[i]
            layers.append(LatentModulatedSIRENLayer_ortho(in_size=layer_in_size, out_size=layer_out_size,
                                                    latent_modulation_dim=latent_modulation_dim,  latent_v_dim=  latent_v_dim  , B=self.B  ,  w0=w0s[i],
                                                    modulate_shift=modulate_shift, modulate_scale=modulate_scale,
                                                    is_first=is_first))

        print('vdim', latent_v_dim) 
        self.layers = nn.ModuleList(layers)
        self.last_layer = LatentModulatedSIRENLayer_ortho(in_size=self.hidden_sizes[-1], out_size=out_size,
                                                    latent_modulation_dim=latent_modulation_dim ,  latent_v_dim=  latent_v_dim, B=self.B  , w0=w0s[-1],
                                                    modulate_shift=modulate_shift, modulate_scale=modulate_scale,
                                                    is_last=True)
        self.modulations = torch.zeros(size=[latent_modulation_dim], requires_grad=True)#.to(device)
        self.vdim = torch.zeros(size=[1,latent_v_dim], requires_grad=True)#.to(device)
        self.encoder = PositionalEncoding2D(input_dim=2, output_dim=128)

    def reset_modulations(self, device):
        self.modulations = self.modulations.detach() * 0
        self.modulations.requires_grad = True
    def reset_vdim(self):
        self.vdim = self.vdim.detach() * 0
        self.vdim.requires_grad = True
    def forward(self, x, fast_params=None,  get_features=False):
      #  print('x0', x.shape)
        
      #  x = self.encoder(x)
      #  print('encoded', x.shape)   #hash encoding

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



class LatentModulatedSIREN_v3(nn.Module):
    print('got v3 mdel')
    def __init__(self,
                 in_size,
                 out_size,
                 hidden_size=256,
                 num_layers=5,
                 latent_modulation_dim=512,
                 v_dim = 512,
                 w0=30.,
                 w0_increments=0.,
                 modulate_shift=True,
                 modulate_scale=False,
                 enable_skip_connections=True,
                 mode = 'concat',
                 num_frames = 60,
                 guidance = False):
        super().__init__()
        layers = []
        print('vdim2', v_dim)
        self.vdim2 = v_dim
        self.mode = mode
        self.num_frames = num_frames
        self.guidance = guidance
        for i in range(num_layers-1):
            is_first = i == 0
            layer_in_size = in_size if is_first else hidden_size
            if v_dim >0:
                if self.mode == 'separate' or self.mode == 'delta' or self.mode == 'try':
                    latentsize = latent_modulation_dim
                    latentsize_v = v_dim
                else:
                     latentsize = latent_modulation_dim  #concatenate vector
                

            else:
                latentsize = latent_modulation_dim
                self.vdim = 0
              
            layers.append(LatentModulatedSIRENLayer_v3(in_size=layer_in_size, out_size=hidden_size,
                                                    latent_modulation_dim=latentsize, latent_v_dim= latentsize_v, w0=w0[i],
                                                    modulate_shift=modulate_shift, modulate_scale=modulate_scale,
                                                    is_first=is_first, k=i))
          #  w0 += w0_increments  # Allows for layer adaptive w0s
            print(i, 'w0 at step ', w0)
        self.layers = nn.ModuleList(layers)
        self.last_layer = LatentModulatedSIRENLayer_v3(in_size=hidden_size, out_size=out_size,
                                                   latent_modulation_dim=latentsize, latent_v_dim= latentsize_v, w0=w0[-1],
                                                    modulate_shift=modulate_shift, modulate_scale=modulate_scale,
                                                    is_last=True)
        
        self.enable_skip_connections = enable_skip_connections
        self.modulations = torch.zeros(size=[latent_modulation_dim], requires_grad=True).to(device)
        self.vdim = torch.zeros(size=[1,latent_modulation_dim], requires_grad=True).to(device)
        if self.guidance:
            
         #   self.simplecls =  nn.Sequential(
          #  nn.Flatten(),
          ##  nn.Linear(latentsize*self.num_frames, 512),
          #  nn.ReLU(),
          #  nn.Linear(512, 1),
       # )
            self.simplecls=PhaseMLP()
            print('got phasemlp')
        

    def reset_modulations(self, device):
        self.modulations = self.modulations.detach() * 0
        self.modulations.requires_grad = True
    
    def reset_vdim(self):
        self.vdim = self.vdim.detach() * 0
        self.vdim.requires_grad = True


    def forward(self, x, get_features=False):
        x = self.layers[0](x, self.modulations, self.vdim)
        last=x
        for layer in self.layers[1:]:
            y = layer(x,  self.modulations, self.vdim)
            if self.enable_skip_connections:
               x = x + y
             #  last = last + x
            else:
                x = y
        features = x
        out = self.last_layer(features, self.modulations, self.vdim) + 0.5   #why is there a +0.5?
     #   if self.guidance:
         #   print('modulations', self.modulations.shape)
        #    flat = torch.flatten(self.modulations)
        #    print('flat', flat.shape)
         #   output_reg = self.simplecls(flat[None,...])
         #   print('outreg', output_reg)
        #    return out, output_reg
            
        if get_features:
            return out, features
        else:
            return out



class LatentModulatedSIREN_v5(nn.Module):
    def __init__(self,
                 in_size,
                 out_size,
                 hidden_size=256,
                 num_layers=5,
                 latent_modulation_dim=512,
                 v_dim = 512,
                 s_dim = 512,
                 w0=30.,
                 w0_increments=0.,
                 modulate_shift=True,
                 modulate_scale=False,
                 enable_skip_connections=True,
                 mode = 'separate',
                 num_frames = 60,
                 guidance = False):
        super().__init__()
        layers = []
        print('vdim2', v_dim, s_dim)
        self.vdim2 = v_dim
        self.sdim = s_dim
        self.mode = mode
        self.num_frames = num_frames
        self.guidance = guidance
        for i in range(num_layers-1):
            is_first = i == 0
            layer_in_size = in_size if is_first else hidden_size
            if v_dim >0:
                if self.mode == 'separate':
                    latentsize = latent_modulation_dim
                    latentsize_v = v_dim
                    latentsize_s = s_dim
                else:
                     latentsize = latent_modulation_dim  #concatenate vector
                

            else:
                latentsize = latent_modulation_dim
                self.vdim = 0
              
            layers.append(LatentModulatedSIRENLayer_v5(in_size=layer_in_size, out_size=hidden_size,
                                                    latent_modulation_dim=latentsize, latent_v_dim= latentsize_v,latent_s_dim= latentsize_s, w0=w0,
                                                    modulate_shift=modulate_shift, modulate_scale=modulate_scale,
                                                    is_first=is_first, k=i))
            w0 += w0_increments  # Allows for layer adaptive w0s
        self.layers = nn.ModuleList(layers)
        self.last_layer = LatentModulatedSIRENLayer_v5(in_size=hidden_size, out_size=out_size,
                                                   latent_modulation_dim=latentsize, latent_v_dim= latentsize_v, latent_s_dim= latentsize_s, w0=w0,
                                                    modulate_shift=modulate_shift, modulate_scale=modulate_scale,
                                                    is_last=True)
        self.enable_skip_connections = enable_skip_connections
        self.modulations = torch.zeros(size=[latent_modulation_dim], requires_grad=True).to(device)
        self.vdim = torch.zeros(size=[1, v_dim], requires_grad=True).to(device)
       

        if self.guidance:
         #   self.simplecls =  nn.Sequential(
          #  nn.Flatten(),
          ##  nn.Linear(latentsize*self.num_frames, 512),
          #  nn.ReLU(),
          #  nn.Linear(512, 1),
       # )
            self.simplecls=ViT(img_size=512)
        

    def reset_modulations(self):
        self.modulations = self.modulations.detach() * 0
        self.modulations.requires_grad = True
    
    def reset_vdim(self):
        self.vdim = self.vdim.detach() * 0
        self.vdim.requires_grad = True
       # self.sdim = self.sdim.detach() * 0
      #  self.sdim.requires_grad = True

    def forward(self, x, label, get_features=False):
        
   
        x = self.layers[0](x, self.modulations, self.vdim, label)
        last = x
    

        # for layer in self.layers[1:]:
        #     x = layer(x,  self.modulations, self.vdim)
        #     if self.enable_skip_connections:
        #        x = x + y
        #         last = last + x
        #     else:
        #         x = x
        # features = last

    

        for layer in self.layers[1:]:
            y = layer(x,  self.modulations, self.vdim, label)
            if self.enable_skip_connections:
                x = x + y
              #  last = last + x
            else:
                x = x
        features = x


        out = self.last_layer(features, self.modulations, self.vdim, label) + 0.5   #why is there a +0.5?
     #   if self.guidance:
         #   print('modulations', self.modulations.shape)
        #    flat = torch.flatten(self.modulations)
        #    print('flat', flat.shape)
         #   output_reg = self.simplecls(flat[None,...])
         #   print('outreg', output_reg)
        #    return out, output_reg
            
        if get_features:
            return out, features
        else:
            return out




class LatentModulatedSIREN_v4(nn.Module):
    def __init__(self,
                 in_size,
                 out_size,
                 hidden_size=256,
                 num_layers=5,
                 latent_modulation_dim=512,
                 v_dim =0,
                 w0=30.,
                 w0_increments=0.,
                 modulate_shift=True,
                 modulate_scale=False,
                 enable_skip_connections=True,
                 mode = 'plain'):
        super().__init__()
        layers = []
        for i in range(num_layers-1):
            is_first = i == 0
            layer_in_size = in_size if is_first else hidden_size
            layers.append(LatentModulatedSIRENLayer(in_size=layer_in_size, out_size=hidden_size,
                                                    latent_modulation_dim=latent_modulation_dim, w0=w0,
                                                    modulate_shift=modulate_shift, modulate_scale=modulate_scale,
                                                    is_first=is_first))
            w0 = w0+ w0_increments  # Allows for layer adaptive w0s
        self.layers = nn.ModuleList(layers)
        self.time_layer = nn.Sequential(nn.Linear(1,256), nn.ReLU(), nn.Linear(256, latent_modulation_dim))
      #  self.time_layer =nn.Linear(1,latent_modulation_dim)

        self.vdim = 0
        self.mode = mode
        self.last_layer = LatentModulatedSIRENLayer(in_size=hidden_size, out_size=out_size,
                                                    latent_modulation_dim=latent_modulation_dim, w0=w0,
                                                    modulate_shift=modulate_shift, modulate_scale=modulate_scale,
                                                    is_last=True)
        self.enable_skip_connections = enable_skip_connections
     #   self.modulations = torch.zeros(size=[latent_modulation_dim], requires_grad=True).to(device)

    def reset_modulations(self):
       
          #  self.time_layer.weight.data.zero_()
          #  self.time_layer.bias.data.zero_()
          #  self.time_layer.reset_parameters()
        with torch.no_grad():
              for param in self.time_layer.parameters():
                    param.zero_()


     

    def forward(self, x, t, get_features=False):
        t=t.permute(1,0,2)
        self.modulations = self.time_layer(t)[:,0,:]
        x = self.layers[0](x, self.modulations)
        for layer in self.layers[1:]:
            y = layer(x, self.modulations)
            if self.enable_skip_connections:
                x = x + y
            else:
                x = y
        features = x
        out = self.last_layer(features, self.modulations) + 0.5
        if get_features:
            return out, features
        else:
            return out


