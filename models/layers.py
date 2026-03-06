import math
import torch
from torch import nn
from sklearn.model_selection import KFold
import torch.nn.functional as F
from torch.nn.utils.parametrize import register_parametrization
from torch.nn.utils.parametrizations import orthogonal

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
        
        # flatten -> (batch, 4L)
        enc = enc.view(x.shape[0], -1)
        
        return enc




class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, w0, is_first, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        print('rank', rank)
        # Low-rank factorization
        self.A = nn.Parameter(torch.randn(out_features, rank))
        self.B = nn.Parameter(torch.randn(rank, in_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        # Optional initialization similar to nn.Linear
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

    def forward(self, x):
        # Equivalent to (x @ B.T) @ A.T + b
        out = x @ self.B.T @ self.A.T
        if self.bias is not None:
            out = out + self.bias
        return out



class LatentModulatedSIRENLayer_LORA(nn.Module):
    def __init__(self, in_size, out_size, low_rank_dim =16, latent_modulation_dim= 512., w0=30.,
                 modulate_shift=True, modulate_scale=False, is_first=False, is_last=False):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.latent_modulation_dim = latent_modulation_dim
        self.w0 = w0
        self.modulate_shift = modulate_shift
        self.modulate_scale = modulate_scale
        self.is_first = is_first
        self.is_last = is_last

        self.linear = nn.Linear(in_size, out_size, bias=True)

        if modulate_shift and not is_first and not is_last:
            self.modulate_shift_layer = LowRankLinear(latent_modulation_dim, out_size, rank=low_rank_dim)
        if modulate_scale and not is_first and not is_last:
            self.modulate_shift_layer = LowRankLinear(latent_modulation_dim, out_size, rank=low_rank_dim)


        self.w0 = nn.Parameter(self.w0*torch.ones(1), requires_grad=True)  #make omega a learnable parameter
        print('omega 0 learnable', self.w0)

        self._init(w0, is_first)

    def _init(self, w0, is_first):
        dim_in = self.in_size
        w_std = 1/dim_in if is_first else math.sqrt(6.0 / dim_in) / w0#.item()
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        nn.init.uniform_(self.linear.bias, -w_std, w_std)

        if hasattr(self, 'modulate_shift_layer'):
            nn.init.uniform_(self.modulate_shift_layer.A, -w_std, w_std)
            nn.init.uniform_(self.modulate_shift_layer.B, -w_std, w_std)
            if self.modulate_shift_layer.bias is not None:
                nn.init.uniform_(self.modulate_shift_layer.bias, -w_std, w_std)

    def forward(self, x, latent):
        x = self.linear(x)

        if not self.is_first and not self.is_last:
            shift = 0.0 if not self.modulate_shift else self.modulate_shift_layer(latent)
            scale = 1.0 if not self.modulate_scale else self.modulate_scale_layer(latent)

            if self.modulate_shift:
                if len(shift.shape) == 2:
                    shift = shift.unsqueeze(dim=1)
            if self.modulate_scale:
                if len(scale.shape) == 2:
                    scale = scale.unsqueeze(dim=1)
                
            x = scale * x + shift
            print('added shift', shift.shape, x.shape)

        if not self.is_last:
            x = torch.sin(self.w0 * x)
        return x


class LatentModulatedSIRENLayerCLB(nn.Module):
    def __init__(self, in_size, out_size, latent_modulation_dim: 512, w0=30.,
                 modulate_shift=True, modulate_scale=False, is_first=False, is_last=False):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.latent_modulation_dim = latent_modulation_dim
        self.w0 = w0
        self.modulate_shift = modulate_shift
        self.modulate_scale = modulate_scale
        self.is_first = is_first
        self.is_last = is_last

        self.linear = nn.Linear(in_size, out_size, bias=True)

        if modulate_shift and not is_first and not is_last:
            self.modulate_shift_layer = nn.Linear(latent_modulation_dim, out_size)
            print('latentdim', latent_modulation_dim, out_size)
         #   self.modulate_shift_layer = LowRankLinear(latent_modulation_dim, out_size, rank=16, w0=self.w0, is_first=self.is_first)
            print('nn.linear')
        if modulate_scale and not is_first and not is_last:
            self.modulate_scale_layer = nn.Linear(latent_modulation_dim, out_size)


        self.w0 = nn.Parameter(self.w0*torch.ones(1), requires_grad=True)  #make omega a learnable parameter
        print('omega 0 learnable', self.w0)

        self._init(w0, is_first)

    def _init(self, w0, is_first):
        dim_in = self.in_size
        w_std = 1/dim_in if is_first else math.sqrt(6.0 / dim_in) / w0#.item()
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        nn.init.uniform_(self.linear.bias, -w_std, w_std)

    def forward(self, x, latent):
        x = self.linear(x)
      
        if not self.is_first and not self.is_last:
            shift = 0.0 if not self.modulate_shift else self.modulate_shift_layer(latent)
            scale = 1.0 if not self.modulate_scale else self.modulate_scale_layer(latent)

            if self.modulate_shift:
                if len(shift.shape) == 2:
                    shift = shift.unsqueeze(dim=1)
            if self.modulate_scale:
                if len(scale.shape) == 2:
                    scale = scale.unsqueeze(dim=1)
                
            x = scale * x + shift
          #  print('add shift', x.shape, shift.shape)

        if not self.is_last:
            x = torch.sin(self.w0 * x)
        return x





class LoRALayer(nn.Module):
    """LoRA adaptation layer for a linear layer."""

    def __init__(self, in_features: int, out_features: int, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA parameters (only these are updated in inner loop)
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
      #  self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.lora_B = nn.Parameter(torch.randn(rank, out_features) * 1e-6) 

    def forward(self, x, weight):
        # x: input, weight: base weight matrix
        # weight is [out_features, in_features], F.linear expects this format
        base_out = F.linear(x, weight, None)
        lora_out = (x @ self.lora_A) @ self.lora_B * self.scaling
        return base_out + lora_out




class SharedLORA(nn.Module):
    """Shared Implicit Neural Representation with LoRA conditioning."""

    def __init__(self,
                 coord_dim: int = 2,
                 fourier_dim: int = 256,
                 hidden_dim: int = 256,
                 num_layers: int = 4,
                 lora_rank: int = 4,
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

    def reset_lora_params(self):
        """Reset LoRA parameters to initial values."""
        for lora in self.lora_layers:
            nn.init.normal_(lora.lora_A, std=0.01)
            nn.init.zeros_(lora.lora_B)



class LatentModulatedSIRENLayerCLB_v3(nn.Module):
    def __init__(self, in_size, out_size, latent_modulation_dim: 512, latent_v_dim: 512,  w0=30.,
                 modulate_shift=True, modulate_scale=False, is_first=False, is_last=False):
        super().__init__()
        self.in_size = in_size
       # self.B = B

        self.out_size = out_size
        self.latent_modulation_dim = latent_modulation_dim
        self.latent_v_dim = latent_v_dim
       # print('vdim', self.latent_v_dim )

        self.w0 = w0
        self.modulate_shift = modulate_shift
        self.modulate_scale = modulate_scale
        self.is_first = is_first
        self.is_last = is_last


        #not leranable
      #  self.register_buffer("B", torch.empty(self.latent_v_dim, self.latent_modulation_dim))   #not learnable, but will be stored
      #  nn.init.orthogonal_(self.B)   # ensures columns are orthogonal

       # orth_check = self.B.T @ self.B
      #  print('ortho', torch.allclose(orth_check, torch.eye(self.latent_modulation_dim), atol=1e-5))  # should be True


        #leranable
                # Make B a learnable parameter
       # B = torch.empty(latent_v_dim, latent_modulation_dim)
      #  nn.init.orthogonal_(B)
      #  self.B = nn.Parameter(B)



        self.linear = nn.Linear(in_size, out_size, bias=True)

        if modulate_shift and not is_first and not is_last:
            self.modulate_shift_layer = nn.Linear(latent_modulation_dim, out_size)
           # self.modulate_shift_layer = LowRankLinear(latent_modulation_dim, out_size, rank=16, w0=self.w0, is_first=self.is_first)

            self.v_shift_layer = nn.Linear(latent_v_dim, out_size)

        if modulate_scale and not is_first and not is_last:
            self.modulate_scale_layer = nn.Linear(latent_modulation_dim, out_size)


        self.w0 = nn.Parameter(self.w0*torch.ones(1), requires_grad=True)  #make omega a learnable parameter
        print('omega 0 learnable', self.w0)

        self._init(w0, is_first)

    def _init(self, w0, is_first):
        dim_in = self.in_size
        w_std = 1/dim_in if is_first else math.sqrt(6.0 / dim_in) / w0#.item()
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        nn.init.uniform_(self.linear.bias, -w_std, w_std)
    
   # def reorthogonalize(self):
      #  with torch.no_grad():
     #       U, _, V = torch.linalg.svd(self.B, full_matrices=False)
       #     self.B.copy_(U @ V)


    def forward(self, x, latent, v):

     #   if self.is_first:
        #    print('first', x.shape)
        x = self.linear(x)

        if not self.is_first and not self.is_last:
            #define orthogonal basis
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Move tensors to the same device
           # self.B = self.B.to(device)
            
            
         #   combined =  latent@ self.B.T+ v 
           # print('latent', latent.shape, combined.shape, v.shape)
          #  print('latentmaxmin', latent.max(), latent.min() )

            #shift = 0.0 if not self.modulate_shift else self.modulate_shift_layer(latent)
            shift = 0.0 if not self.modulate_shift else self.modulate_shift_layer(latent)
            shift_v =  0.0 if not self.modulate_shift else self.v_shift_layer(v)
            try:
                #shift = 0.0 if not self.modulate_shift else self.v_shift_layer(combined)
                shift = 0.0 if not self.modulate_shift else self.modulate_shift_layer(latent)
                shift_v =  0.0 if not self.modulate_shift else self.v_shift_layer(v)
                
                if self.modulate_shift:
                    if len(shift.shape) == 2:
                        shift = shift.unsqueeze(dim=1)
                        shift_v = shift_v.unsqueeze(dim=1)
               
                scale = 1.0 if not self.modulate_scale else self.modulate_scale_layer(latent)
                x = scale * x +  shift  + shift_v

            except:


                shift =  0.0 if not self.modulate_shift else self.v_shift_layer(v)
           

                scale = 1.0 if not self.modulate_scale else self.modulate_scale_layer(latent)

                if self.modulate_shift:
                    if len(shift.shape) == 2:
                        shift = shift.unsqueeze(dim=1)
                        shift_v = shift_v.unsqueeze(dim=1)

          

                
                x = scale * x +  shift  + shift_v


            if not self.is_last:
                x = torch.sin(self.w0 * x)
        return x





class LatentModulatedSIRENLayer(nn.Module):
    def __init__(self, in_size, out_size, latent_modulation_dim: 512, w0=30.,
                 modulate_shift=True, modulate_scale=False, is_first=False, is_last=False, k=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.latent_modulation_dim = latent_modulation_dim
        self.w0 = w0
        self.modulate_shift = modulate_shift
        self.modulate_scale = modulate_scale
        self.is_last = is_last
        self.k=k

        self.linear = nn.Linear(in_size, out_size)

        if modulate_shift:
            self.modulate_shift_layer = nn.Linear(latent_modulation_dim, out_size)
        if modulate_scale:
            self.modulate_scale_layer = nn.Linear(latent_modulation_dim, out_size)

        self._init(w0, is_first)

    def _init(self, w0, is_first):
        dim_in = self.linear.weight.size(1)
        w_std = 1/dim_in if is_first else (math.sqrt(6.0/dim_in)/w0)
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        nn.init.uniform_(self.linear.bias, -w_std, w_std)

    def hermite(x,self):
        y=torch.exp(-x**2)*torch.sin(2*x)
        return y

    def forward(self, x, latent):
        x = self.linear(x)
        
        if not self.is_last:
            shift = 0.0 if not self.modulate_shift else self.modulate_shift_layer(latent)
            scale = 1.0 if not self.modulate_scale else self.modulate_scale_layer(latent)

            if self.modulate_shift:
                if len(shift.shape) == 2:
                    shift = shift.unsqueeze(dim=1)
            if self.modulate_scale:
                if len(scale.shape) == 2:
                    scale = scale.unsqueeze(dim=1)

            x = scale * x + shift
           
            x = torch.sin(self.w0 * x)
        return x



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
        print('wo init',w0 )
        dim_in = self.linear.weight.size(1)
        w_std = 1/dim_in if is_first else (math.sqrt(6.0/dim_in)/w0)
      # w_finer = 1/dim_in# if is_first else 1
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        nn.init.uniform_(self.linear.bias, -w_std, -w_std)

    def forward(self, x, latent, v):
       
       # print('x', x.shape, self.in_size, self.out_size)
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
        self.k=k
        self.B = B

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
      # w_finer = 1/dim_in# if is_first else 1
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        nn.init.uniform_(self.linear.bias, -w_std, -w_std)

    def forward(self, x, latent, v):
       
       # print('x', x.shape, self.in_size, self.out_size)
        x = self.linear(x)

        combined =  latent@ self.B.T+ v 
       # print('latent', latent.shape, combined.shape, v.shape)
       
           
          

        if not self.is_first and not self.is_last:
            shift = 0.0 if not self.modulate_shift else self.v_shift_layer(combined)
          
          #  shift = 0.0 if not self.modulate_shift else self.modulate_shift_layer(latent)
          #  shift_v =  0.0 if not self.modulate_shift else self.v_shift_layer(v)
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






        


class LatentModulatedSIRENLayer_v5(nn.Module):
    def __init__(self, in_size, out_size, latent_modulation_dim: 512, latent_v_dim: 512, latent_s_dim: 512,w0=30.,
                 modulate_shift=True, modulate_scale=False, is_first=False, is_last=False, k=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.latent_modulation_dim = latent_modulation_dim
        self.latent_v_dim = latent_v_dim
        self.latent_s_dim = latent_s_dim

        self.w0 = w0
        self.modulate_shift = modulate_shift
        self.modulate_scale = modulate_scale
        self.is_last = is_last
        self.k=k

        self.linear = nn.Linear(in_size, out_size)

        if modulate_shift:
            self.modulate_shift_layer = nn.Linear(latent_modulation_dim, out_size)
            self.v_shift_layer = nn.Linear(latent_v_dim, out_size)
            self.s_shift_layer = nn.Linear(1, out_size)
        if modulate_scale:
            self.modulate_scale_layer = nn.Linear(latent_modulation_dim, out_size)


        self._init(w0, is_first)

    def _init(self, w0, is_first):
        dim_in = self.linear.weight.size(1)
        w_std = 1/dim_in if is_first else (math.sqrt(6.0/dim_in)/w0)
        w_finer = 1/dim_in# if is_first else 1
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        nn.init.uniform_(self.linear.bias, -w_std, -w_std)

    def forward(self, x, latent, v, s):
        x = self.linear(x)
        if not self.is_last:
            shift = 0.0 if not self.modulate_shift else self.modulate_shift_layer(latent)
            shift_v =  0.0 if not self.modulate_shift else self.v_shift_layer(v)
            shift_s =  0.0 if not self.modulate_shift else self.s_shift_layer(s)
            scale = 1.0 if not self.modulate_scale else self.modulate_scale_layer(latent)
            if self.modulate_shift:
                if len(shift.shape) == 2:
                    shift = shift.unsqueeze(dim=1)
                    shift_v = shift_v.unsqueeze(dim=1)
                    shift_s = shift_s.unsqueeze(dim=0)
            if self.modulate_scale:
                if len(scale.shape) == 2:
                    scale = scale.unsqueeze(dim=1)
            x = scale * x + shift + shift_v + shift_s[None,...]
            x=self.w0 * x
            x=torch.sin(x)

        return x


class LatentModulatedSIRENLayer_spatial(nn.Module):
    def __init__(self, in_size, out_size, latent_modulation_dim: 512, w0=30.,
                 modulate_shift=True, modulate_scale=False, is_first=False, is_last=False):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.latent_modulation_dim = latent_modulation_dim
        self.w0 = w0
        self.modulate_shift = modulate_shift
        self.modulate_scale = modulate_scale
        self.is_last = is_last
        

        self.linear = nn.Linear(in_size, out_size)

        if modulate_shift:
           # self.modulate_shift_layer = nn.Linear(latent_modulation_dim, out_size)
            self.modulate_shift_layer = nn.Conv2d(612, out_size, kernel_size=1)
            self.v_shift_layer = nn.Conv2d(512, out_size, kernel_size=1)
                        

  
        self._init(w0, is_first)

    def _init(self, w0, is_first):
        dim_in = self.linear.weight.size(1)
        w_std = 1/dim_in if is_first else (math.sqrt(6.0/dim_in)/w0)
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        nn.init.uniform_(self.linear.bias, -w_std, w_std)

    def get_nearest_patch_vectors(self, latent_tensor, coords):
        """
        Args:
        latent_tensor: torch.Tensor of shape [B, C, H, W] = [4, 64, 32, 32]
        coords: torch.Tensor of shape [B, N, 2], with values in [-1, 1]

        Returns:
            patch_vectors: torch.Tensor of shape [B, N, C] = [4, 2000, 64]
        """
        B, C, H, W = latent_tensor.shape  # [4, 64, 32, 32]
        _, N, _ = coords.shape            # [4, 2000, 2]

        x = coords[..., 0]  # [B, N]
        y = coords[..., 1]  # [B, N]

        # Normalize coordinates to [0, H-1] / [0, W-1] and round
        ix = ((x + 1) / 2 * (W - 1)).round().long()  # [B, N]
        iy = ((y + 1) / 2 * (H - 1)).round().long()  # [B, N]
        # Clamp to valid range
        ix = torch.clamp(ix, 0, W - 1)
        iy = torch.clamp(iy, 0, H - 1)

        # Prepare indexing tensors
        batch_idx = torch.arange(B).view(B, 1).expand(B, N)  # [B, N]

        # Use advanced indexing to extract the [C] vector for each (iy, ix)
        patch_vectors = latent_tensor[batch_idx, :, iy, ix]  # [B, C, N]
        
        # Transpose to [B, N, C]
        patch_vectors = patch_vectors

        return patch_vectors

    def forward(self, x, latent, v):
   
        
        x = self.linear(x)
        
        
        if not self.is_last:
            shift = 0.0 if not self.modulate_shift else self.modulate_shift_layer(latent)
            scale = 1.0 if not self.modulate_scale else self.modulate_scale_layer(latent)
            patch=self.get_nearest_patch_vectors(shift, x)
            
            shift_v =  0.0 if not self.modulate_shift else self.v_shift_layer(v)
            patch_v=self.get_nearest_patch_vectors(shift_v, x)
            print('patch_v', patch_v.shape, patch_v.min(), patch_v.max())


            if self.modulate_shift:
                if len(shift.shape) == 2:
                    shift = shift.unsqueeze(dim=1)

            if self.modulate_scale:
                if len(scale.shape) == 2:
                    scale = scale.unsqueeze(dim=1)
            x = scale * x + patch + patch_v
           
            x = torch.sin(self.w0 * x)
        return x





class LatentModulatedSIRENLayer_spatial_plain(nn.Module):
    def __init__(self, in_size, out_size, latent_modulation_dim: 512, w0=30.,
                 modulate_shift=True, modulate_scale=False, is_first=False, is_last=False):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.latent_modulation_dim = latent_modulation_dim
        self.w0 = w0
        self.modulate_shift = modulate_shift
        self.modulate_scale = modulate_scale
        self.is_last = is_last
        

        self.linear = nn.Linear(in_size, out_size)

        if modulate_shift:
            self.modulate_shift_layer = nn.Conv2d(512, out_size, kernel_size=1)

  
        self._init(w0, is_first)

    def _init(self, w0, is_first):
        dim_in = self.linear.weight.size(1)
        w_std = 1/dim_in if is_first else (math.sqrt(6.0/dim_in)/w0)
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        nn.init.uniform_(self.linear.bias, -w_std, w_std)

    def get_nearest_patch_vectors(self, latent_tensor, coords):
        """
        Args:
        
        
        coords: torch.Tensor of shape [B, N, 2], with values in [-1, 1]

        Returns:
            patch_vectors: torch.Tensor of shape [B, N, C] = [4, 2000, 512]
        """
        B, C, H, W = latent_tensor.shape  # [4, 64, 32, 32]
        _, N, _ = coords.shape            # [4, 2000, 2]

        x = coords[..., 0]  # [B, N]
        y = coords[..., 1]  # [B, N]

        # Normalize coordinates to [0, H-1] / [0, W-1] and round
        ix = ((x + 1) / 2 * (W - 1)).round().long()  # [B, N]
        iy = ((y + 1) / 2 * (H - 1)).round().long()  # [B, N]
        # Clamp to valid range
        ix = torch.clamp(ix, 0, W - 1)
        iy = torch.clamp(iy, 0, H - 1)

        # Prepare indexing tensors
        batch_idx = torch.arange(B).view(B, 1).expand(B, N)  # [B, N]

        # Use advanced indexing to extract the [C] vector for each (iy, ix)
        patch_vectors = latent_tensor[batch_idx, :, iy, ix]  # [B, C, N]
        
        # Transpose to [B, N, C]
        patch_vectors = patch_vectors

        return patch_vectors

    def forward(self, x, latent):
   
        x = self.linear(x)
        
        
        if not self.is_last:
            shift = 0.0 if not self.modulate_shift else self.modulate_shift_layer(latent)
            scale = 1.0 if not self.modulate_scale else self.modulate_scale_layer(latent)
            patch=self.get_nearest_patch_vectors(shift, x)
            
            
            if self.modulate_shift:
                if len(shift.shape) == 2:
                    shift = shift.unsqueeze(dim=1)

            if self.modulate_scale:
                if len(scale.shape) == 2:
                    scale = scale.unsqueeze(dim=1)
          
            x = scale * x + patch 
           
            x = torch.sin(self.w0 * x)
        return x
