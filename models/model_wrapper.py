import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.nn.utils.stateless import functional_call
from einops import rearrange
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from train.maml_boot import orthogonal_loss


def gaussian_1d(window_size, sigma):
    coords = torch.arange(window_size, dtype=torch.float32)
    gauss = torch.exp(-(coords - window_size // 2) ** 2 / (2 * sigma ** 2))
    return gauss / gauss.sum()

def create_window_1d(window_size, channel):
    _1d_window = gaussian_1d(window_size, 1.5).unsqueeze(0).unsqueeze(0)
    return _1d_window.expand(channel, 1, window_size).contiguous()

def ssim_1d(x, y, window_size=11, data_range=1.0):
    """Compute SSIM for 1D signals (shape: [B, C, L])"""
    assert x.shape == y.shape, "Input tensors must have the same shape"
    channel = x.size(1)
    window = create_window_1d(window_size, channel).to(x.device).type_as(x)

    mu_x = F.conv1d(x, window, padding=window_size // 2, groups=channel)
    mu_y = F.conv1d(y, window, padding=window_size // 2, groups=channel)

    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv1d(x * x, window, padding=window_size // 2, groups=channel) - mu_x2
    sigma_y2 = F.conv1d(y * y, window, padding=window_size // 2, groups=channel) - mu_y2
    sigma_xy = F.conv1d(x * y, window, padding=window_size // 2, groups=channel) - mu_xy

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))

    return ssim_map.mean(dim=2).mean(dim=1)  # per-sample SSIM

def ssim_loss_1d(x, y, window_size=11, data_range=1.0):
    """SSIM loss = 1 - SSIM"""
    return 1 - ssim_1d(x, y, window_size, data_range)
def exists(val):
    return val is not None


class ModelWrapper(nn.Module):
    def __init__(self, args, model):
        super().__init__()
        self.args = args
        self.model = model
        self.data_type = args.data_type
        self.comment = args.comment
        self.lam = args.lam
        self.adversarial = args.adversarial
        self.sampled_coord = None
        self.sampled_index = None
        self.gradncp_coord = None
        self.gradncp_index = None
        self.mode = args.mode
        self.setting = args.setting
        device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
        args.device = device
        self.mode = args.mode
      
        args.data_size = (1, args.img_size, args.img_size)
        if args.dimension == '3d':
             args.data_size = (1, args.num_frames, args.img_size, args.img_size)

        if self.data_type == 'img':
            self.width = args.data_size[1]
            self.height = args.data_size[2]

            mgrid = self.shape_to_coords((self.width, self.height))
            mgrid = rearrange(mgrid, 'h w c -> (h w) c')

        elif self.data_type == 'img3d':
            print('got 3d image')
            self.width = args.data_size[1]
            self.height = args.data_size[2]
            self.depth = args.data_size[3]

            mgrid = self.shape_to_coords((self.width, self.height, self.depth))
            mgrid = rearrange(mgrid, 'h w d c -> (h w d) c')
            print('mgrdig', mgrid.shape)

        elif self.data_type == 'timeseries':
            self.length = args.data_size[-1]
            mgrid = self.shape_to_coords([self.length])

        else:
            raise NotImplementedError()

        self.register_buffer('grid', mgrid)

    def coord_init(self):
        self.sampled_coord = None
        self.sampled_index = None
        self.gradncp_coord = None
        self.gradncp_index = None

    def get_batch_coords(self, x=None):
        if x is None:
            meta_batch_size = 1
        else:
            meta_batch_size = x.size(0)

        # batch of coordinates
        if self.sampled_coord is None and self.gradncp_coord is None:
            coords = self.grid
        elif self.gradncp_coord is not None:
            return self.gradncp_coord, meta_batch_size
        else:
            coords = self.sampled_coord
        coords = coords.clone().detach()[None, ...].repeat((meta_batch_size,) + (1,) * len(coords.shape))
        return coords, meta_batch_size

    def shape_to_coords(self, spatial_dims):
        coords = []
        for i in range(len(spatial_dims)):
            coords.append(torch.linspace(-1.0, 1.0, spatial_dims[i]))
        return torch.stack(torch.meshgrid(*coords, indexing='ij'), dim=-1)

    def sample_coordinates(self, sample_type, data):
        if sample_type == 'random':
            self.random_sample()
        elif sample_type == 'gradncp':
            if random.random() < 0.5:
                self.gradncp(data)
            else:
                self.random_sample()
        else:
            raise NotImplementedError()

    def gradncp(self, x):
        ratio = self.args.data_ratio
        meta_batch_size = x.size(0)
        coords = self.grid
        coords = coords.clone().detach()[None, ...].repeat((meta_batch_size,) + (1,) * len(coords.shape))
        coords = coords.to(self.args.device)
        with torch.no_grad():
            out, feature = self.model(coords, get_features=True)

        if self.data_type == 'img':
            out = rearrange(out, 'b hw c -> b c hw')
            feature = rearrange(feature, 'b hw f -> b f hw')
            x = rearrange(x, 'b c h w -> b c (h w)')
        elif self.data_type == 'img3d':
            out = rearrange(out, 'b hwd c -> b c hwd')
            feature = rearrange(feature, 'b hwd f -> b f hwd')
            x = rearrange(x, 'b c h w d -> b c (h w d)')
        elif self.data_type == 'timeseries':
            out = rearrange(out, 'b l c -> b c l')
            feature = rearrange(feature, 'b l f -> b f l')
        else:
            raise NotImplementedError()

        error = x - out

        gradient = -1 * feature.unsqueeze(dim=1) * error.unsqueeze(dim=2)
        gradient_bias = -1 * error.unsqueeze(dim=2)
        gradient = torch.cat([gradient, gradient_bias], dim=2)
        gradient = rearrange(gradient, 'b c f hw -> b (c f) hw')
        gradient_norm = torch.norm(gradient, dim=1)

        coords_len = gradient_norm.size(1)

        self.gradncp_index = torch.sort(gradient_norm, dim=1, descending=True)[1][:, :int(coords_len * ratio)]
        self.gradncp_coord = torch.gather(coords, 1, self.gradncp_index.unsqueeze(dim=2).repeat(1, 1, self.args.in_size))
        self.gradncp_index = self.gradncp_index.unsqueeze(dim=1).repeat(1, self.args.out_size, 1)

    def random_sample(self):
        coord_size = self.grid.size(0)
        #print('coord', coord_size)
        perm = torch.randperm(coord_size)
        self.sampled_index = perm[:int(self.args.data_ratio * coord_size)]
        self.sampled_coord = self.grid[self.sampled_index]
        return self.sampled_coord

    def forward(self, x=None, lora_params = None, t=torch.zeros(1)):
        if self.data_type == 'img':
            if self.mode == 'time':
                return self.forward_img(x,t)
            elif self.comment == 'separate2':
             try:
                image = x['vid']
                label = x['label']
             except: 
                  image = None
                  label = t.to(self.args.device).float()
             return self.forward_img(image, label)
            else:
                return self.forward_img(x, lora_params)
        if self.data_type == 'img3d':
            return self.forward_img3d(x)
        if self.data_type == 'timeseries':
            return self.forward_timeseries(x)
        else:
            raise NotImplementedError()

    def forward_img(self, x, lora_params=None, t=None):
        coords, meta_batch_size = self.get_batch_coords(x)
        coords = coords.to(self.args.device)
     
        if self.comment == 'separate2':
            t = t.to(self.args.device)
            out = self.model(coords,t)
         
        elif self.setting == 'lora':
            out = self.model.forward_with_params(coords, lora_params)

        else:
            out =  self.model(coords)[:meta_batch_size,...]
           

        compute_kl=False
        if compute_kl==True:
                kl_div =  -0.5 * torch.sum(1 + self.model.modulations_std - self.model.modulations_mean.pow(2) - self.model.modulations_std.exp(), dim=1)
        
        if self.adversarial == True:

            ortho_loss = orthogonal_loss(self.model.B) 
     
            loss_boot =  self.lam * ortho_loss  
       
         
        else:
            loss_boot = 0

        out = rearrange(out, 'b hw c -> b c hw')
        if exists(x):
            if self.sampled_coord is None and self.gradncp_coord is None:
                 return F.mse_loss(x.view(meta_batch_size, -1), out.reshape(meta_batch_size, -1), reduce=False).mean(dim=1)  

            elif self.gradncp_coord is not None:
                x = rearrange(x, 'b c h w -> b c (h w)')
                x = torch.gather(x, 2, self.gradncp_index)
                print('b')

                return F.mse_loss(x.view(meta_batch_size, -1), out.reshape(meta_batch_size, -1), reduce=False).mean(dim=1)
            else:

                x = rearrange(x, 'b c h w -> b c (h w)')[:, :, self.sampled_index]
            
             
                mseloss = F.mse_loss(x.view(meta_batch_size, -1), out.reshape(meta_batch_size, -1), reduce=False).mean(dim=1) 
           
                return  loss_boot + mseloss
        
        out = rearrange(out, 'b c (h w) -> b c h w', h=self.height, w=self.width)
      
        return out, self.model.modulations#self.model.lora_params# self.model.modulations

    def forward_img3d(self, x):
        coords, meta_batch_size = self.get_batch_coords(x)
        coords = coords.to(self.args.device)

        out = self.model(coords)
        out = rearrange(out, 'b whd c -> b c whd')

        if exists(x):
            if self.sampled_coord is None and self.gradncp_coord is None:
                return F.mse_loss(x.view(meta_batch_size, -1), out.reshape(meta_batch_size, -1), reduce=False).mean(dim=1)
            elif self.gradncp_coord is not None:
                x = rearrange(x, 'b c w h d -> b c (w h d)')
                x = torch.gather(x, 2, self.gradncp_index)
                return F.mse_loss(x.view(meta_batch_size, -1), out.reshape(meta_batch_size, -1), reduce=False).mean(dim=1)
            else:
                x = rearrange(x, 'b c w h d -> b c (w h d)')[:, :, self.sampled_index]
                return F.mse_loss(x.view(meta_batch_size, -1), out.reshape(meta_batch_size, -1), reduce=False).mean(dim=1)
  

        out = rearrange(out, 'b c (w h d) -> b c w h d', h=self.height, w=self.width, d=self.depth)  #should be (1,1,10,112,112)

        return out

  