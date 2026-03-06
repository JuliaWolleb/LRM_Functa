import torch
from common.utils import psnr
import sklearn
import torch.nn as nn
import torch.optim as optim
from augment import add_gaussian_high_freq_noise_v2
from torch.nn.utils.stateless import functional_call
from torch.nn import ParameterList, Parameter
from functorch import vmap, grad
import numpy as np
import torch
import torch.nn.functional as F

def modulation_consistency(modulations, modulations_bootstrapped, bs):
    """
    A function that calculates the L2-distance between the modulations and a bootstrapped target.
    Proposed in 'Learning Large-scale Neural Fields via Context Pruned Meta-Learning' by Jihoon Tack, et al. (2023)

    Everything is implemented to use this bootstrap correction. It is however NOT USED IN OUR PAPER.
    """
    updated_modulation = modulations_bootstrapped - modulations
    updated_modulation = updated_modulation.view(bs, -1)
    modulation_norm = torch.mean(updated_modulation ** 2, dim=-1)
    return modulation_norm




def modulation_structure(x):
    # Compute the difference with the previous and next row
    diff_prev = x[1:-1] - x[:-2]     # (4, 512)
    diff_next = x[1:-1] - x[2:]      # (4, 512)


    # Sum the differences
    result = diff_prev + diff_next  # (4, 512)

    total_sum =torch.sum(result ** 2)# torch.norm(result, p=2)**2

  #  diff = x[1:] - x[:-1]  # shape: [5, 512]
   # total_sum = torch.sum(diff**2)   

    # Optional: if you want to sum across all rows
    
    return total_sum




def orthogonal_loss(B):
        # B: (d, r)
        BtB = B.T @ B                    # (r, r)
        I = torch.eye(BtB.size(0), device=B.device)
        loss_ortho = torch.norm(BtB - I, p='fro')
        return loss_ortho




def get_grad_norm(grads, detach=True):
    grad_norm_list = []
    for grad in grads:
        if grad is None:
            grad_norm = 0
        else:
            if detach:
                grad_norm = torch.norm(grad.data, p=2, keepdim=True).unsqueeze(dim=0)
            else:
                grad_norm = torch.norm(grad, p=2, keepdim=True).unsqueeze(dim=0)

        grad_norm_list.append(grad_norm)
    return torch.norm(torch.cat(grad_norm_list, dim=0), p=2, dim=1)


def train_step(args, step, model_wrapper, optimizer, Data,  metric_logger, logger, scheduler):
    """
    Function that performs a single meta update
    """
   # criterion_guidance = torch.nn.BCEWithLogitsLoss()


    model_wrapper.model.train()
    device = next(model_wrapper.model.parameters()).device
    model_wrapper.coord_init()  # Reset coordinates
    model_wrapper.model.reset_modulations(device)  # Reset modulations (zero-initialization)
    if args.v_dim >0:
        model_wrapper.model.reset_vdim()
 
    if args.comment == 'separate2':
        batch_size = Data['vid'].size(0)
        data = Data
    else: 
        data = Data


        batch_size = data.size(0)
    
    
    if step % args.print_step == 0:
         input = data#['vid']
         learned_init = data*0

    """ Inner-loop optimization for G steps """


    if args.mode == 'try': 

        loss_in = inner_adapt_v3(model_wrapper=model_wrapper, data=data, step_size=args.inner_lr,
                            num_steps=args.inner_steps, first_order=False, sample_type=args.sample_type)



    elif args.v_dim ==0:
            loss_in = inner_adapt(model_wrapper=model_wrapper, data=data, step_size=args.inner_lr,
                            num_steps=args.inner_steps, first_order=False, sample_type=args.sample_type)



    else:
            loss_in = inner_adapt_v2(model_wrapper=model_wrapper, data=data, step_size=args.inner_lr,
                            num_steps=args.inner_steps, first_order=False, sample_type=args.sample_type)
            modulations = model_wrapper.model.modulations.detach()
            v1 = model_wrapper.model.vdim.detach()
           
            


   
    """ Compute reconstruction loss using full context set"""
    model_wrapper.coord_init()
    if args.v_dim > 0:
        vdim = model_wrapper.model.vdim.clone()
       
    else:
           loss_out = model_wrapper(data) 
    if step % args.print_step == 0:
        images, modulations = model_wrapper()  # Sample images

    """ Bootstrap correction for additional steps (NOT USED IN THIS PAPER) """
    _ = inner_adapt(model_wrapper=model_wrapper, data=data, step_size=args.inner_lr_boot,
                    num_steps=args.inner_steps_boot, first_order=True)
    #modulations_bootstrapped = model_wrapper.model.modulations.detach()
    if step % args.print_step == 0:
        target_boot,_ = model_wrapper()


    """ Classification guidance if guidance is set to TRUE"""

    """ Adversarial attack if adversarial is set to TRUE"""
    if args.adversarial == True:
 
        ortho_loss = orthogonal_loss(model_wrapper.model.B) 
     
        loss_boot = ortho_loss  
       

        """ Modulation consistency loss and loss aggregation (WE ONLY USE RECONSTRUCTION LOSS) """
 
    else:
        loss_boot = 0 * loss_out

    """ Modulation consistency loss between the different v of a batch """
  

    elif args.adversarial == True:
        loss_boot_weighted = args.lam * loss_boot


    else: 
        loss_boot_weighted = 0 * loss_out
    loss = loss_out.mean() + loss_boot_weighted.mean()
    psnro=psnr(loss_out.mean())
 
  

    """ Meta update (optimize shared weights) """
    optimizer.zero_grad()
    torch.autograd.set_detect_anomaly(True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model_wrapper.model.parameters(), 1.0)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()



    torch.cuda.synchronize()
    

    """ Track stats"""
    metric_logger.meters['loss_inner'].update(loss_in.mean().item(), n=batch_size)
    metric_logger.meters['loss_outer'].update(loss_out.mean().item(), n=batch_size)
    metric_logger.meters['psnr_inner'].update(psnr(loss_in).mean().item(), n=batch_size)
    metric_logger.meters['psnr_outer'].update(psnr(loss_out).mean().item(), n=batch_size)
    metric_logger.meters['loss_boot'].update(loss_boot_weighted.mean().item(), n=batch_size)
    metric_logger.synchronize_between_processes()

    if step % args.print_step == 0:
        logger.scalar_summary('train/loss_inner', metric_logger.loss_inner.global_avg, step)
        logger.scalar_summary('train/loss_outer', metric_logger.loss_outer.global_avg, step)
        logger.scalar_summary('train/psnr_inner', metric_logger.psnr_inner.global_avg, step)
        logger.scalar_summary('train/psnr_outer', metric_logger.psnr_outer.global_avg, step)
        logger.scalar_summary('train/loss_boot', metric_logger.loss_boot.global_avg, step)
        logger.log_image('train/img_in', input, step)
        logger.log_image('train/learninit', learned_init, step)
        logger.log_image('train/img_inner', images, step)
        logger.log_image('train/img_bst', target_boot, step)

        logger.log('[TRAIN] [Step %3d] [LossInner %f] [LossOuter %f] [PSNRInner %.3f] [PSNROuter %.3f]' %
                   (step, metric_logger.loss_inner.global_avg, metric_logger.loss_outer.global_avg,
                    metric_logger.psnr_inner.global_avg, metric_logger.psnr_outer.global_avg))

    metric_logger.reset()


def inner_adapt(model_wrapper, data, step_size=1e-2, num_steps=3, first_order=False, sample_type='none'):
    loss = 0.  # Initialize outer_loop loss


    """ Perform num_step (G) inner-loop updates """
    for step_inner in range(num_steps):
        if sample_type != 'none':
            model_wrapper.sample_coordinates(sample_type, data)  # Sample coordinates for the training step
        loss = inner_loop_step(model_wrapper, data, step_size, first_order)
       
    return loss



def inner_adapt_v2(model_wrapper, data, step_size=1e-2, num_steps=3, first_order=False, sample_type='none'):
    loss = 0.  # Initialize outer_loop loss

    """ Perform num_step (G) inner-loop updates """
    for step_inner in range(num_steps):
        if sample_type != 'none':
            model_wrapper.sample_coordinates(sample_type, data)  # Sample coordinates for the training step
        loss = inner_loop_step_v2(model_wrapper, data, step_size, first_order)
    return loss

def inner_adapt_v3(model_wrapper, data, step_size=1e-2, num_steps=3, first_order=False, sample_type='none'):
    loss = 0.  # Initialize outer_loop loss

    """ Perform num_step (G) inner-loop updates subsequent, first v then m """
   
    for step_inner in range(num_steps):
        if sample_type != 'none':
            model_wrapper.sample_coordinates(sample_type,data)  # Sample coordinates for the training step
        loss = inner_loop_step_v(model_wrapper, data, step_size, first_order)
      #  print('inner loss', loss)
    for step_inner in range(num_steps):
        if sample_type != 'none':
            model_wrapper.sample_coordinates(sample_type, data)  # Sample coordinates for the training step
        loss = inner_loop_step(model_wrapper, data, step_size, first_order)
     
    return loss

def inner_loop_step(model_wrapper, data, inner_lr=1e-2, first_order=False):
    batch_size = data.size(0)
    with torch.enable_grad():
        loss = model_wrapper(data)
        grads = torch.autograd.grad(
            loss.mean() * batch_size,
            model_wrapper.model.modulations,
            create_graph=not first_order,
        )[0]
       # print('grads phi', grads.shape, grads.max(), grads.min())
        model_wrapper.model.modulations = model_wrapper.model.modulations - inner_lr * grads
    return loss

def forward_with_params(self, coords, lora_params):
        """Forward pass with specific LoRA parameters."""
        x = self.model.fourier_features(coords)

        # Split lora_params into A and B for each layer
        param_idx = 0
        for i in range(self.model.num_layers):
            lora_A = lora_params[param_idx]
            lora_B = lora_params[param_idx + 1]
            param_idx += 2

            # Compute layer output with LoRA
            # Note: shared_weights[i] is already [out_features, in_features], so use it directly
            base_out = F.linear(x, self.model.shared_weights[i], self.model.shared_biases[i])
            lora_out = (x @ lora_A) @ lora_B * self.model.lora_layers[i].scaling
            x = base_out + lora_out

            # ReLU for all but last layer
            if i < self.model.num_layers - 1:
                x = F.relu(x)

        return torch.sigmoid(x)




def inner_loop_step_v(model_wrapper, data, inner_lr=1e-2, first_order=False):
    batch_size = data.size(0)


    with torch.enable_grad():
        loss = model_wrapper(data)
        grads = torch.autograd.grad(
            loss.mean() * batch_size,
            model_wrapper.model.vdim,
            allow_unused = True,
            create_graph=not first_order,
        )[0]
      #  print('grads v', grads.shape)
        model_wrapper.model.vdim = model_wrapper.model.vdim - inner_lr * grads
    return loss


def inner_loop_step_v2(model_wrapper, data, inner_lr=1e-2, first_order=False):

    batch_size = data.size(0)
    with torch.enable_grad():

        loss = model_wrapper(data)
        grads_v = torch.autograd.grad(
            loss.mean() * batch_size,
            model_wrapper.model.vdim,
            allow_unused=True,
            create_graph=not first_order,
        )[0]
        model_wrapper.model.vdim = model_wrapper.model.vdim - inner_lr * grads_v
       


        grads = torch.autograd.grad(
            loss.mean() * batch_size,
            model_wrapper.model.modulations,
            create_graph=not first_order,
        )[0]
        model_wrapper.model.modulations = model_wrapper.model.modulations - inner_lr * grads
    return loss


def inner_adapt_test_scale(model_wrapper, data, step_size=1e-2, num_steps=3, first_order=False, sample_type='none',
                           scale_type='grad'):

    loss = 0.  # Initialize outer_loop loss
    for step_inner in range(num_steps):
        if sample_type != 'none':
            model_wrapper.sample_coordinates(sample_type, data)
            loss = inner_loop_step_tt_gradscale(model_wrapper, data, step_size, first_order, scale_type)

    return loss


def inner_adapt_test_scale_lora(model_wrapper, data, step_size=1e-2, num_steps=3, first_order=False, sample_type='none',
                           scale_type='grad'):

    loss = 0.  # Initialize outer_loop loss
    for step_inner in range(num_steps):
        if sample_type != 'none':
            model_wrapper.sample_coordinates(sample_type, data)
  
            loss = inner_loop_step_tt_gradscale_lora(model_wrapper, data, step_size, first_order, scale_type)

    return loss    

def inner_adapt_test_scale_time(model_wrapper, Data, step_size=1e-2, num_steps=3, first_order=False, sample_type='none',
                           scale_type='grad'):
    loss = 0.  # Initialize outer_loop loss
    data = Data['vid']
    time = Data['time']
    optimizer_time = optim.SGD(model_wrapper.model.time_layer.parameters(), lr=step_size)

    for step_inner in range(num_steps):
        if sample_type != 'none':
            model_wrapper.sample_coordinates(sample_type, data)
            loss = inner_loop_step_tt_gradscale_time(model_wrapper, data, time, step_size, first_order, scale_type, optimizer_time)

    return loss

def inner_adapt_test_scale_v2(model_wrapper, data, step_size=1e-2, num_steps=3, first_order=False, sample_type='none',
                           scale_type='grad'):
    loss = 0.  # Initialize outer_loop loss
    for step_inner in range(num_steps):
        if sample_type != 'none':
            model_wrapper.sample_coordinates(sample_type, data)
            loss = inner_loop_step_tt_gradscale_v2(model_wrapper, data, step_size, first_order, scale_type)

    return loss



def inner_adapt_test_scale_v3(model_wrapper, data, step_size=1e-2, num_steps=3, first_order=False, sample_type='none',
                           scale_type='grad'):
    loss = 0.  # Initialize outer_loop loss
    print('data test', data.shape)
    for step_inner in range(num_steps):
        if sample_type != 'none':
            model_wrapper.sample_coordinates(sample_type, data)
            loss = inner_loop_step_tt_gradscale_v(model_wrapper, data, step_size, first_order, scale_type)
    for step_inner in range(num_steps):
        if sample_type != 'none':
            model_wrapper.sample_coordinates(sample_type, data)
            loss = inner_loop_step_tt_gradscale(model_wrapper, data, step_size, first_order, scale_type)

    return loss

def inner_adapt_test_scale_v(model_wrapper, data, step_size=1e-2, num_steps=3, first_order=False, sample_type='none',
                           scale_type='grad'):
    loss = 0.  # Initialize outer_loop loss
    print('data test', data.shape)
    for step_inner in range(num_steps):
        if sample_type != 'none':
            model_wrapper.sample_coordinates(sample_type, data)
            loss = inner_loop_step_tt_gradscale_v(model_wrapper, data, step_size, first_order, scale_type)

    return loss




def inner_loop_step_tt_gradscale_new(model_wrapper, data, inner_lr=1e-2,
                                 first_order=False, scale_type='grad'):
    batch_size = data.size(0)

    # --------------------------------------------------------
    # 1. Define a single-sample forward function
    # --------------------------------------------------------
    def single_forward(mod_i, x_i):
        # Save original full modulations
        full_mod = model_wrapper.model.modulations

        # Replace with mod_i temporarily (shape [1, ...])
        model_wrapper.model.modulations = mod_i.unsqueeze(0)

        # Forward for a single sample
        out = model_wrapper(x_i.unsqueeze(0)).squeeze(0)

        # Restore original modulations
        model_wrapper.model.modulations = full_mod
        return out

    # Per-sample gradient operator
    per_sample_grad_fn = vmap(
        grad(lambda m_i, x_i: single_forward(m_i, x_i)),
        in_dims=(0, 0)   # map over batch for modulations and data
    )

    # --------------------------------------------------------
    # 2. FIRST gradient: subsample_grad
    # --------------------------------------------------------
    with torch.enable_grad():
        subsample_grad = per_sample_grad_fn(
            model_wrapper.model.modulations,  # [B, ...]
            data                              # [B, 1, 112, 112]
        )                                      # -> [B, ...]

    # --------------------------------------------------------
    # 3. Reset model + coord init
    # --------------------------------------------------------
    model_wrapper.model.zero_grad()
    model_wrapper.coord_init()

    # --------------------------------------------------------
    # 4. SECOND gradient: grads
    # --------------------------------------------------------
    with torch.enable_grad():
        grads = per_sample_grad_fn(
            model_wrapper.model.modulations,
            data
        )     # -> [B, ...]

    # --------------------------------------------------------
    # 5. Per-sample gradient scaling
    # --------------------------------------------------------
    if scale_type == 'grad':
        subsample_grad_norm = get_grad_norm(subsample_grad, detach=True)  # [B]
        grad_norm = get_grad_norm(grads, detach=True)                    # [B]
        grad_scale = subsample_grad_norm / (grad_norm + 1e-16)           # [B]

        # Reshape to broadcast over modulation dimensions
        grad_scale_ = grad_scale.view((batch_size,) + (1,) * (grads.ndim - 1)).detach()

    else:
        raise NotImplementedError()

    # --------------------------------------------------------
    # 6. Per-sample modulation update (IMPORTANT: .data)
    # --------------------------------------------------------
    with torch.no_grad():
        model_wrapper.model.modulations.data -= inner_lr * grads * grad_scale_

    # --------------------------------------------------------
    # 7. Return loss after update
    # --------------------------------------------------------
    return model_wrapper(data)


def inner_loop_step_tt_gradscale(model_wrapper, data, inner_lr=1e-2, first_order=False, scale_type='grad'):
  #  batch_size = data['vid'].size(0)
    batch_size = data.size(0)
   # print('batch', batch_size)
    model_wrapper.model.zero_grad()
    with torch.enable_grad():
        subsample_loss = model_wrapper(data)
      #  print('subaple loss', subsample_loss.shape)
       # print('data', data.shape)
        subsample_grad = torch.autograd.grad(
            subsample_loss.mean() * batch_size,
           # subsample_loss.sum()/ batch_size,

            model_wrapper.model.modulations,
            create_graph=False,
            allow_unused=True
        )[0]

    model_wrapper.model.zero_grad()
    model_wrapper.coord_init()

    with torch.enable_grad():
        loss = model_wrapper(data)
       # print('loss', loss.shape, 'data', data.shape)

        grads = torch.autograd.grad(
           # loss.sum()/ batch_size,
            loss.mean() * batch_size,
            model_wrapper.model.modulations,
            create_graph=not first_order,
            allow_unused=True
        )[0]


     
  #  print('grads', grads.shape)
    if scale_type == 'grad':
        # Gradient rescaling at test-time
        subsample_grad_norm = get_grad_norm(subsample_grad, detach=True)
        grad_norm = get_grad_norm(grads, detach=True)
        grad_scale = subsample_grad_norm / (grad_norm + 1e-16)
        grad_scale_ = grad_scale.view((batch_size,) + (1,) * (len(grads.shape) - 1)).detach()
    

    else:
        raise NotImplementedError()
    model_wrapper.model.modulations = model_wrapper.model.modulations - inner_lr *grads* grad_scale_ 
   #print('update modulations')

    return loss

def get_param_grad_norms(grads, detach=True):
    norms = []
    for g in grads:
        if g is None:
            norms.append(None)
        else:
            norm = g.norm()
            if detach:
                norm = norm.detach()
            norms.append(norm)
    return norms

def inner_loop_step_tt_gradscale_lora(model_wrapper, data, inner_lr=1e-2, first_order=False, scale_type='grad'):
    batch_size = data.size(0)
    model_wrapper.model.zero_grad()
    fast_params = model_wrapper.model.get_lora_param_dict()

    # --- compute subsample grads ---
    with torch.enable_grad():
        subsample_loss = model_wrapper(data)
        subsample_grads = torch.autograd.grad(
            subsample_loss.mean() * batch_size,
            list(fast_params.values()), 
            create_graph=False,
            allow_unused=True
        )

    subsample_norms = get_param_grad_norms(subsample_grads, detach=True)

    model_wrapper.model.zero_grad()
    model_wrapper.coord_init()

    # --- compute full grads ---
    with torch.enable_grad():
        loss = model_wrapper(data)
        fast_params = model_wrapper.model.get_lora_param_dict()
        grads = torch.autograd.grad(
            loss.mean() * batch_size,
             list(fast_params.values()),
            create_graph=not first_order,
            allow_unused=True
        )
    grad_norms = get_param_grad_norms(grads, detach=True)


    # --- compute per-parameter scales ---
    grad_scales = []
    for sn, gn in zip(subsample_norms, grad_norms):
        if sn is None or gn is None:
            grad_scales.append(None)
        else:
            grad_scales.append(sn / (gn + 1e-16))

    # --- update parameters in-place ---
    with torch.no_grad():
        for key, g, scale in zip(fast_params.keys(), grads, grad_scales):
            param = fast_params[key]
            if g is not None and scale is not None:
                param.copy_(param - inner_lr * g * scale)  # safe in-place update

    return loss



def inner_loop_step_tt_gradscale_v(model_wrapper, data, inner_lr=1e-2, first_order=False, scale_type='grad'):
    batch_size = data.size(0)
    print('batch', batch_size)
    model_wrapper.model.zero_grad()

    with torch.enable_grad():
        subsample_loss = model_wrapper(data)
        subsample_grad = torch.autograd.grad(
            subsample_loss.mean() * batch_size,
            model_wrapper.model.vdim,
            create_graph=False,
            allow_unused=True
        )[0]

    model_wrapper.model.zero_grad()
    model_wrapper.coord_init()

    with torch.enable_grad():
        loss = model_wrapper(data)
     #   print('loss before v', loss.shape)

        grads = torch.autograd.grad(
            loss.mean() * batch_size,
            model_wrapper.model.vdim,
            create_graph=not first_order,
            allow_unused=True
        )[0]
    if scale_type == 'grad':
        # Gradient rescaling at test-time
        subsample_grad_norm = get_grad_norm(subsample_grad, detach=True)
        grad_norm = get_grad_norm(grads, detach=True)
        grad_scale = subsample_grad_norm / (grad_norm + 1e-16)
        grad_scale_ = grad_scale.view((1,) + (1,) * (len(grads.shape) - 1)).detach()
    else:
        raise NotImplementedError()

    
    model_wrapper.model.vdim = model_wrapper.model.vdim - inner_lr *grads* grad_scale_ 


    return loss



def inner_loop_step_tt_gradscale_v2(model_wrapper, data, inner_lr=1e-2, first_order=False, scale_type='grad'):
   # batch_size = data['vid'].size(0)
    batch_size = data.size(0)
    model_wrapper.model.zero_grad()
    

    with torch.enable_grad():

        subsample_loss = model_wrapper(data)

        subsample_grad_v = torch.autograd.grad(
            subsample_loss.mean() * batch_size,
            model_wrapper.model.vdim,
            retain_graph=True,
            allow_unused=True
        )[0]

        subsample_grad = torch.autograd.grad(
            subsample_loss.mean() * batch_size,
            model_wrapper.model.modulations,
            create_graph=False,
            allow_unused=True
        )[0]

    model_wrapper.model.zero_grad()
    model_wrapper.coord_init()

    with torch.enable_grad():
        loss = model_wrapper(data)
        grads_v = torch.autograd.grad(
            loss.mean() * batch_size,
            model_wrapper.model.vdim,
            retain_graph=True,
            allow_unused=True
        )[0]
        if scale_type == 'grad':
            # Gradient rescaling at test-time
            subsample_grad_norm = get_grad_norm(subsample_grad_v, detach=True)
            grad_norm = get_grad_norm(grads_v, detach=True)
            grad_scale = subsample_grad_norm / (grad_norm + 1e-16)
            grad_scale_ = grad_scale.view((1,) + (1,) * (len(grads_v.shape) - 1)).detach()
        else:
            raise NotImplementedError()
        model_wrapper.model.vdim = model_wrapper.model.vdim - inner_lr *grads_v* grad_scale_ 
        loss_m = model_wrapper(data)
        grads = torch.autograd.grad(
            loss_m.mean() * batch_size,
            model_wrapper.model.modulations,
            create_graph=not first_order,
            allow_unused=True
        )[0]



    if scale_type == 'grad':
        # Gradient rescaling at test-time
        subsample_grad_norm = get_grad_norm(subsample_grad, detach=True)
        grad_norm = get_grad_norm(grads, detach=True)
        grad_scale = subsample_grad_norm / (grad_norm + 1e-16)
        grad_scale_ = grad_scale.view((batch_size,) + (1,) * (len(grads.shape) - 1)).detach()
    else:
        raise NotImplementedError()

    model_wrapper.model.modulations = model_wrapper.model.modulations - inner_lr *grads* grad_scale_ 

    return loss_m


