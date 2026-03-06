import torch
import torchvision.transforms as transforms
from train.maml_boot import inner_adapt_test_scale, inner_adapt_test_scale_v3, inner_adapt_test_scale_v2
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit_nfs(args, model_wrapper, dataloader, double = False, set=None):

    model_wrapper.model.eval()
    model_wrapper.coord_init()

    for n, data in enumerate(dataloader):
        if double == False:
            data, label = data
           
            if  data.isnan().any():
                        print('got nan')
                        continue
            if args.dimension == 'swap':
                data = torch.permute(data, (1, 0, 2, 3))

     #   if n < 3500:
     #       continue
     
        try:
            data = data.float().to(device)
            batch_size = data.size(1)
            time = data.size(0)
            print('time', time, 'data', data.shape)
            target = torch.zeros(time, args.latent_modulation_dim)
        except:
            continue
        
        for t in range(1):

            slice = data
        model_wrapper.model.reset_modulations()   #set modulation vector to 0
      


        _ = inner_adapt_test_scale(model_wrapper=model_wrapper, data=slice, step_size=args.inner_lr,
                                   num_steps=args.inner_steps_test, first_order=True,
                                   sample_type=args.sample_type, scale_type='grad')

        if set == 'train' and n < 10:
                with torch.no_grad():
                    pred = model_wrapper().clamp(0, 1)
                    print('pred', pred.shape)
                    slice = data[1,...]
                    predslice = pred[1,...]
                if n < 10:# and 'video' not in args.dataset:
                    # Convert to PIL image
                    to_pil = transforms.ToPILImage()
                    image = to_pil(slice.squeeze())
                    image.save(f"./imgs/{n}_input.png")
                    image = to_pil(predslice.squeeze())
                    image.save(f"./imgs/{n}_recon.png")

       
        m=model_wrapper.model.modulations.detach().cpu()
          #  v = model_wrapper.model.vdim.detach().cpu()
            
        target=m  
       
        datapoint = {
                'modulations': target.detach().cpu(),
                'label': label.detach().cpu(),
                'v': v.detach().cpu()
            }
        sdir = args.save_dir + f'/{set}/' + f'datapoint_{(n * batch_size)}.pt'
        torch.save(datapoint, sdir)
        print('saved' , sdir, n)
    return



def fit_nfs_autoregressive(args, model_wrapper, dataloader, double = False, set=None):

    model_wrapper.model.eval()
    model_wrapper.coord_init()


    for n, data in enumerate(dataloader):
        model_wrapper.model.reset_modulations()
        print('mods0',model_wrapper.model.modulations.shape )
        if args.v_dim > 0:
         model_wrapper.model.reset_vdim()
        if double == False:
            data, label = data
            if set == 'BUV_OOD':
                label = 2
            print('label', label)
            if  data.isnan().any():
                        print('got nan')
                        continue
            if args.dimension == 'swap':
                data = torch.permute(data, (1, 0, 2, 3))

       
        num_steps =math.floor(data.shape[0]/args.num_frames)  #should be 15
        print('num_steps', num_steps)
        data = data.float().to(device)
        batch_size = data.size(1)
        time = data.size(0)
        print('time', time, 'data', data.shape)
        target = torch.zeros(time, 512)
     
      
        for k in range(int(num_steps)):
           # slice = data[:,t,...][None,...]
            if k ==0:
                model_wrapper.model.reset_modulations()
                
                chunk = data[0:args.num_frames,...]
                print('mods',model_wrapper.model.modulations.shape )
              
                if args.v_dim >0:
                    model_wrapper.model.reset_vdim()
                    _ = inner_adapt_test_scale_v2(model_wrapper=model_wrapper, data=chunk, step_size=args.inner_lr,
                                            num_steps=args.inner_steps_test, first_order=True,
                                            sample_type=args.sample_type, scale_type='grad')
                    v = model_wrapper.model.vdim.detach().cpu()
                else:
                    _ = inner_adapt_test_scale(model_wrapper=model_wrapper, data=chunk, step_size=args.inner_lr,
                                        num_steps=args.inner_steps_test, first_order=True,
                                        sample_type=args.sample_type, scale_type='grad')

                target[0:args.num_frames,...] = model_wrapper.model.modulations.detach().cpu()
                
            else:
                model_wrapper.model.reset_modulations()
                chunk = data[k*args.num_frames: (k+1)*args.num_frames ,...]
                
                _ = inner_adapt_test_scale(model_wrapper=model_wrapper, data=chunk, step_size=args.inner_lr,
                                        num_steps=args.inner_steps_test, first_order=True,
                                        sample_type=args.sample_type, scale_type='grad')
                target[k*args.num_frames: (k+1)*args.num_frames ,...] = model_wrapper.model.modulations.detach().cpu()
        
        print('target', target.shape)
        if set == 'train' and n < 10:
                with torch.no_grad():
                    model_wrapper.model.modulations = target[40:46,...].to(device)
                    pred = model_wrapper().clamp(0, 1)
                    print('pred', pred.shape)
                    print('data', data.shape)
                    slice = data[40,...]
                    predslice = pred[0,...]
          
                    # Convert to PIL image
                    to_pil = transforms.ToPILImage()
                    image = to_pil(slice.squeeze())
                    image.save(f"./imgs_concat/{n}_input.png")
                    image = to_pil(predslice.squeeze())
                    image.save(f"./imgs_concat/{n}_recon.png")

       
     
            
        
        print('label', label)
        if args.v_dim >0:
            datapoint = {
                'modulations': target.detach().cpu(),
                'label': label,
                'v': v.detach().cpu()
                
            }
        else: 
            datapoint = {
                'modulations': target.detach().cpu(),
                'label': label,
                'v': 0
                
            }
        sdir = args.save_dir + f'/{set}/' + f'datapoint_{(n * batch_size)}.pt'
        torch.save(datapoint, sdir)
        print('saved' , sdir, n)
    return
