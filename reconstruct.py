import torch
from torch.utils.data import DataLoader
from data import dataset_echonet
import os
import numpy as np
from common.args import parse_args
from common.utils import set_random_seed, load_model
from data.dataset import get_dataset
from eval.maml_full_eval import test_model, test_model_autoregressive, reconstruct_model_autoregressive
from models.inrs import  LatentModulatedSIRENLCB_ortho, LatentModulatedSIRENLCB_basic,LatentModulatedSIRENLCB_separate
from models.model_wrapper import ModelWrapper


def main(args):
    """
    Main function to call for running an evaluation procedure.
    :param args: parameters parsed from the command line.
    :return: Nothing.
    """

    """ Set a device to use """
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    args.device = device
    print('device 2', args.device)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}:", torch.cuda.get_device_name(i))
    else:
        print("No GPUs available to this job")

    os.system("nvidia-smi")

    args.data_size = (1, args.img_size, args.img_size)
    if args.dimension == '3d':
        args.data_size = (1, args.num_frames, args.img_size, args.img_size)


    """ Enable determinism """
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """ Define test dataset """




    if args.dataset == 'framewise_echo': 
        test_loader = torch.utils.data.DataLoader(dataset_echonet.Echo(split="test", period=1, length=None, start = 0 ), batch_size=args.test_batch_size)
        print('got test loader')
        train_loader = torch.utils.data.DataLoader(dataset_echonet.Echo(split="train", period=1, length=None, start = 0), batch_size=args.test_batch_size)
        val_loader = torch.utils.data.DataLoader(dataset_echonet.Echo(split="val", period=1, length=None, start = 0), batch_size=args.test_batch_size)

    else:
  
        train, val, test = get_dataset(args, all=True, double =False)
        train_loader = DataLoader(train, batch_size=args.test_batch_size, shuffle=False, num_workers=4, pin_memory=True,
                                drop_last=True)
        val_loader = DataLoader(val, batch_size=args.test_batch_size, shuffle=False, num_workers=4, pin_memory=True,
                                drop_last=True)
        test_loader = DataLoader(test, batch_size=args.test_batch_size, shuffle=False, num_workers=4, pin_memory=True,
                                drop_last=True)



    print('vdim', args.v_dim)
    """ Initialize model and optimizer """


    if args.model == 'siren':
        print('got CLB v3')
        w0s = np.linspace(args.w0, args.wK, args.num_layers)
        print('w0s', w0s)
        print('vdim', args.v_dim)
        if args.comment == 'separate':
            print('got separate model')
            model = LatentModulatedSIRENLCB_separate(
                in_size=args.in_size,
                out_size=args.out_size,
                min_hidden_size=256,
                max_hidden_size=256,
                progression_type='exponential',
                num_layers=args.num_layers,
                latent_modulation_dim=args.latent_modulation_dim,
                latent_v_dim= args.v_dim,
                w0s=w0s,
                modulate_shift=args.modulate_shift,
                modulate_scale=args.modulate_scale,
            ).to(device)
            model.modulations = torch.zeros(size=[args.num_frames, args.latent_modulation_dim], requires_grad=True).to(device)
            if args.v_dim ==0:
                    model.vdim = 0
            else:
                    model.vdim = torch.zeros(size=[1, args.v_dim], requires_grad=True).to(device)
           


        elif args.comment =='ortho':
                print('got ortho clb')
                model = LatentModulatedSIRENLCB_ortho(
                    in_size=args.in_size,
                    out_size=args.out_size,
                    min_hidden_size=256,
                    max_hidden_size=256,
                    progression_type='exponential',
                    num_layers=args.num_layers,
                    latent_modulation_dim=args.latent_modulation_dim,
                    latent_v_dim= args.v_dim,
                    w0s=w0s,
                    modulate_shift=args.modulate_shift,
                    modulate_scale=args.modulate_scale,
                ).to(device)
                model.modulations = torch.zeros(size=[args.num_frames, args.latent_modulation_dim], requires_grad=True).to(device)
                if args.v_dim ==0:
                        model.vdim = 0
                else:
                        model.vdim = torch.zeros(size=[1, args.v_dim], requires_grad=True).to(device)

        elif args.comment =='basic':
                print('got basic LRM-Functa model')
                model = LatentModulatedSIRENLCB_basic(
                    in_size=args.in_size,
                    out_size=args.out_size,
                    min_hidden_size=256,
                    max_hidden_size=256,
                    progression_type='exponential',
                    num_layers=args.num_layers,
                    latent_modulation_dim=args.latent_modulation_dim,
                    latent_v_dim= args.v_dim,
                    w0s=w0s,
                    modulate_shift=args.modulate_shift,
                    modulate_scale=args.modulate_scale,
                ).to(device)
                model.modulations = torch.zeros(size=[args.num_frames, args.latent_modulation_dim], requires_grad=True).to(device)
                if args.v_dim ==0:
                        model.vdim = 0
                else:
                        model.vdim = torch.zeros(size=[1, args.v_dim], requires_grad=True).to(device)



    print('modulations', model.modulations.shape)


    if not os.path.exists(args.save_dir):
        print(f'Create: {args.save_dir }nfset')
        os.mkdir(args.save_dir)
        os.mkdir(args.save_dir+'videos')
        os.mkdir(args.save_dir+'nfset')

    """ Define test function """
   
   
    model = ModelWrapper(args, model)
    load_model(args, model)
    print('loaded model')

    if not os.path.exists(args.save_dir + 'nfset/test'):
        os.mkdir(args.save_dir + 'nfset/test')
    reconstruct_model_autoregressive(args, model, test_loader, logger=None, set = 'test')
    print('recontructed test')


    if not os.path.exists(args.save_dir + 'nfset/train'):
                os.mkdir(args.save_dir + 'nfset/train')
    reconstruct_model_autoregressive(args, model, train_loader, logger=None, set = 'train')
    print('recontructed train')

 

    if not os.path.exists(args.save_dir + 'nfset/val'):
             os.mkdir(args.save_dir + 'nfset/val')
    reconstruct_model_autoregressive(args, model, val_loader, logger=None, set = 'val')
    print('recontructed val')



if __name__ == "__main__":
    args = parse_args()
    main(args)
