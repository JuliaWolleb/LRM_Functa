import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from common.utils import set_random_seed, load_model
from data import dataset_echonet
from common.args import parse_args
from common.utils import set_random_seed, Logger, InfiniteSampler
from data.dataset import get_dataset
from models.inrs import  LatentModulatedSIREN_ortho, LatentModulatedSIREN_separate,   LatentModulatedSIREN_basic

from models.model_wrapper import ModelWrapper
from train.trainer import trainer
from train.maml_boot import train_step
from eval.maml_scale import test_model
import numpy as np


def main(args):
    """
    Main function to call for running a training procedure.
    :param args: parameters parsed from the command line.
    :return: Nothing.
    """

    """ Set a device to use """
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    args.data_size = (1,args.img_size, args.img_size)
    if args.dimension == '3d':
        args.data_size = (1, args.num_frames, args.img_size, args.img_size)


    """ Enable determinism """
   # set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    print('dataset', args.dataset)
    """ Define dataset """
    if args.dataset == 'framewise_echo' or args.dataset == 'echo_3d':
        print('got 3d dataset')
        infinite_sampler = InfiniteSampler(dataset_echonet.Echo(split="train", period=10, length=args.num_frames  ), rank=0, num_replicas=1, shuffle=True, seed=args.seed)
 
        train_loader = torch.utils.data.DataLoader(dataset_echonet.Echo(split="train", period=10, length=args.num_frames  ), batch_size=args.batch_size, sampler = infinite_sampler, num_workers=4, prefetch_factor=2)
        val_loader = torch.utils.data.DataLoader(dataset_echonet.Echo(split="val",  period=10, length=args.num_frames  ), batch_size=args.test_batch_size)
        test_loader = torch.utils.data.DataLoader(dataset_echonet.Echo(split="test", period=10, length=args.num_frames  ), batch_size=args.test_batch_size)
       
    else:
        
        """ Define dataloader """
        train_set, val_set, test_set = get_dataset(args, all = True)
      
        print('train_set', len(train_set))

        infinite_sampler = InfiniteSampler(train_set, rank=0, num_replicas=1, shuffle=True)

        train_loader = DataLoader(train_set, sampler=infinite_sampler, batch_size=args.batch_size, num_workers=4,
                                prefetch_factor=2, drop_last=True)
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=True, num_workers=4 , drop_last=True)
       
    """ Initialize model and optimizer """



    if args.model == 'siren':
        
        w0s = np.linspace(args.w0, args.wK, args.num_layers)
        
        if args.comment == 'separate':
            print('got MedFuncta or VidFuncta model')


            model = LatentModulatedSIREN_separate(
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
                guidance = args.guidance
            ).to(device)
            model.modulations = torch.zeros(size=[args.num_frames, args.latent_modulation_dim], requires_grad=True).to(device)
            if args.v_dim ==0:
                    model.vdim = 0
            else:
                    model.vdim = torch.zeros(size=[1, args.v_dim], requires_grad=True).to(device)
        
        
        elif args.comment =='ortho':
                print('got orthogonal LRM-Functa Model')
                model = LatentModulatedSIREN_ortho(
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
                print('got basic LRM-Functa')
                model = LatentModulatedSIREN_basic(
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

  

    model = ModelWrapper(args, model)
    load_model(args, model)
    print('loaded model')

    meta_optimizer = optim.AdamW(params=model.parameters(), lr=args.meta_lr)


    """ Initialize a global lr-scheduler (optional - but recommended) """
    scheduler = None
    if args.lr_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(meta_optimizer, eta_min=1e-7, T_max=args.max_iter)


    """ Define training and test functions """
    train_function = train_step
    test_function = test_model

    """ Define logger """
    fname = (f'{args.dataset}_model{args.model}_numlayers{args.num_layers}_option{args.option}_guidance{args.guidance}_adversarial{args.adversarial}_lam{args.lam}_omega{args.w0}_inner{args.inner_steps}_mode{args.mode}_innerlr{args.inner_lr}_meatlr{args.meta_lr}_latentsize{args.latent_modulation_dim}_gamma{args.data_ratio}_hiddensize{args.hidden_size}_vdim{args.v_dim}_frames{args.num_frames}_comment{args.comment}'
             f'{args.config.split("/")[-1].split(".yaml")[0]}')
    logger = Logger(fname, ask=args.resume_path is None, rank=args.gpu_id)
    logger.log(args)
    logger.log(model)

    """ Perform training """
    print('start trainer')
    trainer(args, train_function, test_function, model, meta_optimizer, train_loader, test_loader, logger, scheduler)

    """ Close logger """
    logger.close_writer()


if __name__ == "__main__":
    args = parse_args()
    print('args', args)
    main(args)
