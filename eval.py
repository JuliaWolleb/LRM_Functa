import torch
from torch.utils.data import DataLoader
from data import dataset_echonet
from models.inrs_spatial import LatentModulatedSIREN_spatial, LatentModulatedSIREN_v2, LatentModulatedSIREN_v3, LatentModulatedSIREN_v4

from common.args import parse_args
from common.utils import set_random_seed, load_model
from data.dataset import get_dataset
from eval.maml_full_eval import test_model, test_model_autoregressive
from models.inrs import LatentModulatedSIREN, LatentModulatedSIREN_v2, LatentModulatedSIREN_v3
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
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device

    args.data_size = (1, args.img_size, args.img_size)
    if args.dimension == '3d':
        args.data_size = (1, args.num_frames, args.img_size, args.img_size)


    """ Enable determinism """
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """ Define test dataset """

    if args.dataset == 'echo_3d':
        print('got 3d dataset')
        test_loader = torch.utils.data.DataLoader(dataset_echonet.Echo(split="test", period=8, length=args.num_frames  ), batch_size=args.test_batch_size)

    elif args.dataset == 'framewise_echo': 
        test_loader = torch.utils.data.DataLoader(  dataset_echonet.Echo(split="test", period=1, length=120 ), batch_size=args.test_batch_size)
    else:
      test_set = get_dataset(args, only_test=True)
      test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=4, pin_memory=True,
                             drop_last=True)
    print('vdim', args.v_dim)

    """ Initialize model and optimizer """
    if args.mode == 'spatial':
        model = LatentModulatedSIREN_spatial(
            in_size=args.in_size,
            out_size=args.out_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            latent_modulation_dim=args.latent_modulation_dim,
            w0=args.w0,
            w0_increments=args.w0_increment,
            modulate_shift=args.modulate_shift,
            modulate_scale=args.modulate_scale,
            enable_skip_connections=args.enable_skip_connections,
            guidance = args.guidance
        ).to(device)
        print('got spatial siren')
        model.modulations = torch.zeros(size=[args.num_frames, 32, 16,16 ], requires_grad=True).to(device)
        print('modulations init', model.modulations.shape)
      

    elif args.v_dim==0:
        model = LatentModulatedSIREN(
            in_size=args.in_size,
            out_size=args.out_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            latent_modulation_dim=args.latent_modulation_dim,
            w0=args.w0,
            w0_increments=args.w0_increment,
            modulate_shift=args.modulate_shift,
            modulate_scale=args.modulate_scale,
            enable_skip_connections=args.enable_skip_connections,
        ).to(device)
        model.modulations = torch.zeros(size=[args.num_frames, args.latent_modulation_dim], requires_grad=True).to(device)
        print('got normal siren')
        if args.dimension == '3d':
            model.modulations = torch.zeros(size=[args.batch_size, args.latent_modulation_dim], requires_grad=True).to(device)
            print('got 3d modulations')


    elif args.model == 'siren2':
        print('got siren2')
        model = LatentModulatedSIREN_v3(
                in_size=args.in_size,
                out_size=args.out_size,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                latent_modulation_dim=args.latent_modulation_dim,
                v_dim = args.v_dim,
                w0=args.w0,
                w0_increments=args.w0_increment,
                modulate_shift=args.modulate_shift,
                modulate_scale=args.modulate_scale,
                enable_skip_connections=args.enable_skip_connections,
                mode=args.mode,
                num_frames=args.num_frames,
                guidance = args.guidance

            ).to(device)
        if args.v_dim ==0:
                model.vdim = 0
        else:
                model.vdim = torch.zeros(size=[1, args.v_dim], requires_grad=True).to(device)
        model.modulations = torch.zeros(size=[args.num_frames, args.latent_modulation_dim], requires_grad=True).to(device)
    


    else:
        model = LatentModulatedSIREN_v2(
            in_size=args.in_size,
            out_size=args.out_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            latent_modulation_dim=args.latent_modulation_dim,
            v_dim = args.v_dim,
            w0=args.w0,
            w0_increments=args.w0_increment,
            modulate_shift=args.modulate_shift,
            modulate_scale=args.modulate_scale,
            enable_skip_connections=args.enable_skip_connections,
            mode = args.mode,
        ).to(device)
        if args.v_dim ==0:
            model.vdim = 0
        else:
            model.vdim = torch.zeros(size=[1, args.v_dim], requires_grad=True).to(device)
        model.modulations = torch.zeros(size=[args.num_frames, args.latent_modulation_dim], requires_grad=True).to(device)

    
    
    model = ModelWrapper(args, model)
    load_model(args, model)

    """ Define test function """
    if args.dimension == '3d':
        test_model(args, model, test_loader, logger=None)
    else:
        print('test model autoregressive')
        test_model_autoregressive(args, model, test_loader, logger=None)


if __name__ == "__main__":
    args = parse_args()
    main(args)
