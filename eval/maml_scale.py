import torch
import matplotlib.pyplot as plt
from common.utils import MetricLogger, psnr
from train.maml_boot import inner_adapt_test_scale, inner_adapt_test_scale_lora, inner_adapt_test_scale_v3, inner_adapt_test_scale_v2, inner_adapt_test_scale_time, inner_adapt_test_scale_vae


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_model(args, step, model_wrapper, test_loader, logger=None):
    metric_logger = MetricLogger(delimiter="  ")

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    model_wrapper.model.eval()
    model_wrapper.coord_init()

    for n, data in enumerate(test_loader):
        if n * args.test_batch_size > args.num_test_signals:
            break
        print('n', n)
    
        data, label,_,_ = data
           

  
        if args.dimension == 'swap':
          
            if data.isnan().any():
                    print('got nan')
                    continue
              
            data = data.float().to(device, non_blocking=True) 
            data = torch.permute(data, (1, 0, 2, 3))
            batch_size = data.size(0)

              
        elif args.dimension == '2d':
                if data.isnan().any():
                    print('got nan')
                    continue
              
                data = data.float().to(device, non_blocking=True) 
                batch_size = data.size(0)
           
      
        model_wrapper.model.reset_modulations(device)
        if args.v_dim >0:
            model_wrapper.model.reset_vdim()


        if n == 3:
            if args.mode == 'time':
                input = Data['vid']
            else:
                input = data
                print('input', input.shape)
        input = data
        if args.mode == 'time':
        

            loss_in_tt_gradscale = inner_adapt_test_scale_v2(model_wrapper=model_wrapper, data=data, step_size=args.inner_lr,
                                                      num_steps=args.inner_steps_test, first_order=True,
                                                      sample_type=args.sample_type, scale_type='grad')

       
        elif args.mode =='delta' or args.mode == 'try':

           
            print('gradscale v3')
            loss_in_tt_gradscale = inner_adapt_test_scale_v3(model_wrapper=model_wrapper, data=data, step_size=args.inner_lr,
                                                      num_steps=args.inner_steps_test, first_order=True,
                                                      sample_type=args.sample_type, scale_type='grad')


     

        elif args.comment == 'separate2':
            train_batch= {
                    'vid': data,
                    'label': label.float().to(device, non_blocking=True) 
                }
            loss_in_tt_gradscale = inner_adapt_test_scale_v2(model_wrapper=model_wrapper, data=train_batch, step_size=args.inner_lr,
                                                      num_steps=args.inner_steps_test, first_order=True,
                                                      sample_type=args.sample_type, scale_type='grad')
       
        elif args.v_dim > 0:

            print('gradscale v2')
            loss_in_tt_gradscale = inner_adapt_test_scale_v2(model_wrapper=model_wrapper, data=data, step_size=args.inner_lr,
                                                      num_steps=args.inner_steps_test, first_order=True,
                                                      sample_type=args.sample_type, scale_type='grad')

        else: 
            loss_in_tt_gradscale = inner_adapt_test_scale(model_wrapper=model_wrapper, data=data, step_size=args.inner_lr,
                                                      num_steps=args.inner_steps_test, first_order=True,
                                                      sample_type=args.sample_type, scale_type='grad')
        psnr_in_tt_gradscale = psnr(loss_in_tt_gradscale)
   

        """ Outer loss aggregation """
        with torch.no_grad():
            if args.mode == 'time':
                loss_out_tt_gradscale = model_wrapper(Data['vid'], Data['time'])
            elif args.comment == 'separate2':
                loss_out_tt_gradscale = model_wrapper(train_batch)
                print('got outer loss')
            
            else:
                loss_out_tt_gradscale = model_wrapper(data)
            psnr_out_tt_gradscale = psnr(loss_out_tt_gradscale)
           
            if n == 0:
                out = model_wrapper()

        metric_logger.meters['loss_inner_tt_gradscale'].update(loss_in_tt_gradscale.mean().item(), n=batch_size)
        metric_logger.meters['loss_outer_tt_gradscale'].update(loss_out_tt_gradscale.mean().item(), n=batch_size)
        metric_logger.meters['psnr_inner_tt_gradscale'].update(psnr_in_tt_gradscale.mean().item(), n=batch_size)
        metric_logger.meters['psnr_outer_tt_gradscale'].update(psnr_out_tt_gradscale.mean().item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    log_('*[EVAL Gradscale TestTime][LossInnerGSTT %.3f][LossOuterGSTT %.3f][PSNRInnerGSTT %.3f][PSNROuterGSTT %.3f]' %
         (metric_logger.loss_inner_tt_gradscale.global_avg, metric_logger.loss_outer_tt_gradscale.global_avg,
          metric_logger.psnr_inner_tt_gradscale.global_avg, metric_logger.psnr_outer_tt_gradscale.global_avg))

    logger.scalar_summary('eval/loss_inner_TT_gradscale', metric_logger.loss_inner_tt_gradscale.global_avg, step)
    logger.scalar_summary('eval/loss_outer_TT_gradscale', metric_logger.loss_outer_tt_gradscale.global_avg, step)
    logger.scalar_summary('eval/psnr_inner_TT_gradscale', metric_logger.psnr_inner_tt_gradscale.global_avg, step)
    logger.scalar_summary('eval/psnr_outer_TT_gradscale', metric_logger.psnr_outer_tt_gradscale.global_avg, step)
    out, modulations = model_wrapper()
    

    print('out', out.shape)
    logger.log_image('eval/img_in', input[:,:1,...], step)
    logger.log_image('eval/img_adapt_tt', out[:,:1,...], step)
    if args.sobel == True:
        logger.log_image('eval/img_sobel_in', input[:,1:,...], step)
        logger.log_image('eval/img_adapt_tt_sobel', out[:,1:,...], step)
        print('input 1', input[:,1:,...].max(), input[:,1:,...].min())
    input = input.detach().cpu()
    




    

    return metric_logger.psnr_outer_tt_gradscale.global_avg