#!/usr/bin/env python
# coding: utf-8

import torch
import tqdm
import time
import warnings
import os.path as osp
import torch.nn as nn
from torch import optim
from termcolor import colored
# ## might be related to the memory issue https://github.com/referit3d/referit3d/issues/5
# ## A temp solution is to add at evlauation mode to avoid "received 0 items of ancdata" (uncomment next line in eval)
# torch.multiprocessing.set_sharing_strategy('file_system')

from referit3d.in_out.arguments import parse_arguments
from referit3d.in_out.neural_net_oriented import load_scan_related_data, load_referential_data
from referit3d.in_out.neural_net_oriented import compute_auxiliary_data, trim_scans_per_referit3d_data
from referit3d.in_out.pt_datasets.listening_dataset import make_data_loaders
from referit3d.utils import set_gpu_to_zero_position, create_logger, seed_training_code
from referit3d.utils.tf_visualizer import Visualizer
from referit3d.models.referit3d_net import instantiate_referit3d_net
from referit3d.models.referit3d_net_utils import single_epoch_train, evaluate_on_dataset
from referit3d.models.utils import load_state_dicts, save_state_dicts
from referit3d.analysis.deepnet_predictions import analyze_predictions
from referit3d.utils.scheduler import GradualWarmupScheduler

from referit3d.in_out.pt_datasets.utils import create_sr3d_classes_2_idx


def log_train_test_information():
    """Helper logging function.
    Note uses "global" variables defined below.
    """
    logger.info('Epoch:{}'.format(epoch))
    for phase in ['train', 'test']:
        if phase == 'train':
            meters = train_meters
        else:
            meters = test_meters

        info = '{}: Total-Loss {:.4f}, Listening-Acc {:.4f}'.format(phase,
                                                                    meters[phase + '_total_loss'],
                                                                    meters[phase + '_referential_acc'])

        if args.anchors == 'cot':
            info += ', Listening-Acc-AUX: {:.4f}'.format(meters[phase + '_referential_acc_aux_tgt'])
            info += ', Listening-Acc-AUX-Anchor: {:.4f}'.format(meters[phase + '_referential_acc-aux_anchor1'])
            
        if args.obj_cls_alpha > 0:
            info += ', Object-Clf-Acc: {:.4f}'.format(meters[phase + '_object_cls_acc'])

        if args.lang_cls_alpha > 0:
            info += ', Text-Clf-Acc: {:.4f}'.format(meters[phase + '_txt_cls_acc'])

        logger.info(info)
        logger.info('{}: Epoch-time {:.3f}'.format(phase, timings[phase]))
    logger.info('Best so far {:.3f} (@epoch {})'.format(best_test_acc, best_test_epoch))
        
if __name__ == '__main__':

    # Parse arguments
    args = parse_arguments()
    if args.context_2d!='unaligned':
        args.mmt_mask = None
        print('not in unaligned mode, set mmt-mask to None!\n')

    # Prepare GPU environment
    set_gpu_to_zero_position(args.gpu)  # Pnet++ seems to work only at "gpu:0"
    device = torch.device('cuda')
    seed_training_code(args.random_seed,strict=True)

    # Read the scan related information
    # all_scans_in_dict, scans_split, class_to_idx = load_scan_related_data(args.scannet_file)
    all_scans_in_dict, scans_split, class_to_idx = load_scan_related_data(args.scannet_file,
                                                                           add_no_obj=args.anchors != 'none' or args.predict_lang_anchors)
    is_nr = True if 'nr' in args.referit3D_file else False
    if is_nr:
        class_to_idx = create_sr3d_classes_2_idx(json_pth="referit3d/data/mappings/scannet_instance_class_to_semantic_class.json")
    # Read the linguistic data of ReferIt3D
    referit_data = load_referential_data(args, args.referit3D_file, scans_split)

    # Prepare data & compute auxiliary meta-information.
    all_scans_in_dict = trim_scans_per_referit3d_data(referit_data, all_scans_in_dict)
    mean_rgb, vocab = compute_auxiliary_data(referit_data, all_scans_in_dict, args)
    data_loaders = make_data_loaders(args, referit_data, vocab, class_to_idx, all_scans_in_dict, mean_rgb, seed=args.random_seed)   ## seed loader workers
    # Losses:
    criteria = dict()

    # Referential, "find the object in the scan" loss
    if args.s_vs_n_weight is not None:  # TODO - move to a better place
        assert args.augment_with_sr3d is not None
        ce = nn.CrossEntropyLoss(reduction='none').to(device)
        s_vs_n_weight = args.s_vs_n_weight


        def weighted_ce(logits, batch):
            loss_per_example = ce(logits, batch['target_pos'])
            sr3d_mask = ~batch['is_nr3d']
            weights = torch.ones(loss_per_example.shape).to(device)
            weights[sr3d_mask] = s_vs_n_weight
            loss_per_example = loss_per_example * weights
            loss = loss_per_example.sum() / len(loss_per_example)
            return loss


        criteria['logits'] = weighted_ce
        criteria['logit_aux'] = weighted_ce
    else:
        criteria['logits'] = nn.CrossEntropyLoss().to(device)
        criteria['logit_aux'] = nn.CrossEntropyLoss().to(device)
    criteria['logits_nondec'] = nn.CrossEntropyLoss(reduction='none').to(device)
    # Prepare the Listener
    n_classes = len(class_to_idx) - 1  # -1 to ignore the <pad> class -1 for no object
    pad_idx = class_to_idx['pad']
    # Object-type classification
    if args.obj_cls_alpha > 0:
        reduction = 'mean' if args.s_vs_n_weight is None else 'none'
        criteria['class_logits'] = nn.CrossEntropyLoss(ignore_index=pad_idx,reduction=reduction).to(device)
    # Target-in-language guessing
    if args.lang_cls_alpha > 0:
        reduction = 'mean' if args.s_vs_n_weight is None else 'none'
        criteria['lang_logits'] = nn.CrossEntropyLoss(reduction=reduction).to(device)
        criteria['lang_logit_aux']= nn.CrossEntropyLoss(reduction=reduction).to(device)
    
    
    model = instantiate_referit3d_net(args, vocab, n_classes, class_to_idx).to(device)
    args.n_obj_classes = n_classes
    same_backbone_lr = False
    if same_backbone_lr:
        optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    else:
        backbone_name = []
        if args.transformer:
            backbone_name.append('text_bert.')  ## exclude text_bert_out_linear
            # backbone_name.append('object_encoder.')
            # backbone_name.append('cnt_object_encoder.')

        if args.anchors == 'cot':
            cot_names = ["object_language_clf.", "object_language_clf_2d.", "mmt."]
        else:
            cot_names = []

        backbone_param, cot_param, rest_param = [], [], []
        for kv in model.named_parameters():
            isbackbone = [int(key in kv[0]) for key in backbone_name]
            iscot = [int(key in kv[0]) for key in cot_names]
            if sum(isbackbone+[0]):
                backbone_param.append(kv[1])
            elif sum(iscot+[0]):
                cot_param.append(kv[1])
            else:
                rest_param.append(kv[1])
        optimizer = optim.Adam([{'params': rest_param},
                {'params': backbone_param, 'lr': args.init_lr/10.},
                {'params': cot_param, 'lr': args.init_lr/10.}], lr=args.init_lr)

        sum_backbone = sum([param.nelement() for param in backbone_param])
        sum_fusion = sum([param.nelement() for param in rest_param])
        sum_all = sum([param.nelement() for param in model.parameters()])
        print('backbone, fusion module parameters:', sum_backbone, sum_fusion, sum_all)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.65,
                                                              patience=5, verbose=True)
    if args.patience==args.max_train_epochs:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,50,60,70,80,90], gamma=0.65)    ## custom2
        if args.max_train_epochs==120: lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25,40,55,70,85,100], gamma=0.5)    ## custom3-120ep
    if args.warmup:
        scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=lr_scheduler)
        optimizer.zero_grad()   ## this zero gradient update is needed to avoid a warning message, issue #8.
        optimizer.step()

    start_training_epoch = 1
    best_test_acc = -1
    best_test_epoch = -1
    no_improvement = 0

    if args.resume_path:
        warnings.warn('Resuming assumes that the BEST per-val model is loaded!')
        # perhaps best_test_acc, best_test_epoch, best_test_epoch =  unpickle...
        loaded_epoch = load_state_dicts(args.resume_path, map_location=device, model=model)
        print('Loaded a model stopped at epoch: {}.'.format(loaded_epoch))
        if not args.fine_tune:
            print('Loaded a model that we do NOT plan to fine-tune.')
            load_state_dicts(args.resume_path, optimizer=optimizer, lr_scheduler=lr_scheduler)
            start_training_epoch = loaded_epoch + 1
            best_test_epoch = loaded_epoch
            best_test_acc = lr_scheduler.best
            print('Loaded model had {} test-accuracy in the corresponding dataset used when trained.'.format(
                best_test_acc))
        else:
            print('Parameters that do not allow gradients to be back-propped:')
            ft_everything = True
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    print(name)
                    exist = False
            if ft_everything:
                print('None, all wil be fine-tuned')
            # if you fine-tune the previous epochs/accuracy are irrelevant.
            dummy = args.max_train_epochs + 1 - start_training_epoch
            print('Ready to *fine-tune* the model for a max of {} epochs'.format(dummy))

    if args.pretrain_path:
        load_model = torch.load(args.pretrain_path)
        pretrained_dict = load_model['model']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        assert (len([k for k, v in pretrained_dict.items()])!=0)
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("=> loaded pretrain model at {}"
              .format(args.pretrain_path))
        if 'best' in load_model['lr_scheduler']:
            print('Loaded model had {} test-accuracy in the corresponding dataset used when trained.'.format(
                    load_model['lr_scheduler']['best']))
    # Training.
    if args.mode == 'train':
        train_vis = Visualizer(args.tensorboard_dir)
        logger = create_logger(args.log_dir)
        logger.info('Starting the training. Good luck!')
        eval_acc = 0.

        with tqdm.trange(start_training_epoch, args.max_train_epochs + 1, desc='epochs') as bar:
            timings = dict()
            for epoch in bar:
                if args.warmup: 
                    scheduler_warmup.step(epoch=epoch, metrics=eval_acc)    ## using the previous epoch's metrics
                print('lr:', epoch, optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'])
                logger.info('LR of param_groups[0] {}'.format(optimizer.param_groups[0]['lr']))
                logger.info('LR of param_groups[1] {}'.format(optimizer.param_groups[1]['lr']))

                # Train:
                tic = time.time()
                train_meters = single_epoch_train(model, data_loaders['train'], criteria, optimizer,
                                                  device, pad_idx, args=args, epoch=epoch)
                # import pdb; pdb.set_trace()
                toc = time.time()
                timings['train'] = (toc - tic) / 60

                # Evaluate:
                tic = time.time()
                test_meters = evaluate_on_dataset(model, data_loaders['test'], criteria, device, pad_idx, args=args)
                toc = time.time()
                timings['test'] = (toc - tic) / 60

                eval_acc = test_meters['test_referential_acc']
                if not args.warmup: 
                    lr_scheduler.step()

                if best_test_acc < eval_acc:
                    logger.info(colored('Test accuracy, improved @epoch {}'.format(epoch), 'green'))
                    best_test_acc = eval_acc
                    best_test_epoch = epoch

                    # Save the model (overwrite the best one)
                    save_state_dicts(osp.join(args.checkpoint_dir, 'best_model.pth'),
                                     epoch, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler)
                    no_improvement = 0
                else:
                    no_improvement += 1
                    logger.info(colored('Test accuracy, did not improve @epoch {}'.format(epoch), 'red'))

                log_train_test_information()
                train_meters.update(test_meters)
                train_vis.log_scalars({k: v for k, v in train_meters.items() if '_acc' in k}, step=epoch,
                                      main_tag='acc')
                train_vis.log_scalars({k: v for k, v in train_meters.items() if '_loss' in k},
                                      step=epoch, main_tag='loss')

                bar.refresh()

                if no_improvement == args.patience:
                    logger.warning(colored('Stopping the training @epoch-{} due to lack of progress in test-accuracy '
                                           'boost (patience hit {} epochs)'.format(epoch, args.patience),
                                           'red', attrs=['bold', 'underline']))
                    break

        with open(osp.join(args.checkpoint_dir, 'final_result.txt'), 'w') as f_out:
            msg = ('Best accuracy: {:.4f} (@epoch {})'.format(best_test_acc, best_test_epoch))
            f_out.write(msg)

        logger.info('Finished training successfully. Good job!')

    elif args.mode == 'evaluate':

        meters = evaluate_on_dataset(model, data_loaders['test'], criteria, device, pad_idx, args=args)
        print('Reference-Accuracy: {:.4f}'.format(meters['test_referential_acc']))
        print('Object-Clf-Accuracy: {:.4f}'.format(meters['test_object_cls_acc']))
        print('Text-Clf-Accuracy {:.4f}:'.format(meters['test_txt_cls_acc']))

        out_file = osp.join(args.checkpoint_dir, 'test_result.txt')
        res = analyze_predictions(model, data_loaders['test'].dataset, class_to_idx, pad_idx, device,
                                  args, out_file=out_file)
        print(res)
