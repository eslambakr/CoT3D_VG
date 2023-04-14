"""
Handle arguments for train/test scripts.

The MIT License (MIT)
Originally created at 5/25/20, for Python 3.x
Copyright (c) 2020 P Achlioptas (pachlioptas@gmail.com) & Ahmed (@gmail.com)

-scannet-file /home/e/scannet_dataset/scannet/scan_4_nr3d/keep_all_points_00_view_no_global_scan_alignment_saveJPG.pkl -referit3D-file /home/e/scannet_dataset/scannet/nr3d.csv --log-dir ../log/referit3d_r18_32_loadimgs --n-workers 8 --batch-size 16 -load-imgs True --img-encoder True --object-encoder r18 -load-dense False
"""

import argparse
import json
import pprint
import os.path as osp
from datetime import datetime
from argparse import ArgumentParser
from ..utils import str2bool, create_dir


def parse_arguments(notebook_options=None):
    """Parse the arguments for the training (or test) execution of a ReferIt3D net.
    :param notebook_options: (list) e.g., ['--max-distractors', '100'] to give/parse arguments from inside a jupyter notebook.
    :return:
    """
    parser = argparse.ArgumentParser(description='ReferIt3D Nets + Ablations')

    #
    # Non-optional arguments
    #
    parser.add_argument('-scannet-file', type=str, required=True, help='pkl file containing the data of Scannet'
                                                                       ' as generated by running XXX')
    parser.add_argument('-referit3D-file', type=str, required=True)
    parser.add_argument('-load-dense', type=str2bool, required=True, default=False)
    parser.add_argument('-load-imgs', type=str2bool, required=True, default=False)
    parser.add_argument('--img-encoder', type=str2bool, required=True, default=False)
    parser.add_argument('--imgsize', default=32, type=int, help='Size of projected image')
    parser.add_argument('--train-vis-enc-only', default=False, type=str2bool, help='train visual encoder only')
    parser.add_argument('--cocoon', default=False, type=str2bool,
                        help='train visual encoder using cocoon setup')
    parser.add_argument('--twoStreams', default=False, type=str2bool, help='train using raw PC & 2d Images')
    parser.add_argument('--sceneCocoonPath', default=None, type=str,
                        help='path to the 2d cocoon images for the whole scene')
    parser.add_argument('--twoTrans', default=False, type=str2bool, help='Use 2 trans. (vision Trans + Lang Trans')
    parser.add_argument('--sharetwoTrans', default=False, type=str2bool, help='Share VisLang trans between 2D & 3D')
    parser.add_argument('--softtripleloss', default=False, type=str2bool, help='Use softtripleLoss instead of softmax '
                                                                               'for cls&ref losses for 2D&3D')
    parser.add_argument('--tripleloss', default=False, type=str2bool, help='Use tripleLoss to mitigate '
                                                                           'the distractor effect.')
    parser.add_argument('--contrastiveloss', default=False, type=str2bool, help='Use contrastiveloss to mitigate the'
                                                                                ' distractor effect.')
    parser.add_argument('--eval-path', type=str, help='Path for best model to evaluate on it')
    parser.add_argument('--train-scanRefer', default=False, type=str2bool, help='to load scanRefer data')
    parser.add_argument('--feat2ddim', default=2048, type=int, help='2D feature dim')

    #
    # I/O file-related arguments
    #
    parser.add_argument('--log-dir', type=str, help='where to save training-progress, model, etc')
    parser.add_argument('--resume-path', type=str, help='model-path to resume')
    parser.add_argument('--config-file', type=str, default=None, help='config file')
    parser.add_argument('--pretrained-path', type=str, default=None,
                        help='if set a pretrained weights will be loaded for visual encoder part')
    parser.add_argument('--freeze_backbone', type=str2bool, required=False, default=False)

    #
    # Dataset-oriented arguments
    #
    parser.add_argument('--max-distractors', type=int, default=51,
                        help='Maximum number of distracting objects to be drawn from a scan.')
    parser.add_argument('--max-seq-len', type=int, default=24,
                        help='utterances with more tokens than this they will be ignored.')
    parser.add_argument('--points-per-object', type=int, default=1024,
                        help='points sampled to make a point-cloud per object of a scan.')
    parser.add_argument('--unit-sphere-norm', type=str2bool, default=True,
                        help="Normalize each point-cloud to be in a unit sphere.")
    parser.add_argument('--mentions-target-class-only', type=str2bool, default=True,
                        help='If True, drop references that do not explicitly mention the target-class.')
    parser.add_argument('--min-word-freq', type=int, default=3)
    parser.add_argument('--max-test-objects', type=int, default=88)
    parser.add_argument('--context_info_2d_cached_file', type=str, help='path to pretrained 2d context information')

    #
    # Training arguments
    #
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate'])
    parser.add_argument('--max-train-epochs', type=int, default=100, help='number of training epochs. [default: 100]')
    parser.add_argument('--n-workers', type=int, default=-1,
                        help='number of data loading workers [default: -1 is all cores available -1.]')
    parser.add_argument('--random-seed', type=int, default=2020,
                        help='Control pseudo-randomness (net-wise, point-cloud sampling etc.) fostering reproducibility.')
    parser.add_argument('--init-lr', type=float, default=0.0005, help='learning rate for training.')
    parser.add_argument('--patience', type=int, default=10, help='if test-acc does not improve for patience consecutive'
                                                                 'epoch, stop training.')
    parser.add_argument("--warmup", action="store_true", default=True, help="if lr linear warmup.")

    #
    # Model arguments
    #
    parser.add_argument('--model', type=str, default='mmt_referIt3DNet', choices=['referIt3DNet',
                                                                                  'directObj2Lang',
                                                                                  'referIt3DNetAttentive',
                                                                                  'mmt_referIt3DNet'])
    parser.add_argument('--object-latent-dim', type=int, default=768)
    parser.add_argument('--language-latent-dim', type=int, default=768)
    parser.add_argument('--mmt-latent-dim', type=int, default=768)
    parser.add_argument('--word-embedding-dim', type=int, default=64)
    parser.add_argument('--graph-out-dim', type=int, default=128)
    parser.add_argument('--dgcnn-intermediate-feat-dim', nargs='+', type=int, default=[128, 128, 128, 128])

    parser.add_argument('--object-encoder', type=str, default='pnet_pp', choices=['pnet_pp', 'pnet',
                                                                                  'r50', 'r18', 'convnext',
                                                                                  'convnext_p++', 'clip_p++'])
    parser.add_argument('--language-fusion', type=str, default='both', choices=['before', 'after', 'both'])
    parser.add_argument('--word-dropout', type=float, default=0.1)
    parser.add_argument('--knn', type=int, default=7, help='For DGCNN number of neighbors')
    parser.add_argument('--lang-cls-alpha', type=float, default=0.5, help='if > 0 a loss for guessing the target via '
                                                                          'language only is added.')
    parser.add_argument('--obj-cls-alpha', type=float, default=0.5, help='if > 0 a loss for guessing for each segmented'
                                                                         ' object its class type is added.')
    parser.add_argument("--transformer", action="store_true", default=True, help="transformer mmt fusion module.")
    parser.add_argument('--context_obj', type=str, default=None, help="context object; rand, closest, farthest.")
    parser.add_argument('--feat2d', type=str, default="clsvecROI", help="ROI/clsvec/clsvecROI/ROIclspredGeo")
    parser.add_argument('--context_2d', type=str, default=None,
                        help="how to use 2D context; None, aligned or unaligned.")
    parser.add_argument('--mmt_mask', type=str, default=None, help="if apply certain mmt mask.")
    parser.add_argument('--geo3d', type=str2bool, default=False, help="If set the 3D geometry info will be added"
                                                                      " to the 3D branch")
    parser.add_argument('--clspred3d', type=str2bool, default=False, help="If set the 3D predicted class info "
                                                                          "will be added to the 3D branch.")
    parser.add_argument('--imgaug', type=str2bool, default=False, help="If set the imgaug will be activated.")
    parser.add_argument('--camaug', type=str2bool, default=False, help="If set the camaug will be activated.")
    parser.add_argument('--mask_2d_in_testing', type=str2bool, default=False,
                        help="If set this means that we are mimicking SAT, where 2D will be discarded during testing.")

    # Distributed Training:
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    #
    # Misc arguments
    #
    parser.add_argument('--batch-size', type=int, default=32, help='batch size per gpu. [default: 32]')
    parser.add_argument('--save-args', type=str2bool, default=True, help='save arguments in a json.txt')
    parser.add_argument('--experiment-tag', type=str, default=None, help='will be used to name a subdir '
                                                                         'for log-dir if given')
    parser.add_argument('--cluster-pid', type=str, default=None)

    #
    # "Joint" (Sr3d+Nr3D) training
    #
    parser.add_argument('--augment-with-sr3d', type=str, default=None,
                        help='csv with sr3d data to augment training data'
                             'of args.referit3D-file')
    parser.add_argument('--vocab-file', type=str, default=None, help='optional, .pkl file for vocabulary (useful when '
                                                                     'working with multiple dataset and single model.')
    parser.add_argument('--fine-tune', type=str2bool, default=False,
                        help='use if you train with dataset x and then you '
                             'continue training with another dataset')
    parser.add_argument('--s-vs-n-weight', type=float, default=None, help='importance weight of sr3d vs nr3d '
                                                                          'examples [use less than 1]')

    # Parse args
    if notebook_options is not None:
        args = parser.parse_args(notebook_options)
    else:
        args = parser.parse_args()

    if not args.resume_path and not args.log_dir:
        raise ValueError

    if args.config_file is not None:
        with open(args.config_file, 'r') as fin:
            configs_dict = json.load(fin)
            apply_configs(args, configs_dict)

    if args.mask_2d_in_testing:
        args.twoTrans = False
    # Create logging related folders and arguments
    if args.log_dir:
        timestamp = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")

        if args.experiment_tag:
            args.log_dir = osp.join(args.log_dir, args.experiment_tag, timestamp)
        else:
            args.log_dir = osp.join(args.log_dir, timestamp)

        args.checkpoint_dir = create_dir(osp.join(args.log_dir, 'checkpoints'))
        args.tensorboard_dir = create_dir(osp.join(args.log_dir, 'tb_logs'))

    if args.resume_path and not args.log_dir:  # resume and continue training in previous log-dir.
        checkpoint_dir = osp.split(args.resume_path)[0]  # '/xxx/yyy/log_dir/checkpoints/model.pth'
        args.checkpoint_dir = checkpoint_dir
        args.log_dir = osp.split(checkpoint_dir)[0]
        args.tensorboard_dir = osp.join(args.log_dir, 'tb_logs')

    # Print them nicely
    args_string = pprint.pformat(vars(args))
    print(args_string)

    if args.save_args:
        out = osp.join(args.log_dir, 'config.json.txt')
        with open(out, 'w') as f_out:
            json.dump(vars(args), f_out, indent=4, sort_keys=True)

    return args


def read_saved_args(config_file, override_args=None, verbose=True):
    """
    :param config_file:
    :param override_args: dict e.g., {'gpu': '0'}
    :param verbose:
    :return:
    """
    parser = ArgumentParser()
    args = parser.parse_args([])
    with open(config_file, 'r') as f_in:
        args.__dict__ = json.load(f_in)

    if override_args is not None:
        for key, val in override_args.items():
            args.__setattr__(key, val)

    if verbose:
        args_string = pprint.pformat(vars(args))
        print(args_string)

    return args


def apply_configs(args, config_dict):
    for k, v in config_dict.items():
        setattr(args, k, v)
