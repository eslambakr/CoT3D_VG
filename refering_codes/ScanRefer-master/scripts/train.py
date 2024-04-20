import os
import sys
import json
import h5py
import argparse
import importlib
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import csv
import ast

from torch.utils.data import DataLoader
from datetime import datetime
from copy import deepcopy

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.dataset import ScannetReferenceDataset
from lib.solver import Solver
from lib.config import CONF
from models.refnet import RefNet


def str2bool(v):
    """
    Boolean values for argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))

# constants
DC = ScannetDatasetConfig()

def get_dataloader(args, scanrefer, all_scene_list, split, config, augment):
    dataset = ScannetReferenceDataset(
        scanrefer=scanrefer[split], 
        scanrefer_all_scene=all_scene_list, 
        split=split, 
        num_points=args.num_points, 
        use_height=(not args.no_height),
        use_color=args.use_color, 
        use_normal=args.use_normal, 
        use_multiview=args.use_multiview,
        args=args
    )
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    if split == "val":
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    return dataset, dataloader

def get_model(args):
    # initiate model
    input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(not args.no_height)
    model = RefNet(
        num_class=DC.num_class,
        num_heading_bin=DC.num_heading_bin,
        num_size_cluster=DC.num_size_cluster,
        mean_size_arr=DC.mean_size_arr,
        input_feature_dim=input_channels,
        num_proposal=args.num_proposals,
        use_lang_classifier=(not args.no_lang_cls),
        use_bidir=args.use_bidir,
        no_reference=args.no_reference,
        anchors=args.anchors, cot_type=args.cot_type, 
        predict_lang_anchors=args.predict_lang_anchors, max_num_anchors=args.max_num_anchors, feedcotpath=args.feedcotpath
    )

    # trainable model
    if args.use_pretrained:
        # load model
        print("loading pretrained VoteNet...")
        pretrained_model = RefNet(
            num_class=DC.num_class,
            num_heading_bin=DC.num_heading_bin,
            num_size_cluster=DC.num_size_cluster,
            mean_size_arr=DC.mean_size_arr,
            num_proposal=args.num_proposals,
            input_feature_dim=input_channels,
            use_bidir=args.use_bidir,
            no_reference=True
        )

        pretrained_path = os.path.join(CONF.PATH.OUTPUT, args.use_pretrained, "model_last.pth")
        pretrained_model.load_state_dict(torch.load(pretrained_path), strict=False)

        # mount
        model.backbone_net = pretrained_model.backbone_net
        model.vgen = pretrained_model.vgen
        model.proposal = pretrained_model.proposal

        if args.no_detection:
            # freeze pointnet++ backbone
            for param in model.backbone_net.parameters():
                param.requires_grad = False

            # freeze voting
            for param in model.vgen.parameters():
                param.requires_grad = False
            
            # freeze detector
            for param in model.proposal.parameters():
                param.requires_grad = False
    
    # to CUDA
    model = model.cuda()

    return model

def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))

    return num_params

def get_solver(args, dataloader):
    model = get_model(args)
    if args.anchors == "cot":
        param_list = []
        for name, param in model.named_parameters():
            if "object_language_clf" in name:
                param_list.append({'params': param, 'lr': args.lr * args.cot_trans_lr_scale})
            else:
                param_list.append({'params': param, 'lr': args.lr})
        optimizer = optim.Adam(param_list, lr=args.lr, weight_decay=args.wd)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if args.use_checkpoint:
        print("loading checkpoint {}...".format(args.use_checkpoint))
        stamp = args.use_checkpoint
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        checkpoint = torch.load(os.path.join(CONF.PATH.OUTPUT, args.use_checkpoint, "checkpoint.tar"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args.tag: stamp += "_"+args.tag.upper()
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        os.makedirs(root, exist_ok=True)

    # scheduler parameters for training solely the detection pipeline
    LR_DECAY_STEP = [80, 120, 160] if args.no_reference else None
    LR_DECAY_RATE = 0.1 if args.no_reference else None
    BN_DECAY_STEP = 20 if args.no_reference else None
    BN_DECAY_RATE = 0.5 if args.no_reference else None

    solver = Solver(
        model=model, 
        config=DC, 
        dataloader=dataloader, 
        optimizer=optimizer, 
        stamp=stamp, 
        val_step=args.val_step,
        detection=not args.no_detection,
        reference=not args.no_reference, 
        use_lang_classifier=not args.no_lang_cls,
        lr_decay_step=LR_DECAY_STEP,
        lr_decay_rate=LR_DECAY_RATE,
        bn_decay_step=BN_DECAY_STEP,
        bn_decay_rate=BN_DECAY_RATE,
        args=args
    )
    num_params = get_num_params(model)

    return solver, num_params, root

def save_info(args, root, num_params, train_dataset, val_dataset):
    info = {}
    for key, value in vars(args).items():
        info[key] = value
    
    info["num_train"] = len(train_dataset)
    info["num_val"] = len(val_dataset)
    info["num_train_scenes"] = len(train_dataset.scene_list)
    info["num_val_scenes"] = len(val_dataset.scene_list)
    info["num_params"] = num_params

    with open(os.path.join(root, "info.json"), "w") as f:
        json.dump(info, f, indent=4)

def get_scannet_scene_list(split):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])

    return scene_list


def merge_cot_data(org_data, cot_data_dict):
    merged_list = []
    for j in tqdm(range(len(org_data))):
        found_flag = False
        org_uni_id = org_data[j]['scene_id'] + str(org_data[j]['object_id']) + str(org_data[j]['ann_id'])  # "<scene_id>-<object_id>_<ann_id>"
        for i in range(len(cot_data_dict)):
            cot_uni_id = cot_data_dict[i]['stimulus_id'] + str(cot_data_dict[i]['target_id']) + str(cot_data_dict[i]['ann_id'])  # "<scene_id>-<object_id>_<ann_id>"
            if cot_uni_id == org_uni_id:
                merged_dict = org_data[j].copy()
                merged_dict.update({'path': cot_data_dict[i]['path'], 'anchor_ids': cot_data_dict[i]['anchor_ids']})
                merged_list.append(merged_dict)
                found_flag = True
                break
        if not found_flag:
            merged_dict = org_data[j].copy()
            merged_dict.update({'path': [merged_dict['object_name']], 'anchor_ids': [merged_dict['object_id']]})
            merged_list.append(merged_dict)
    print("--- The number of merged CoT items: ", len(merged_list))

    return merged_list


def clean_path_and_anchorsid(cot_data_list_of_dict):
    for i in tqdm(range(len(cot_data_list_of_dict))):
        cot_data_list_of_dict[i]['path'] = ast.literal_eval(cot_data_list_of_dict[i]['path'])
        cot_data_list_of_dict[i]['anchor_ids'] = ast.literal_eval(cot_data_list_of_dict[i]['anchor_ids'])
        cot_data_list_of_dict[i]['anchor_ids'] = [int(anchor_id) for anchor_id in cot_data_list_of_dict[i]['anchor_ids']]
        cot_data_list_of_dict[i]["object_id"] = str(cot_data_list_of_dict[i]["object_id"])
        cot_data_list_of_dict[i]["ann_id"] = str(cot_data_list_of_dict[i]["ann_id"])


def save_listofdicts_to_csv(list_of_dicts, saving_dir):
    keys = list_of_dicts[0].keys()

    with open(saving_dir, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(list_of_dicts)


def get_scanrefer(scanrefer_train, scanrefer_val, num_scenes):
    if args.no_reference:
        train_scene_list = get_scannet_scene_list("train")
        new_scanrefer_train = []
        for scene_id in train_scene_list:
            data = deepcopy(SCANREFER_TRAIN[0])
            data["scene_id"] = scene_id
            new_scanrefer_train.append(data)

        val_scene_list = get_scannet_scene_list("val")
        new_scanrefer_val = []
        for scene_id in val_scene_list:
            data = deepcopy(SCANREFER_VAL[0])
            data["scene_id"] = scene_id
            new_scanrefer_val.append(data)
    else:
        # get initial scene list
        train_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_train])))
        val_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_val])))
        if num_scenes == -1: 
            num_scenes = len(train_scene_list)
        else:
            assert len(train_scene_list) >= num_scenes
        
        # slice train_scene_list
        train_scene_list = train_scene_list[:num_scenes]

        # filter data in chosen scenes
        new_scanrefer_train = []
        for data in scanrefer_train:
            if data["scene_id"] in train_scene_list:
                new_scanrefer_train.append(data)

        new_scanrefer_val = scanrefer_val

    # all scanrefer scene
    all_scene_list = train_scene_list + val_scene_list

    if args.newdataset:
        load_preprocessed_data = False
        if load_preprocessed_data:
            new_scanrefer_val = pd.read_csv("./data/merged_val_scanrefer_cot.csv").to_dict('records')
            clean_path_and_anchorsid(new_scanrefer_val)
            new_scanrefer_train = pd.read_csv("./data/merged_train_scanrefer_cot.csv").to_dict('records')
            clean_path_and_anchorsid(new_scanrefer_train)
            print("--- The number of merged CoT items for training: ", len(new_scanrefer_train))
            print("--- The number of merged CoT items for val: ", len(new_scanrefer_val))
            dummy_counter = 0
            for i in range(len(new_scanrefer_train)):
                if len(new_scanrefer_train[i]['anchor_ids']) == 6:
                    dummy_counter += 1
            print(100*dummy_counter/len(new_scanrefer_train))
            dummy_counter = 0
            for i in range(len(new_scanrefer_val)):
                if len(new_scanrefer_val[i]['anchor_ids']) == 6:
                    dummy_counter += 1
            print(100*dummy_counter/len(new_scanrefer_val))
        else:
            cot_data_dict = pd.read_csv("./data/scanrefer_cot_sortFixed.csv").to_dict('records')
            new_scanrefer_train = merge_cot_data(org_data=new_scanrefer_train, cot_data_dict=cot_data_dict)
            save_listofdicts_to_csv(new_scanrefer_train, saving_dir="./data/merged_train_scanrefer_cot.csv")
            new_scanrefer_val = merge_cot_data(org_data=new_scanrefer_val, cot_data_dict=cot_data_dict)
            save_listofdicts_to_csv(new_scanrefer_val, saving_dir="./data/merged_val_scanrefer_cot.csv")
            exit()
    
    # Truncate the data based on the desired percentage:
    new_scanrefer_train = new_scanrefer_train[:int(len(new_scanrefer_train)*args.train_data_percent)]

    print("train on {} samples and val on {} samples".format(len(new_scanrefer_train), len(new_scanrefer_val)))

    return new_scanrefer_train, new_scanrefer_val, all_scene_list

def train(args):
    # init training dataset
    print("preparing data...")
    scanrefer_train, scanrefer_val, all_scene_list = get_scanrefer(SCANREFER_TRAIN, SCANREFER_VAL, args.num_scenes)
    scanrefer = {
        "train": scanrefer_train,
        "val": scanrefer_val
    }

    # dataloader
    train_dataset, train_dataloader = get_dataloader(args, scanrefer, all_scene_list, "train", DC, True)
    val_dataset, val_dataloader = get_dataloader(args, scanrefer, all_scene_list, "val", DC, False)
    dataloader = {
        "train": train_dataloader,
        "val": val_dataloader
    }

    print("initializing...")
    solver, num_params, root = get_solver(args, dataloader)

    print("Start training...\n")
    save_info(args, root, num_params, train_dataset, val_dataset)
    solver(args.epoch, args.verbose)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, help="tag for the training, e.g. cuda_wl", default="")
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    parser.add_argument("--batch_size", type=int, help="batch size", default=14)
    parser.add_argument("--epoch", type=int, help="number of epochs", default=50)
    parser.add_argument("--verbose", type=int, help="iterations of showing verbose", default=10)
    parser.add_argument("--val_step", type=int, help="iterations of validating", default=5000)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--wd", type=float, help="weight decay", default=1e-5)
    parser.add_argument("--num_points", type=int, default=40000, help="Point Number [default: 40000]")
    parser.add_argument("--num_proposals", type=int, default=256, help="Proposal number [default: 256]")
    parser.add_argument("--num_scenes", type=int, default=-1, help="Number of scenes [default: -1]")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--no_height", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--no_augment", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--no_lang_cls", action="store_true", help="Do NOT use language classifier.")
    parser.add_argument("--no_detection", action="store_true", help="Do NOT train the detection module.")
    parser.add_argument("--no_reference", action="store_true", help="Do NOT train the localization module.")
    parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_normal", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_multiview", action="store_true", help="Use multiview images.")
    parser.add_argument("--use_bidir", action="store_true", help="Use bi-directional GRU.")
    parser.add_argument("--use_pretrained", type=str, help="Specify the folder name containing the pretrained detection module.")
    parser.add_argument("--use_checkpoint", type=str, help="Specify the checkpoint root", default="")
    # CoT arguments:
    parser.add_argument('--anchors', type=str, default='none', choices=['parallel', 'cot', 'none'],
                        help='train using all anchors in parallel, or sequential/CoT, or with no anchors')
    parser.add_argument('--cot_type', type=str, default='cross', choices=['cross', 'causal', 'self_cons'],
                        help='cross:  transformer decoder will refine all inputs based even on the future predections\
                              causal: transformer decoder will refine inputs based on causal manner(only previous predictions)')
    parser.add_argument('--predict_lang_anchors', type=str2bool, default=False)
    parser.add_argument('--max_num_anchors', type=int, default=1,  help="maximum number of allowed anchors")
    parser.add_argument('--train_data_percent', type=float, default=1.0, 
                        help="sample from the training data given this ratio, for data effeciency expirements.")
    parser.add_argument('--distractor_aux_loss_flag', type=str2bool, default=False,  help="Add head to predict which objs are distractors")
    parser.add_argument('--newdataset', type=str2bool, default=False,  help="Use the new dataset.")
    parser.add_argument('--feedcotpath', type=str2bool, default=False,  help="Feed the CoT path as an input to the network.")
    parser.add_argument('--cot_trans_lr_scale', type=float, default=1,  help="CoT Transformer LR scaler")

    args = parser.parse_args()

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    train(args)
    
