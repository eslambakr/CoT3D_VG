import os
import jsonlines
import json
import numpy as np
import random
import collections
import copy
import pandas as pd
from random import sample

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

try:    
    from .common import pad_tensors, gen_seq_masks, flipcoin, relation_synonyms
except:
    from common import pad_tensors, gen_seq_masks, flipcoin, relation_synonyms


ROTATE_ANGLES = [0, np.pi/2, np.pi, np.pi*3/2]

class GTLabelDataset(Dataset):
    def __init__(
        self, scan_id_file, anno_file, scan_dir, category_file,
        cat2vec_file=None, keep_background=False, random_rotate=False, 
        max_txt_len=50, max_obj_len=80, gt_scan_dir=None, iou_replace_gt=0, 
        anchors_mode="none", max_anchors=7, predict_lang_anchors=False, target_aug_percentage=None, is_train=None,
        distractor_aux_loss_flag=False, data_csv_pth="", train_data_percent=1.0, is_nr3d=True, repeat=1
    ):
        super().__init__()

        split_scan_ids = set([x.strip() for x in open(scan_id_file, 'r')])

        self.scan_dir = scan_dir
        self.max_txt_len = max_txt_len
        self.max_obj_len = max_obj_len
        self.keep_background = keep_background
        self.random_rotate = random_rotate
        self.gt_scan_dir = gt_scan_dir
        self.iou_replace_gt = iou_replace_gt
        self.is_nr3d = is_nr3d
        # CoT
        self.anchors_mode = anchors_mode
        self.max_anchors = max_anchors
        self.train_data_percent = train_data_percent
        self.target_aug_percentage = target_aug_percentage
        self.distractor_aux_loss_flag = distractor_aux_loss_flag
        self.repeat = repeat

        if self.is_nr3d:
            self.nr3d_cot_dict = pd.read_csv(data_csv_pth).to_dict()
            self.stimulus_id_list = list(self.nr3d_cot_dict['stimulus_id'].values())
            cot_keys = ['path', 'anchor_ids', 'num_anchors']
            if self.target_aug_percentage:
                unique_rel = pd.read_csv("/home/abdelrem/3d_codes/CoT3D_VG/extract_anchors/nr3d_cot_unique_rel_anchor_data.csv").to_dict()
                self.stimulus_unique_rel_id_list = list(unique_rel['stimulus_id'].values())
                self.unique_rel_list = list(unique_rel['relation'].values())
                with open('/home/abdelrem/3d_codes/CoT3D_VG/extract_anchors/unique_rel_map_dict_opposite.json') as f:
                    self.opposite_dict = json.load(f)

        # Load samples info:
        self.scan_ids = set()
        self.data = []
        self.scan_to_item_idxs = collections.defaultdict(list)
        with jsonlines.open(anno_file, 'r') as f:
            for item in f:
                if item['scan_id'] in split_scan_ids:
                    if (len(item['tokens']) > 24) and (not item['item_id'].startswith('scanrefer')): continue
                    # if not is_explicitly_view_dependent(item['tokens']): continue
                    self.scan_ids.add(item['scan_id'])
                    self.scan_to_item_idxs[item['scan_id']].append(len(self.data))
                    if self.anchors_mode != 'none':
                        if self.is_nr3d:
                            index = self.stimulus_id_list.index(item['stimulus_id'])
                            for cot_k in cot_keys:
                                item[cot_k] = self.nr3d_cot_dict[cot_k][index]

                    self.data.append(item)

        # Repeat the data:
        self.data = self.data * self.repeat

        # Target_aug_percentage
        if self.target_aug_percentage and (self.anchors_mode != 'none'):
            # Create word->encoded tokens dict:
            self.enc_token_word_dict = {}
            for i in range(len(self.data)):
                for word in self.data[i]['tokens']:
                    self.enc_token_word_dict[word] = self.data[i]['enc_tokens'][self.data[i]['tokens'].index(word)]
            """
            # Create relation->encoded tokens dict:
            self.enc_token_rel_dict = {}
            for _, value in relation_synonyms.items():
                for rel in value:
                    for i in range(len(self.data)):
                        if rel in self.data[i]['tokens']:
                            self.enc_token_rel_dict[rel] = self.data[i]['enc_tokens'][self.data[i]['tokens'].index(rel)]
                            break
            """

        # Train_data_percent:
        self.data = self.data[:int(len(self.data)*self.train_data_percent)]

        self.scans = {}
        for scan_id in self.scan_ids:
            inst_labels = json.load(open(os.path.join(scan_dir, 'instance_id_to_name', '%s.json'%scan_id)))
            inst_locs = np.load(os.path.join(scan_dir, 'instance_id_to_loc', '%s.npy'%scan_id))
            inst_colors = json.load(open(os.path.join(scan_dir, 'instance_id_to_gmm_color', '%s.json'%scan_id)))
            inst_colors = [np.concatenate(
                [np.array(x['weights'])[:, None], np.array(x['means'])],
                axis=1
            ).astype(np.float32) for x in inst_colors]
            self.scans[scan_id] = {
                'inst_labels': inst_labels, # (n_obj, )
                'inst_locs': inst_locs,     # (n_obj, 6) center xyz, whl
                'inst_colors': inst_colors, # (n_obj, 3x4) cluster * (weight, mean rgb)
            }
        if self.gt_scan_dir is not None:
            for scan_id in self.scan_ids:
                inst_labels = json.load(open(os.path.join(gt_scan_dir, 'instance_id_to_name', '%s.json'%scan_id)))
                inst_locs = np.load(os.path.join(gt_scan_dir, 'instance_id_to_loc', '%s.npy'%scan_id))
                inst_colors = json.load(open(os.path.join(gt_scan_dir, 'instance_id_to_gmm_color', '%s.json'%scan_id)))
                inst_colors = [np.concatenate(
                    [np.array(x['weights'])[:, None], np.array(x['means'])],
                    axis=1
                ).astype(np.float32) for x in inst_colors]
                self.scans[scan_id].update({
                    'gt_inst_labels': inst_labels, # (n_obj, )
                    'gt_inst_locs': inst_locs,     # (n_obj, 6) center xyz, whl
                    'gt_inst_colors': inst_colors, # (n_obj, 3x4) cluster * (weight, mean rgb)
                })

        self.int2cat = json.load(open(category_file, 'r'))
        self.cat2int = {w: i for i, w in enumerate(self.int2cat)}
        if cat2vec_file is None:
            self.cat2vec = None
        else:
            self.cat2vec = json.load(open(cat2vec_file, 'r'))

        # Add no_obj
        if self.anchors_mode != 'none':
            self.cat2int['no_obj'] = len(self.cat2int)
            if self.cat2vec is not None:
                self.cat2vec['no_obj'] = [-1]*300
        
    def __len__(self):
        return len(self.data)

    def _convertstringlist_to_list(self, string_list):
        list_strings = string_list.strip('][').split(', ')
        list_int = []
        for s in list_strings:
            if (s != '') and (s != ""):
                list_int.append(int(s))
        return list_int
            
    def _path_parser(self, string_list):
        list_strings = string_list.strip('][').split(', ')
        list_string = []
        for s in list_strings:
            if (s != '') and (s != "") and (not ('*' in s)):
                list_string.append(s[1:-1])
        return list_string

    def _get_obj_inputs(self, obj_labels, obj_locs, obj_colors, obj_ids, tgt_obj_idx, theta=None, anchor_objs_idx=None):
        tgt_obj_type = obj_labels[tgt_obj_idx]
        if (self.max_obj_len is not None) and (len(obj_labels) > self.max_obj_len):
            selected_obj_idxs = [tgt_obj_idx]
            if anchor_objs_idx is not None:
                for i in range(len(anchor_objs_idx)):
                    if anchor_objs_idx[i] != -1:
                        selected_obj_idxs.append(anchor_objs_idx[i])
                anchor_objs_idx = list(range(1, len(selected_obj_idxs)))

            remained_obj_idxs = []
            for kobj, klabel in enumerate(obj_labels):
                if kobj != tgt_obj_idx:
                    if klabel == tgt_obj_type:  # Fill in the distractors
                        selected_obj_idxs.append(kobj)
                    else:
                        remained_obj_idxs.append(kobj)
            if len(selected_obj_idxs) < self.max_obj_len:
                random.shuffle(remained_obj_idxs)
                selected_obj_idxs += remained_obj_idxs[:self.max_obj_len - len(selected_obj_idxs)]

            obj_labels = [obj_labels[i] for i in selected_obj_idxs]
            obj_locs = [obj_locs[i] for i in selected_obj_idxs]
            obj_colors = [obj_colors[i] for i in selected_obj_idxs]
            obj_ids = [obj_ids[i] for i in selected_obj_idxs]
            tgt_obj_idx = 0

        obj_locs = np.array(obj_locs)
        obj_colors = np.array(obj_colors)

        # Rotate obj:
        if (theta is not None) and (theta != 0):
            rot_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ], dtype=np.float32)
            obj_locs[:, :3] = np.matmul(obj_locs[:, :3], rot_matrix.transpose())
            
        # CoT:
        if anchor_objs_idx is not None:
            # Add dummy object (no_obj):
            if self.is_nr3d:
                obj_labels.append('no_obj')
                obj_locs = np.append(obj_locs, np.ones((1,6))*-1, axis=0)
                obj_colors = np.append(obj_colors, np.ones((1,3,4))*-1, axis=0)
                obj_ids.append('-1')
                padded_value = len(obj_labels)-1  # options are: -100 or len(obj_labels)-1
                #padded_value = -100  # options are: -100 or len(obj_labels)-1

            for i in range(len(anchor_objs_idx)):
                if anchor_objs_idx[i] == -1:
                    anchor_objs_idx[i] = padded_value
            # Trim additional anchors
            if len(anchor_objs_idx) > self.max_anchors:
                anchor_objs_idx = anchor_objs_idx[:self.max_anchors]
            # Pad it to the self.max_anchors
            elif len(anchor_objs_idx) < self.max_anchors:
                anchor_objs_idx = anchor_objs_idx + [padded_value]*(self.max_anchors - len(anchor_objs_idx))
            # TODO: Eslam should remove this. This is for debuging only
            # anchor_objs_idx = [-100]*self.max_anchors
            
            anchor_objs_idx = np.array(anchor_objs_idx)

        return obj_labels, obj_locs, obj_colors, obj_ids, tgt_obj_idx, anchor_objs_idx
        
    def __getitem__(self, idx):
        item = self.data[idx]
        scan_id = item['scan_id']
        tgt_obj_idx = item['target_id']
        tgt_obj_type = item['instance_type']
        if self.anchors_mode != 'none':
            if self.is_nr3d:
                anchor_objs_idx = self._convertstringlist_to_list(item['anchor_ids'])
            else:
                anchor_objs_idx = item['anchor_ids']
        else:
            anchor_objs_idx = None

        # txt data
        txt_tokens = torch.LongTensor(item['enc_tokens'][:self.max_txt_len])
        txt_lens = len(txt_tokens)
        
        # obj data
        if self.gt_scan_dir is None or item['max_iou'] > self.iou_replace_gt:
            obj_labels = self.scans[scan_id]['inst_labels']
            obj_locs = self.scans[scan_id]['inst_locs']
            obj_colors = self.scans[scan_id]['inst_colors']
        else:
            tgt_obj_idx = item['gt_target_id']
            obj_labels = self.scans[scan_id]['gt_inst_labels']
            obj_locs = self.scans[scan_id]['gt_inst_locs']
            obj_colors = self.scans[scan_id]['gt_inst_colors']

        obj_ids = [str(x) for x in range(len(obj_labels))]

        # Remove background:
        if not self.keep_background:
            selected_obj_idxs = [i for i, obj_label in enumerate(obj_labels) if obj_label not in ['wall', 'floor', 'ceiling']]
            tgt_obj_idx = selected_obj_idxs.index(tgt_obj_idx)
            obj_labels = [obj_labels[i] for i in selected_obj_idxs]
            obj_locs = obj_locs[selected_obj_idxs]
            obj_colors = [obj_colors[i] for i in selected_obj_idxs]
            obj_ids = [obj_ids[i] for i in selected_obj_idxs]
            if anchor_objs_idx is not None:
                filtered_anchor_objs_idx = []
                for anchor_idx in anchor_objs_idx:
                    if anchor_idx in selected_obj_idxs:
                        filtered_anchor_objs_idx.append(selected_obj_idxs.index(anchor_idx))
                    else:
                        filtered_anchor_objs_idx.append(-1)
                anchor_objs_idx = filtered_anchor_objs_idx


        # Rotate:
        if self.random_rotate:
            theta_idx = np.random.randint(len(ROTATE_ANGLES))
            theta = ROTATE_ANGLES[theta_idx]
        else:
            theta = 0
        
        aug_obj_labels, aug_obj_locs, aug_obj_colors, aug_obj_ids, aug_tgt_obj_idx, aug_anchor_objs_idx = \
            self._get_obj_inputs(
                obj_labels, obj_locs, obj_colors, obj_ids, tgt_obj_idx,
                theta=theta, anchor_objs_idx=anchor_objs_idx
            )
        
        anchor_objs_classes = None
        if aug_anchor_objs_idx is not None:
            anchor_objs_type = []
            for i in range(len(aug_anchor_objs_idx)):
                if aug_anchor_objs_idx[i] == -100:
                    anchor_objs_type.append('no_obj')
                else:
                    anchor_objs_type.append(obj_labels[aug_anchor_objs_idx[i]])
            anchor_objs_classes = [self.cat2int[anchor_type] for anchor_type in anchor_objs_type]

        # Target_aug_percentage:
        if self.is_nr3d and self.target_aug_percentage and (self.anchors_mode != 'none'):
            path = self._path_parser(item['path'])
            if item['stimulus_id'] in self.stimulus_unique_rel_id_list:
                org_relation_idx = self.stimulus_unique_rel_id_list.index(item['stimulus_id'])
                org_relation = self.unique_rel_list[org_relation_idx]
                if flipcoin(self.target_aug_percentage) and (len(path)==2) and (type(org_relation)==str):
                    path.reverse()
                    # swap target idx with anchor idx:
                    temp = np.array([aug_tgt_obj_idx])
                    aug_tgt_obj_idx = aug_anchor_objs_idx[0]  # [0] as we are sure it is only one anchor
                    aug_anchor_objs_idx = temp
                    tgt_obj_type = anchor_objs_type[0]
                    relation = sample(relation_synonyms[self.opposite_dict[org_relation]], 1)[0]
                    tokens = [path[-1], relation, path[0]]
                    enc_tokens = []
                    for tok in tokens:
                        if " " in tok:
                            for word in tok.split(" "):
                                if word in self.enc_token_word_dict:  # Check if we have encoding for the word
                                    enc_tokens.append(self.enc_token_word_dict[word])
                        else:  # tok is one word
                            if tok in self.enc_token_word_dict:  # Check if we have encoding for the word
                                enc_tokens.append(self.enc_token_word_dict[tok])

                    txt_tokens = torch.LongTensor(enc_tokens)
                    txt_lens = len(txt_tokens)
        
        if aug_anchor_objs_idx is not None:
            aug_anchor_objs_idx = torch.from_numpy(aug_anchor_objs_idx)

        # Distractor Aux Loss:
        if self.distractor_aux_loss_flag:
            distractor_mask = np.zeros(len(aug_obj_labels))
            for kobj, klabel in enumerate(aug_obj_labels):
                if kobj != aug_tgt_obj_idx:
                    if klabel == tgt_obj_type:  # Fill in the distractors
                        distractor_mask[kobj] = 1.0
            distractor_mask = torch.from_numpy(distractor_mask)
        else:
            distractor_mask = None
        
        aug_obj_locs = torch.from_numpy(aug_obj_locs)
        aug_obj_colors = torch.from_numpy(aug_obj_colors)
        aug_obj_classes = torch.LongTensor([self.cat2int[x] for x in aug_obj_labels])
        if self.cat2vec is None:
            aug_obj_fts = aug_obj_classes
        else:
            aug_obj_fts = torch.FloatTensor([self.cat2vec[x] for x in aug_obj_labels])

                
        outs = {
            'item_ids': item['item_id'],
            'scan_ids': scan_id,
            'txt_ids': txt_tokens,
            'txt_lens': txt_lens,
            'obj_fts': aug_obj_fts,
            'obj_locs': aug_obj_locs,
            'obj_colors': aug_obj_colors,
            'obj_lens': len(aug_obj_fts),
            'obj_classes': aug_obj_classes, 
            'tgt_obj_idxs': aug_tgt_obj_idx,
            'anchor_objs_idxs': aug_anchor_objs_idx,
            'tgt_obj_classes': self.cat2int[tgt_obj_type],
            'anchor_objs_classes': anchor_objs_classes,
            'obj_ids': aug_obj_ids,
            'distractor_mask': distractor_mask,
        }

        return outs

def gtlabel_collate_fn(data):
    outs = {}
    for key in data[0].keys():
        outs[key] = [x[key] for x in data]

    outs['txt_ids'] = pad_sequence(outs['txt_ids'], batch_first=True)
    outs['txt_lens'] = torch.LongTensor(outs['txt_lens'])
    outs['txt_masks'] = gen_seq_masks(outs['txt_lens'])

    if len(outs['obj_fts'][0].size()) == 1:
        outs['obj_fts'] = pad_sequence(outs['obj_fts'], batch_first=True).float()
    else:
        outs['obj_fts'] = pad_tensors(outs['obj_fts'], lens=outs['obj_lens']).float()
    outs['obj_locs'] = pad_tensors(outs['obj_locs'], lens=outs['obj_lens'], pad=0).float()
    outs['obj_colors'] = pad_tensors(outs['obj_colors'], lens=outs['obj_lens'], pad=0).float()
    outs['obj_lens'] = torch.LongTensor(outs['obj_lens'])
    outs['obj_masks'] = gen_seq_masks(outs['obj_lens'])

    outs['obj_classes'] = pad_sequence(
        outs['obj_classes'], batch_first=True, padding_value=-100
    )
    outs['tgt_obj_idxs'] = torch.LongTensor(outs['tgt_obj_idxs'])
    outs['tgt_obj_classes'] = torch.LongTensor(outs['tgt_obj_classes'])
    if outs['anchor_objs_idxs'][0] is not None:
        outs['anchor_objs_idxs'] = pad_sequence(outs['anchor_objs_idxs'], batch_first=True, padding_value=-100)

    if outs['distractor_mask'][0] is not None:
        outs['distractor_mask'] = pad_sequence(outs['distractor_mask'], batch_first=True, padding_value=0).float()

    return outs

