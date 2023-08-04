import os
import jsonlines
import json
import numpy as np
import random
import pandas as pd
from random import sample

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

try:
    from .common import pad_tensors, gen_seq_masks, flipcoin, relation_synonyms
    from .gtlabel_dataset import GTLabelDataset, ROTATE_ANGLES
except:
    from common import pad_tensors, gen_seq_masks, flipcoin, relation_synonyms
    from gtlabel_dataset import GTLabelDataset, ROTATE_ANGLES

class GTLabelPcdDataset(GTLabelDataset):
    def __init__(
        self, scan_id_file, anno_file, scan_dir, category_file,
        cat2vec_file=None, keep_background=False, random_rotate=False,
        num_points=1024, max_txt_len=50, max_obj_len=80,
        in_memory=False, gt_scan_dir=None, iou_replace_gt=0,
        anchors_mode="none", max_anchors=7, predict_lang_anchors=False, target_aug_percentage=None, is_train=None,
        distractor_aux_loss_flag=False, data_csv_pth="", train_data_percent=1.0, is_nr3d=True, repeat=1
    ):
        super().__init__(
            scan_id_file, anno_file, scan_dir, category_file,
            cat2vec_file=cat2vec_file, keep_background=keep_background,
            random_rotate=random_rotate, 
            max_txt_len=max_txt_len, max_obj_len=max_obj_len,
            gt_scan_dir=gt_scan_dir, iou_replace_gt=iou_replace_gt,
            anchors_mode=anchors_mode, max_anchors=max_anchors, predict_lang_anchors=predict_lang_anchors, 
            target_aug_percentage=target_aug_percentage, is_train=is_train,
            distractor_aux_loss_flag=distractor_aux_loss_flag, data_csv_pth=data_csv_pth, train_data_percent=train_data_percent,
            is_nr3d=is_nr3d, repeat=repeat
        )
        self.num_points = num_points
        self.in_memory = in_memory

        if self.in_memory:
            for scan_id in self.scan_ids:
                self.get_scan_pcd_data(scan_id)

    def get_scan_pcd_data(self, scan_id):
        if self.in_memory and 'pcds' in self.scans[scan_id]:
            return self.scans[scan_id]['pcds']
        
        pcd_data = torch.load(
            os.path.join(self.scan_dir, 'pcd_with_global_alignment', '%s.pth'%scan_id)
        )
        points, colors = pcd_data[0], pcd_data[1]  # (68469, 3) $ (68469, 3)
        colors = colors / 127.5 - 1
        pcds = np.concatenate([points, colors], 1)
        instance_labels = pcd_data[-1]
        obj_pcds = []
        for i in range(instance_labels.max() + 1):
            mask = instance_labels == i     # time consuming
            obj_pcds.append(pcds[mask])
        if self.in_memory:
            self.scans[scan_id]['pcds'] = obj_pcds
        return obj_pcds

    def get_scan_gt_pcd_data(self, scan_id):
        if self.in_memory and 'gt_pcds' in self.scans[scan_id]:
            return self.scans[scan_id]['gt_pcds']
        
        pcd_data = torch.load(
            os.path.join(self.gt_scan_dir, 'pcd_with_global_alignment', '%s.pth'%scan_id)
        )
        points, colors = pcd_data[0], pcd_data[1]
        colors = colors / 127.5 - 1
        pcds = np.concatenate([points, colors], 1)
        instance_labels = pcd_data[-1]
        obj_pcds = []
        for i in range(instance_labels.max() + 1):
            mask = instance_labels == i     # time consuming
            obj_pcds.append(pcds[mask])
        if self.in_memory:
            self.scans[scan_id]['gt_pcds'] = obj_pcds
        return obj_pcds

    def _get_obj_inputs(self, obj_pcds, obj_colors, obj_labels, obj_ids, tgt_obj_idx, theta=None, anchor_objs_idx=None):
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
                    if klabel == tgt_obj_type:
                        selected_obj_idxs.append(kobj)
                    else:
                        remained_obj_idxs.append(kobj)
            if len(selected_obj_idxs) < self.max_obj_len:
                random.shuffle(remained_obj_idxs)
                selected_obj_idxs += remained_obj_idxs[:self.max_obj_len - len(selected_obj_idxs)]

            obj_pcds = [obj_pcds[i] for i in selected_obj_idxs]
            obj_labels = [obj_labels[i] for i in selected_obj_idxs]
            obj_colors = [obj_colors[i] for i in selected_obj_idxs]
            obj_ids = [obj_ids[i] for i in selected_obj_idxs]
            tgt_obj_idx = 0

        # Rotate obj:
        if (theta is not None) and (theta != 0):
            rot_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ], dtype=np.float32)
        else:
            rot_matrix = None
        obj_fts, obj_locs = [], []
        for obj_pcd in obj_pcds:
            # obj_center = obj_pcd[:, :3].mean(0)
            # obj_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
            # obj_locs.append(np.concatenate([obj_center, obj_size], 0))
            if rot_matrix is not None:
                obj_pcd[:, :3] = np.matmul(obj_pcd[:, :3], rot_matrix.transpose())
            obj_center = obj_pcd[:, :3].mean(0)
            obj_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
            obj_locs.append(np.concatenate([obj_center, obj_size], 0))
            # obj_locs[-1][:3] = obj_center
            # sample points
            pcd_idxs = np.random.choice(len(obj_pcd), size=self.num_points, replace=(len(obj_pcd) < self.num_points))
            obj_pcd = obj_pcd[pcd_idxs]
            # normalize
            obj_pcd[:, :3] = obj_pcd[:, :3] - obj_pcd[:, :3].mean(0)
            max_dist = np.max(np.sqrt(np.sum(obj_pcd[:, :3]**2, 1)))
            if max_dist < 1e-6: # take care of tiny point-clouds, i.e., padding
                max_dist = 1
            obj_pcd[:, :3] = obj_pcd[:, :3] / max_dist
            obj_fts.append(obj_pcd)

        obj_fts = np.stack(obj_fts, 0)  # [L, 1024, 6]
        obj_locs = np.array(obj_locs)  # (L, 6)
        obj_colors = np.array(obj_colors)  # (L, 3, 4)

        # CoT:
        if anchor_objs_idx is not None:
            # Add dummy object (no_obj):
            if self.is_nr3d:
                obj_labels.append('no_obj')
                obj_fts = np.append(obj_fts, np.ones((1,1024,6))*-1, axis=0)
                obj_locs = np.append(obj_locs, np.ones((1,6))*-1, axis=0)
                obj_colors = np.append(obj_colors, np.ones((1,3,4))*-1, axis=0)
                obj_ids.append('-1')
                self.padded_value = len(obj_labels)-1  # options are: -100 or len(obj_labels)-1
                #self.padded_value = -100  # options are: -100 or len(obj_labels)-1
            else:
                self.padded_value = -100

            for i in range(len(anchor_objs_idx)):
                if anchor_objs_idx[i] == -1:
                    anchor_objs_idx[i] = self.padded_value
            # Trim additional anchors
            if len(anchor_objs_idx) > self.max_anchors:
                anchor_objs_idx = anchor_objs_idx[:self.max_anchors]
            # Pad it to the self.max_anchors
            elif len(anchor_objs_idx) < self.max_anchors:
                anchor_objs_idx = anchor_objs_idx + [self.padded_value]*(self.max_anchors - len(anchor_objs_idx))
            # TODO: Eslam should remove this. This is for debuging only
            # anchor_objs_idx = [-100]*self.max_anchors
            
            anchor_objs_idx = np.array(anchor_objs_idx)
            
        return obj_fts, obj_locs, obj_colors, obj_labels, obj_ids, tgt_obj_idx, anchor_objs_idx

    def __getitem__(self, idx):
        item = self.data[idx]
        scan_id = item['scan_id']
        txt_tokens = torch.LongTensor(item['enc_tokens'][:self.max_txt_len])
        tgt_obj_idx = item['target_id']
        tgt_obj_type = item['instance_type']
        if self.anchors_mode != 'none':
            if self.is_nr3d:
                if self.anchors_ids_type == "pseudoWneg" or self.anchors_ids_type == "pseudoWneg_old":
                    anchor_objs_idx = self._convertstringlist_to_list(item['anchor_ids'])
                elif self.anchors_ids_type == "pseudoWOneg":
                    anchor_objs_idx = self._convertstringlist_to_list(item['ours_with_neg_ids'])
                elif self.anchors_ids_type == "ourPathGTids":
                    anchor_objs_idx = self._convertstringlist_to_list(item['our_gt_id'])
                elif self.anchors_ids_type == "GT":
                    anchor_objs_idx = self._convertstringlist_to_list(item['true_gt_id'])
            else:
                anchor_objs_idx = item['anchor_ids']
        else:
            anchor_objs_idx = None

        txt_tokens = torch.LongTensor(item['enc_tokens'][:self.max_txt_len])
        txt_lens = len(txt_tokens)

        if self.gt_scan_dir is None or item['max_iou'] > self.iou_replace_gt:
            obj_pcds = self.get_scan_pcd_data(scan_id)
            obj_labels = self.scans[scan_id]['inst_labels']
            obj_gmm_colors = self.scans[scan_id]['inst_colors']
            # print("obj_pcds = ", obj_pcds[0].shape)  # (2550, 6)
            # print("obj_gmm_colors = ", obj_gmm_colors[0].shape)  # (3, 4)
        else:
            tgt_obj_idx = item['gt_target_id']
            obj_pcds = self.get_scan_gt_pcd_data(scan_id)
            obj_labels = self.scans[scan_id]['gt_inst_labels']
            obj_gmm_colors = self.scans[scan_id]['gt_inst_colors']
        obj_ids = [str(x) for x in range(len(obj_labels))]

        # Remove background:
        if not self.keep_background:
            selected_obj_idxs = [i for i, obj_label in enumerate(obj_labels) if obj_label not in ['wall', 'floor', 'ceiling']]
            tgt_obj_idx = selected_obj_idxs.index(tgt_obj_idx)
            obj_labels = [obj_labels[i] for i in selected_obj_idxs]
            obj_pcds = [obj_pcds[i] for i in selected_obj_idxs]
            obj_gmm_colors = [obj_gmm_colors[i] for i in selected_obj_idxs]
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

        aug_obj_fts, aug_obj_locs, aug_obj_gmm_colors, aug_obj_labels, \
            aug_obj_ids, aug_tgt_obj_idx, aug_anchor_objs_idx = self._get_obj_inputs(
                obj_pcds, obj_gmm_colors, obj_labels, obj_ids, tgt_obj_idx,
                theta=theta, anchor_objs_idx=anchor_objs_idx
            )

        anchor_objs_classes = None
        if aug_anchor_objs_idx is not None:
            anchor_objs_type = []
            for i in range(len(aug_anchor_objs_idx)):
                if aug_anchor_objs_idx[i] == self.padded_value:
                    anchor_objs_type.append('no_obj')
                else:
                    anchor_objs_type.append(aug_obj_labels[aug_anchor_objs_idx[i]])
            anchor_objs_classes = [self.cat2int[anchor_type] for anchor_type in anchor_objs_type]
            # Extend number of classes to fixed number as this couldn't be changed like boxes:
            max_num_anchors_lang = 7 if self.is_nr3d else 2
            anchor_objs_classes += [self.cat2int['no_obj'] for _ in range(max_num_anchors_lang - len(anchor_objs_classes))]

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

        aug_obj_fts = torch.from_numpy(aug_obj_fts)
        aug_obj_locs = torch.from_numpy(aug_obj_locs)
        aug_obj_gmm_colors = torch.from_numpy(aug_obj_gmm_colors)
        aug_obj_classes = torch.LongTensor([self.cat2int[x] for x in aug_obj_labels])
        
        if self.cat2vec is None:
            aug_obj_gt_fts = aug_obj_classes
        else:
            aug_obj_gt_fts = torch.FloatTensor([self.cat2vec[x] for x in aug_obj_labels])

        # Pad them to the maximum obj & txt length:
        #txt_tokens = torch.cat((txt_tokens, torch.zeros(self.max_txt_len-len(txt_tokens))), dim=0).long()
        if self.anchors_mode != 'none':
            max_obj_len = self.max_obj_len + 1  # to include "no_obj"
        aug_obj_gt_fts = torch.cat((aug_obj_gt_fts, torch.zeros((max_obj_len-len(aug_obj_gt_fts), aug_obj_gt_fts.shape[1]))), dim=0)
        aug_obj_fts = torch.cat((aug_obj_fts, torch.zeros((max_obj_len-len(aug_obj_fts), aug_obj_fts.shape[1], aug_obj_fts.shape[2]))), dim=0)
        aug_obj_locs = torch.cat((aug_obj_locs, torch.zeros((max_obj_len-len(aug_obj_locs), aug_obj_locs.shape[1]))), dim=0)
        aug_obj_gmm_colors = torch.cat((aug_obj_gmm_colors, torch.zeros((max_obj_len-len(aug_obj_gmm_colors), aug_obj_gmm_colors.shape[1], aug_obj_gmm_colors.shape[2]))), dim=0)
        aug_obj_gt_fts = torch.cat((aug_obj_gt_fts, torch.zeros((max_obj_len-len(aug_obj_gt_fts), aug_obj_gt_fts.shape[1]))), dim=0)
        aug_obj_classes = torch.cat((aug_obj_classes, torch.ones(max_obj_len-len(aug_obj_classes))*-100), dim=0).long()
        aug_obj_ids = aug_obj_ids + ['-100']*(max_obj_len-len(aug_obj_ids))
        distractor_mask = torch.cat((distractor_mask, torch.zeros(max_obj_len-len(distractor_mask))), dim=0)
        

        outs = {
            'item_ids': item['item_id'],
            'scan_ids': scan_id,
            'txt_ids': txt_tokens,
            'txt_lens': txt_lens,
            'obj_gt_fts': aug_obj_gt_fts,
            'obj_fts': aug_obj_fts,
            'obj_locs': aug_obj_locs,
            'obj_colors': aug_obj_gmm_colors,
            'obj_lens': len(aug_obj_fts),
            'obj_classes': aug_obj_classes, 
            'tgt_obj_idxs': aug_tgt_obj_idx,
            'tgt_obj_classes': self.cat2int[tgt_obj_type],
            'obj_ids': aug_obj_ids,
            'anchor_objs_idxs': aug_anchor_objs_idx,
            'anchor_objs_classes': anchor_objs_classes,
            'distractor_mask': distractor_mask,
        }

        """
        print("item_ids = ", item['item_id'])  # No need
        print("scan_id = ", scan_id)  # No need
        print("txt_tokens = ", txt_tokens.shape)
        print("txt_tokens = ", txt_tokens)
        print("txt_lens = ", txt_lens)  # No need
        print("aug_obj_gt_fts = ", aug_obj_gt_fts.shape)
        print("aug_obj_fts = ", aug_obj_fts.shape)
        print("aug_obj_locs = ", aug_obj_locs.shape)
        print("aug_obj_gmm_colors = ", aug_obj_gmm_colors.shape)
        print("aug_obj_classes = ", aug_obj_classes.shape)
        print("aug_obj_ids = ", len(aug_obj_ids))
        print("distractor_mask = ", distractor_mask.shape)
        """
        return outs

def gtlabelpcd_collate_fn(data):
    outs = {}
    for key in data[0].keys():
        outs[key] = [x[key] for x in data]
        
    outs['txt_ids'] = pad_sequence(outs['txt_ids'], batch_first=True)
    outs['txt_lens'] = torch.LongTensor(outs['txt_lens'])
    outs['txt_masks'] = gen_seq_masks(outs['txt_lens'])

    outs['obj_gt_fts'] = pad_tensors(outs['obj_gt_fts'], lens=outs['obj_lens'])
    outs['obj_fts'] = pad_tensors(outs['obj_fts'], lens=outs['obj_lens'], pad_ori_data=True).float()
    outs['obj_locs'] = pad_tensors(outs['obj_locs'], lens=outs['obj_lens'], pad=0).float()
    outs['obj_colors'] = pad_tensors(outs['obj_colors'], lens=outs['obj_lens'], pad=0).float()
    outs['obj_lens'] = torch.LongTensor(outs['obj_lens'])
    outs['obj_masks'] = gen_seq_masks(outs['obj_lens'])

    outs['obj_classes'] = pad_sequence(
        outs['obj_classes'], batch_first=True, padding_value=-100
    )
    outs['tgt_obj_idxs'] = torch.LongTensor(outs['tgt_obj_idxs'])
    outs['tgt_obj_classes'] = torch.LongTensor(outs['tgt_obj_classes'])
    
    if outs['anchor_objs_classes'][0] is not None:
        outs['anchor_objs_classes'] = torch.LongTensor(outs['anchor_objs_classes'])

    if outs['anchor_objs_idxs'][0] is not None:
        outs['anchor_objs_idxs'] = pad_sequence(outs['anchor_objs_idxs'], batch_first=True, padding_value=-100)

    if outs['distractor_mask'][0] is not None:
        outs['distractor_mask'] = pad_sequence(outs['distractor_mask'], batch_first=True, padding_value=0).float()
    return outs

