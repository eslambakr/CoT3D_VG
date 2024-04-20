import torch
import argparse
from torch import nn
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
from . import DGCNN
from .utils import get_siamese_features, my_get_siamese_features, find_matching_indices, vector_gather
from ..in_out.vocabulary import Vocabulary
import math

try:
    from . import PointNetPP
except ImportError:
    PointNetPP = None

from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig
from transformers import BertTokenizer, BertModel, BertConfig
from referit3d.models import MLP
from referit3d.models.CoT.seq_seq_transformer import CoTTransformer
from referit3d.models.CoT.lstm_w_att import DecoderWithAttention
from referit3d.models.vil.cmt_module import CMT
from referit3d.models.referit3d_net import ReferIt3DNet_transformer


class ReferIt3DNetTeacherStudent(nn.Module):

    def __init__(self,
                 args,
                 n_obj_classes,
                 class_name_tokens,
                 ignore_index,
                 class_to_idx):

        super().__init__()

        self.args = args
        self.model_teacher = ReferIt3DNet_transformer(args, n_obj_classes, class_name_tokens, ignore_index=ignore_index,
                                                      class_to_idx=class_to_idx, dist_mode="teacher")
        for param in self.model_teacher.parameters():
            param.requires_grad = False
        self.model_student = ReferIt3DNet_transformer(args, n_obj_classes, class_name_tokens, ignore_index=ignore_index,
                                                      class_to_idx=class_to_idx, dist_mode="student")

    def forward(self, batch: dict, epoch=None):
        # Run the Teacher:
        _, _, _, _, _, teacher_multi_modal_trans_att_dict = self.model_teacher(batch, epoch, compute_loss=False,
                                                                               output_attentions=True, output_hidden_states=True)
        for k, v in teacher_multi_modal_trans_att_dict.items():
            if isinstance(v, list):
                teacher_multi_modal_trans_att_dict[k] = [x.detach() for x in v]
            else:
                teacher_multi_modal_trans_att_dict[k] = v.detach()

        # Run the Student:
        LOSS_target, CLASS_LOGITS, LANG_LOGITS, LOGITS, AUX_LOGITS, student_multi_modal_trans_att_dict = self.model_student(batch, epoch,
                                                                                                                            compute_loss=True,
                                                                                                                            output_attentions=True,
                                                                                                                            output_hidden_states=True)
        
        # Distilation Loss:
        num_layers, hidden_size = 4, 768
        dist_loss = {'total': 0}
        if self.args.distill_cross_attns > 0:
            cross_attn_masks = batch['obj_masks'].unsqueeze(2) * batch['txt_masks'].unsqueeze(1)
            cross_attn_masks = cross_attn_masks.float()
            cross_attn_sum = cross_attn_masks.sum()
            for i in range(num_layers):
                mse_loss = (teacher_multi_modal_trans_att_dict['all_cross_attns'][i] - student_multi_modal_trans_att_dict['all_cross_attns'][i])**2
                mse_loss = torch.sum(mse_loss * cross_attn_masks) / cross_attn_sum
                dist_loss['cross_attn_%d' % i] = mse_loss * self.args.distill_cross_attns
                dist_loss['total'] += dist_loss['cross_attn_%d' % i]

        if self.args.distill_self_attns > 0:
            self_attn_masks = batch['obj_masks'].unsqueeze(2) * batch['obj_masks'].unsqueeze(1)
            self_attn_masks = self_attn_masks.float()
            self_attn_sum = self_attn_masks.sum()
            for i in range(num_layers):
                mse_loss = (teacher_multi_modal_trans_att_dict['all_self_attns'][i] - student_multi_modal_trans_att_dict['all_self_attns'][i])**2
                mse_loss = torch.sum(mse_loss * self_attn_masks) / self_attn_sum
                dist_loss['self_attn_%d' % i] = mse_loss * self.args.distill_self_attns
                dist_loss['total'] += dist_loss['self_attn_%d' % i]

        if self.args.distill_hiddens > 0:
            hidden_masks = batch['obj_masks'].unsqueeze(2).float()
            hidden_sum = hidden_masks.sum() * hidden_size
            for i in range(num_layers + 1):
                mse_loss = (teacher_multi_modal_trans_att_dict['all_hidden_states'][i] - student_multi_modal_trans_att_dict['all_hidden_states'][i])**2
                mse_loss = torch.sum(mse_loss * hidden_masks) / hidden_sum
                dist_loss['hidden_state_%d' % i] = mse_loss * self.args.distill_hiddens
                dist_loss['total'] += dist_loss['hidden_state_%d' % i]
        
        return (LOSS_target[0]+dist_loss['total'], LOSS_target[1]), CLASS_LOGITS, LANG_LOGITS, LOGITS, AUX_LOGITS
