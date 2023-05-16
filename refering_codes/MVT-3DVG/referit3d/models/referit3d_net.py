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
import time


class ReferIt3DNet_transformer(nn.Module):

    def __init__(self,
                 args,
                 n_obj_classes,
                 class_name_tokens,
                 ignore_index,
                 class_to_idx):

        super().__init__()

        self.bert_pretrain_path = args.bert_pretrain_path

        self.view_number = args.view_number
        self.rotate_number = args.rotate_number

        self.label_lang_sup = args.label_lang_sup
        self.aggregate_type = args.aggregate_type

        self.encoder_layer_num = args.encoder_layer_num
        self.decoder_layer_num = args.decoder_layer_num
        self.decoder_nhead_num = args.decoder_nhead_num

        self.object_dim = args.object_latent_dim
        self.inner_dim = args.inner_dim

        self.dropout_rate = args.dropout_rate
        self.lang_cls_alpha = args.lang_cls_alpha
        self.obj_cls_alpha = args.obj_cls_alpha

        self.anchors_mode = args.anchors
        self.cot_type = args.cot_type
        self.predict_lang_anchors = args.predict_lang_anchors
        self.lang_filter_objs = args.lang_filter_objs
        self.gaussian_latent = args.gaussian_latent
        self.distractor_aux_loss_flag = args.distractor_aux_loss_flag
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.cls_names = np.array(list(self.idx_to_class.values()))
        self.is_nr = True if 'nr' in args.referit3D_file else False

        if self.anchors_mode == "cot" or self.anchors_mode == "parallel":
            if self.is_nr:
                self.max_num_anchors = args.max_num_anchors
            else:
                self.max_num_anchors = 1
        else:
            self.max_num_anchors = 0
        self.ref_out = self.max_num_anchors + 1
        
        if self.predict_lang_anchors:
            if self.is_nr:
                self.lang_out = self.max_num_anchors + 1
            else:
                self.lang_out = 3 
            self.n_obj_classes = n_obj_classes + 1  # +1 to include the no_obj class
        else:
            self.lang_out = 1
            self.n_obj_classes = n_obj_classes

        self.object_encoder = PointNetPP(sa_n_points=[32, 16, None],
                                         sa_n_samples=[[32], [32], [None]],
                                         sa_radii=[[0.2], [0.4], [None]],
                                         sa_mlps=[[[3, 64, 64, 128]],
                                                  [[128, 128, 128, 256]],
                                                  [[256, 256, self.object_dim, self.object_dim]]])

        self.language_encoder = BertModel.from_pretrained(self.bert_pretrain_path)
        self.language_encoder.encoder.layer = BertModel(BertConfig()).encoder.layer[:self.encoder_layer_num]

        self.refer_encoder = nn.TransformerDecoder(torch.nn.TransformerDecoderLayer(d_model=self.inner_dim,
                                                                                    nhead=self.decoder_nhead_num,
                                                                                    dim_feedforward=2048,
                                                                                    activation="gelu"),
                                                   num_layers=self.decoder_layer_num)

        # Classifier heads
        if self.is_nr:
            self.parallel_language_emedding = nn.Sequential(nn.Linear(self.inner_dim, self.inner_dim),
                                                            nn.ReLU(), nn.Dropout(self.dropout_rate))
            self.language_clf = nn.Linear(self.inner_dim, self.n_obj_classes*self.lang_out)
            self.lang_map = nn.Sequential(nn.Linear(self.n_obj_classes, self.inner_dim),
                                                    nn.ReLU(), nn.Dropout(self.dropout_rate))
            self.language_trans = nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(d_model=self.n_obj_classes, nhead=7, dim_feedforward=512,
                                                                                         activation="gelu"), num_layers=1)
            self.language_clf2 = nn.Sequential(nn.Linear(self.inner_dim, self.inner_dim),
                                               nn.ReLU(), nn.Dropout(self.dropout_rate),
                                               nn.Linear(self.inner_dim, self.n_obj_classes*self.lang_out))
        else:
            self.language_clf = nn.Sequential(nn.Linear(self.inner_dim, self.inner_dim),
                                          nn.ReLU(), nn.Dropout(self.dropout_rate),
                                          nn.Linear(self.inner_dim, self.n_obj_classes*self.lang_out))

        if not self.label_lang_sup:
            self.obj_clf = MLP(self.inner_dim, [self.object_dim, self.object_dim, self.n_obj_classes],
                               dropout_rate=self.dropout_rate)

        self.obj_feature_mapping = nn.Sequential(
            nn.Linear(self.object_dim, self.inner_dim),
            nn.LayerNorm(self.inner_dim),
        )

        self.box_feature_mapping = nn.Sequential(
            nn.Linear(4, self.inner_dim),
            nn.LayerNorm(self.inner_dim),
        )

        if self.gaussian_latent:
            self.inner_dim = int(self.inner_dim / 2)

        if self.anchors_mode == 'cot':
            if self.cot_type == "cross":
                self.parallel_embedding = nn.Sequential(nn.Linear(self.inner_dim, self.inner_dim),
                                                    nn.ReLU(), nn.Dropout(self.dropout_rate))
                self.object_language_clf_parallel = nn.Linear(self.inner_dim, self.ref_out)
                self.object_language_clf = nn.TransformerDecoder(torch.nn.TransformerDecoderLayer(d_model=self.inner_dim, nhead=16, dim_feedforward=512,
                                                                                                activation="gelu"), num_layers=1)
                self.fc_out = nn.Sequential(nn.Linear(self.inner_dim, self.inner_dim),
                                                        nn.ReLU(), nn.Dropout(self.dropout_rate),
                                                        nn.Linear(self.inner_dim, self.ref_out))
            elif self.cot_type == "causal":
                self.parallel_embedding = nn.Sequential(nn.Linear(self.inner_dim, self.inner_dim),
                                                    nn.ReLU(), nn.Dropout(self.dropout_rate))
                self.object_language_clf_parallel = nn.Linear(self.inner_dim, 1)
                self.object_language_clf = nn.TransformerDecoder(torch.nn.TransformerDecoderLayer(d_model=self.inner_dim, nhead=8, dim_feedforward=512,
                                                                                                activation="gelu"), num_layers=1)
                self.fc_out = nn.Sequential(nn.Linear(self.inner_dim, self.inner_dim),
                                                        nn.ReLU(), nn.Dropout(self.dropout_rate),
                                                        nn.Linear(self.inner_dim, 1))
        else:
            self.object_language_clf = nn.Sequential(nn.Linear(self.inner_dim, self.inner_dim),
                                                     nn.ReLU(), nn.Dropout(self.dropout_rate),
                                                     nn.Linear(self.inner_dim, self.ref_out))

        if self.distractor_aux_loss_flag:
            self.distractor_aux_head = nn.Sequential(nn.Linear(self.inner_dim, self.inner_dim),
                                                     nn.ReLU(), nn.Dropout(self.dropout_rate),
                                                     nn.Linear(self.inner_dim, 1))
            self.distractor_aux_bce = nn.BCEWithLogitsLoss()
            
        if self.gaussian_latent:
            self.inner_dim = int(self.inner_dim * 2)

        self.class_name_tokens = class_name_tokens

        self.logit_loss = nn.CrossEntropyLoss()
        self.logit_loss_aux = nn.CrossEntropyLoss()
        self.lang_logits_loss = nn.CrossEntropyLoss()
        self.lang_logits_loss_aux = nn.CrossEntropyLoss()
        self.class_logits_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    @torch.no_grad()
    def aug_input(self, input_points, box_infos):
        input_points = input_points.float().to(self.device)
        box_infos = box_infos.float().to(self.device)
        xyz = input_points[:, :, :, :3]
        bxyz = box_infos[:, :, :3]  # B,N,3
        B, N, P = xyz.shape[:3]
        rotate_theta_arr = torch.Tensor([i * 2.0 * np.pi / self.rotate_number for i in range(self.rotate_number)]).to(
            self.device)
        view_theta_arr = torch.Tensor([i * 2.0 * np.pi / self.view_number for i in range(self.view_number)]).to(
            self.device)

        # rotation
        if self.training:
            # theta = torch.rand(1) * 2 * np.pi  # random direction rotate aug
            theta = rotate_theta_arr[torch.randint(0, self.rotate_number, (B,))]  # 4 direction rotate aug
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
            rotate_matrix = torch.Tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]).to(self.device)[
                None].repeat(B, 1, 1)
            rotate_matrix[:, 0, 0] = cos_theta
            rotate_matrix[:, 0, 1] = -sin_theta
            rotate_matrix[:, 1, 0] = sin_theta
            rotate_matrix[:, 1, 1] = cos_theta

            input_points[:, :, :, :3] = torch.matmul(xyz.reshape(B, N * P, 3), rotate_matrix).reshape(B, N, P, 3)
            bxyz = torch.matmul(bxyz.reshape(B, N, 3), rotate_matrix).reshape(B, N, 3)

        # multi-view
        bsize = box_infos[:, :, -1:]
        boxs = []
        for theta in view_theta_arr:
            rotate_matrix = torch.Tensor([[math.cos(theta), -math.sin(theta), 0.0],
                                          [math.sin(theta), math.cos(theta), 0.0],
                                          [0.0, 0.0, 1.0]]).to(self.device)
            rxyz = torch.matmul(bxyz.reshape(B * N, 3), rotate_matrix).reshape(B, N, 3)
            boxs.append(torch.cat([rxyz, bsize], dim=-1))
        boxs = torch.stack(boxs, dim=1)
        return input_points, boxs

    def compute_loss(self, batch, CLASS_LOGITS, LANG_LOGITS, LOGITS, AUX_LOGITS=None, AUX_LANG_LOGITS=None, distractor_aux_logits=None):
        """
        LOGITS: [N, trg_seq_length, 52]
        """
        if self.anchors_mode != 'none':
            #trg_pass = torch.cat((batch['anchors_pos'][:, 0].unsqueeze(-1), batch['target_pos'].unsqueeze(-1)), -1)  # [N, trg_seq_length]
            trg_pass = torch.cat((batch['anchors_pos'], batch['target_pos'].unsqueeze(-1)), -1)  # [N, trg_seq_length]
            if self.lang_filter_objs_activated and (self.matched_indices is not None):
                #print("trg_pass[:, 0] = ", trg_pass[:, 0])
                #print("self.matched_indices = ", self.matched_indices.shape)
                #print("self.matched_indices = ", self.matched_indices)
                new_anchors_indices, max_matched_anchor_len = find_matching_indices(trg_pass[:, 0], self.matched_indices, self.device)
                #print("new_anchors_indices = ", new_anchors_indices)
                if max_matched_anchor_len:
                    new_tgt_indices, max_matched_tgt_len = find_matching_indices(trg_pass[:, 1], self.matched_indices, self.device)
                    #print("new_tgt_indices = ", new_tgt_indices)
                
                if max_matched_anchor_len and max_matched_tgt_len:
                    trg_pass = torch.cat((new_anchors_indices, new_tgt_indices), -1)
                    #print("new_tgt_indices = ", new_tgt_indices.shape)
                    #print("new_tgt_indices = ", new_tgt_indices)
                else:
                    trg_pass = None
            
            if trg_pass is None:
                referential_loss = 0
            else:
                trg_pass_reshaped = trg_pass.reshape(-1)  # [N*trg_seq_length]
                LOGITS_reshaped = LOGITS.reshape(-1, LOGITS.shape[2])  # [N*trg_seq_length, num_cls]
                referential_loss = self.logit_loss(LOGITS_reshaped, trg_pass_reshaped)
                if AUX_LOGITS != None:
                    referential_loss += self.logit_loss_aux(AUX_LOGITS.reshape(-1, AUX_LOGITS.shape[2]), trg_pass_reshaped)
        else:
            referential_loss = self.logit_loss(LOGITS, batch['target_pos'])
        
        obj_clf_loss = self.class_logits_loss(CLASS_LOGITS.transpose(2, 1), batch['class_labels'])
        
        if self.predict_lang_anchors:
            LANG_LOGITS = LANG_LOGITS.contiguous().view(-1, self.n_obj_classes, self.lang_out).permute(0, 2, 1).reshape(-1, self.n_obj_classes)  # [N*trg_seq_length, num_cls]
            lang_tgt = torch.cat((batch['anchor_classes'], batch['target_class'].unsqueeze(-1)), dim=-1).reshape(-1).long()  # [N*trg_seq_length]
            lang_clf_loss = self.lang_logits_loss(LANG_LOGITS, lang_tgt)
            if AUX_LANG_LOGITS != None:
                AUX_LANG_LOGITS = AUX_LANG_LOGITS.contiguous().view(-1, self.n_obj_classes, self.lang_out).permute(0, 2, 1).reshape(-1, self.n_obj_classes)  # [N*trg_seq_length, num_cls]
                lang_clf_loss += self.lang_logits_loss_aux(LANG_LOGITS, lang_tgt)
        else:
            lang_clf_loss = self.lang_logits_loss(LANG_LOGITS, batch['target_class'])
        
        if distractor_aux_logits is not None:
            # https://discuss.pytorch.org/t/multi-label-classification-in-pytorch/905/44
            distractor_aux_loss = self.distractor_aux_bce(distractor_aux_logits, batch['distractor_mask'])
        else:
            distractor_aux_loss = 0

        total_loss = referential_loss + self.obj_cls_alpha * obj_clf_loss + self.lang_cls_alpha * lang_clf_loss + distractor_aux_loss

        if self.gaussian_latent:
            total_loss += self.reg_term
        
        if trg_pass is None:
            return (total_loss, batch['target_pos'])
        else:
            return (total_loss, trg_pass[:, -1])
        

    def forward(self, batch: dict, epoch=None):
        # batch['class_labels']: GT class of each obj
        # batch['target_class']：GT class of target obj
        # batch['target_pos']: GT id

        if (epoch > 10) and self.lang_filter_objs:
            self.lang_filter_objs_activated = True
        else:
            self.lang_filter_objs_activated = False

        self.device = self.obj_feature_mapping[0].weight.device

        ## rotation augmentation and multi_view generation
        obj_points, boxs = self.aug_input(batch['objects'], batch['box_info'])  # [128, 52, 1024, 6], [128, 4, 52, 4]
        B, N, P = obj_points.shape[:3]

        ## obj_encoding
        #  torch.Size([128, 52, 1024, 6]) --> torch.Size([128, 52, 768])
        objects_features = get_siamese_features(self.object_encoder, obj_points, aggregator=torch.stack)

        ## obj_encoding
        obj_feats = self.obj_feature_mapping(objects_features)  # torch.Size([128, 52, 768])
        # [128, 4, 52, 4] --> [128, 4, 52, 768]
        box_infos = self.box_feature_mapping(boxs)
        obj_infos = obj_feats[:, None].repeat(1, self.view_number, 1, 1) + box_infos

        # <LOSS>: obj_cls
        if self.label_lang_sup:
            label_lang_infos = self.language_encoder(**self.class_name_tokens)[0][:, 0]
            CLASS_LOGITS = torch.matmul(obj_feats.reshape(B * N, -1), label_lang_infos.permute(1, 0)).reshape(B, N, -1)
        else:
            CLASS_LOGITS = self.obj_clf(obj_feats.reshape(B * N, -1)).reshape(B, N, -1)  # [128, 52, 608]

        ## language_encoding
        lang_tokens = batch['lang_tokens']
        lang_infos = self.language_encoder(**lang_tokens)[0]  # [128, 20, 768]

        # <LOSS>: lang_cls
        if self.is_nr:
            lang_parallel_embd = self.parallel_language_emedding(lang_infos[:, 0])
            AUX_LANG_LOGITS = self.language_clf(lang_parallel_embd)  # [B, num_cls*trg_seq_length]
            sampled_embd = AUX_LANG_LOGITS.contiguous().view(-1, self.n_obj_classes, self.lang_out).permute(0, 2, 1)  # [N, trg_seq_length, num_cls]
            #sampled_embd = self.lang_map(AUX_LANG_LOGITS)  # [B, trg_seq_length, num_cls] -->   # [B, trg_seq_length, E]
            #lang_trans_out = self.language_trans(lang_infos[:, 0].permute(1, 0, 2), sampled_embd.permute(1, 0, 2)).permute(1, 0, 2)  # [B, trg_seq_length, E]
            #LANG_LOGITS = self.language_clf2(lang_trans_out).permute(0, 2, 1)  # [N, anchors_length+1, E] --> [N, anchors_length+1, num_cls]
            LANG_LOGITS = self.language_trans(sampled_embd.permute(1, 0, 2)).permute(1, 2, 0)  # [trg_seq_length, B, num_cls] --> [B, num_cls, trg_seq_length]
            LANG_LOGITS = LANG_LOGITS.contiguous().view(-1, self.n_obj_classes*self.lang_out)  # [B, num_cls, trg_seq_length] --> [B, num_cls*trg_seq_length]
        else:
            AUX_LANG_LOGITS = None
            LANG_LOGITS = self.language_clf(lang_infos[:, 0])  # [128, 607]  # [B, num_cls*trg_seq_length]

        ## multi-modal_fusion
        cat_infos = obj_infos.reshape(B * self.view_number, -1, self.inner_dim)  # [512, 52, 768]
        mem_infos = lang_infos[:, None].repeat(1, self.view_number, 1, 1).reshape(B * self.view_number,
                                                                                  -1, self.inner_dim)  # [512, 20, 768]
        out_feats = self.refer_encoder(cat_infos.transpose(0, 1),
                                       mem_infos.transpose(0, 1)).transpose(0, 1).reshape(B, self.view_number, -1,
                                                                                          self.inner_dim)  # [128, 4, 52, 768]

        ## view_aggregation
        refer_feat = out_feats
        if self.aggregate_type == 'avg':
            agg_feats = (refer_feat / self.view_number).sum(dim=1)
        elif self.aggregate_type == 'avgmax':
            agg_feats = (refer_feat / self.view_number).sum(dim=1) + refer_feat.max(dim=1).values
        else:
            agg_feats = refer_feat.max(dim=1).values
        # print("agg_feats: ", agg_feats.shape)  # [128, 52, 768]

        if self.lang_filter_objs_activated:
            # Get lang predictions:
            pred_lang_objs_idx = torch.argmax(LANG_LOGITS.contiguous().view(-1, self.n_obj_classes, 3), dim=1)  # [B, 607, 3] --> [B, 3]
            pred_lang_objs_names = self.cls_names[pred_lang_objs_idx.reshape(-1).cpu().numpy()]  # [B*(anchors_length+1)]
            pred_lang_objs_names = pred_lang_objs_names.reshape(-1,3)  # [B*(anchors_length+1)] -->   # [B, (anchors_length+1)]
            # Get cls predictions:
            pred_vis_objs_idx = torch.argmax(CLASS_LOGITS, dim=-1)  # [B, 52, 608] --> [B, 52]
            pred_vis_objs_names = self.cls_names[pred_vis_objs_idx.reshape(-1).cpu().numpy()]  # [B*52]
            pred_vis_objs_names = np.array([s.replace('no_obj', 'dummy_class') for s in pred_vis_objs_names])
            pred_vis_objs_names = pred_vis_objs_names.reshape(-1,N)  # [B*52] --> [B, 52]
            #print("pred_lang_objs_names = ", pred_lang_objs_names[0])
            #print("pred_vis_objs_names = ", pred_vis_objs_names[0])
            # Find matched indices:
            self.matched_indices, max_matched_obj_len = find_matching_indices(pred_lang_objs_names, pred_vis_objs_names, self.device)
            if self.matched_indices is not None:
                #print("self.matched_indices = ", self.matched_indices.shape)
                agg_feats = vector_gather(agg_feats, self.matched_indices)  # [B, num_cls, E] --> [B, anchors_length+1, E]
            
            #print("agg_feats = ", agg_feats.shape)
            #print("max_matched_obj_len = ", max_matched_obj_len)
            
        self.reg_term = None
        if self.gaussian_latent:
            B, N, E = agg_feats.shape  # [N, num_cls, E]
            mean, logvar = agg_feats[:, :, :int(E/2)], agg_feats[:, :, int(E/2):]  # [N, num_cls, E]
            # sampling
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            agg_feats = mean + eps*std
            self.reg_term = (mean**2 + logvar.exp() - logvar).mean() - 1
            factor = mean.shape[-1] / obj_points.shape[2:].numel()  # 1024*6
            self.reg_term *= factor

        if self.anchors_mode == "cot":
            if self.cot_type == "cross":
                parallel_embd = self.parallel_embedding(agg_feats)  # [N, num_cls, E] --> [N, num_cls, E]
                AUX_LOGITS = self.object_language_clf_parallel(parallel_embd)  # [N, num_cls, E] --> [N, num_cls, anchors_length+1]
                AUX_LOGITS = AUX_LOGITS.permute(0, 2, 1)  # [N, num_cls, anchors_length+1] --> [N, anchors_length+1, num_cls]
                trg_pass = torch.argmax(AUX_LOGITS, dim=-1)  # [N, trg_seq_length]
                sampled_embd = vector_gather(parallel_embd, trg_pass)  # [N, num_cls, E] --> [N, anchors_length+1, E]
                cot_out = self.object_language_clf(agg_feats.permute(1, 0, 2), sampled_embd.permute(1, 0, 2)).permute(1, 0, 2)  # [N, anchors_length+1, E]
                LOGITS = self.fc_out(cot_out).permute(0, 2, 1)  # [N, anchors_length+1, E] --> [N, anchors_length+1, num_cls]
            elif self.cot_type == "causal":
                parallel_embd = self.parallel_embedding(agg_feats)  # [N, num_cls, E] --> [N, num_cls, E]
                anchor_LOGITS = self.object_language_clf_parallel(parallel_embd)  # [N, num_cls, E] --> [N, num_cls, 1]
                anchor_LOGITS = anchor_LOGITS.permute(0, 2, 1)  # [N, num_cls, 1] --> [N, 1, num_cls]
                trg_pass = torch.argmax(anchor_LOGITS, dim=-1)  # [N, 1]
                sampled_embd = vector_gather(parallel_embd, trg_pass)  # [N, num_cls, E] --> [N, 1, E]
                cot_out = self.object_language_clf(agg_feats.permute(1, 0, 2), sampled_embd.permute(1, 0, 2)).permute(1, 0, 2)  # [N, 52, E]
                target_LOGITS = self.fc_out(cot_out).permute(0, 2, 1)  # [N, 52, E] --> [N, 52, 1] --> [N, 1, 52]
                LOGITS = torch.cat((anchor_LOGITS, target_LOGITS), dim=1)  # [N, 2, 52]
                AUX_LOGITS = None
        elif self.anchors_mode == "parallel":
            cot_out = self.object_language_clf(agg_feats)  # [N, 52, E] --> [N, 52, trg_seq_length]
            LOGITS = cot_out.permute(0, 2, 1)  # [N, 52, trg_seq_length] --> [N, trg_seq_length, 52]
            AUX_LOGITS = None
        else:
            LOGITS = self.object_language_clf(agg_feats).squeeze(-1)  # [128, 52, trg_seq_length]
            AUX_LOGITS = None

        if self.distractor_aux_loss_flag:
            distractor_aux_logits = self.distractor_aux_head(agg_feats)  # [B, num_cls, E] --> [B, num_cls, 1]
            distractor_aux_logits = distractor_aux_logits.squeeze(-1)  # [B, num_cls, 1] --> [B, num_cls]
        else:
            distractor_aux_logits = None
        # <LOSS>: ref_cls
        LOSS = self.compute_loss(batch, CLASS_LOGITS, LANG_LOGITS, LOGITS, AUX_LOGITS, AUX_LANG_LOGITS, distractor_aux_logits)  # []
        if self.anchors_mode != 'none':
            LOGITS = LOGITS[:, -1]
        
        if self.predict_lang_anchors:
            LANG_LOGITS = LANG_LOGITS.contiguous().view(-1, self.n_obj_classes, self.lang_out)[:,:,-1]
        
        return LOSS, CLASS_LOGITS, LANG_LOGITS, LOGITS
