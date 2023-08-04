import math
from easydict import EasyDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from transformers import BertConfig, BertModel

from .obj_encoder import GTObjEncoder, PcdObjEncoder, ObjColorEncoder
from .txt_encoder import GloveGRUEncoder
from .mmt_module import MMT
from .cmt_module import CMT


def vector_gather(vectors, indices):
    """
    Gathers (batched) vectors according to indices.
    Arguments:
        vectors: Tensor[N, L, D]
        indices: Tensor[N, K] or Tensor[N]
    Returns:
        Tensor[N, K, D] or Tensor[N, D]
    """
    N, L, D = vectors.shape
    squeeze = False
    if indices.ndim == 1:
        squeeze = True
        indices = indices.unsqueeze(-1)
    N2, K = indices.shape
    assert N == N2
    indices = einops.repeat(indices, "N K -> N K D", D=D)
    out = torch.gather(vectors, dim=1, index=indices)
    if squeeze:
        out = out.squeeze(1)
    return out


def get_mlp_head(input_size, hidden_size, output_size, dropout=0):
    return nn.Sequential(
                nn.Linear(input_size, hidden_size//2),
                nn.ReLU(),
                nn.LayerNorm(hidden_size//2, eps=1e-12),
                nn.Dropout(dropout),
                nn.Linear(hidden_size//2, output_size)
            )

def freeze_bn(m):
    '''Freeze BatchNorm Layers'''
    for layer in m.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.eval()


class CoTMMConfig:
    def __init__(self):
        self.type = "cmt"
        self.spatial_dec = False
        self.spatial_multihead = False
        self.spatial_dim = 0
        self.spatial_dist_norm = False
        self.spatial_attn_fusion = "cond"
        self.num_layers = 1
        self.obj_loc_encoding = "same_all"
        self.pairwise_rel_type = "center"
        self.dim_feedforward = 512
        self.hidden_size = 768
        self.num_attention_heads = 16
        self.dim_loc = 6
        

class ReferIt3DNet(nn.Module):
    def __init__(self, config, device, cot_cfg):
        super().__init__()
        self.config = config
        self.device = device
        self.cot_cfg = cot_cfg
        self.anchors_mode = cot_cfg.anchors
        self.max_num_anchors = cot_cfg.max_num_anchors
        self.cot_type = cot_cfg.cot_type
        config.num_obj_classes = config.num_obj_classes + 1  # to add the no_obj class
        self.tgt_mul_w = cot_cfg.tgt_mul_w
        self.gaussian_latent = cot_cfg.gaussian_latent
        self.distractor_aux_loss_flag = cot_cfg.distractor_aux_loss_flag
        self.predict_lang_anchors = cot_cfg.predict_lang_anchors
        self.is_nr = True if 'nr' in cot_cfg.data_csv_pth else False
        self.anchors_ids_type = cot_cfg.anchors_ids_type

        config.obj_encoder.num_obj_classes = config.num_obj_classes
        if self.config.model_type == 'gtlabel':
            self.obj_encoder = GTObjEncoder(config.obj_encoder, config.hidden_size)
        elif self.config.model_type == 'gtpcd':
            self.obj_encoder = PcdObjEncoder(config.obj_encoder)
        if self.config.obj_encoder.freeze:
            freeze_bn(self.obj_encoder)
            for p in self.obj_encoder.parameters():
                p.requires_grad = False
        if self.config.obj_encoder.freeze_bn:
            freeze_bn(self.obj_encoder)

        if self.config.obj_encoder.use_color_enc:
            self.obj_color_encoder = ObjColorEncoder(config.hidden_size, config.obj_encoder.dropout)

        if self.config.txt_encoder.type == 'gru':
            # glove embedding
            self.txt_encoder = GloveGRUEncoder(config.hidden_size, config.txt_encoder.num_layers)
        else:
            txt_bert_config = BertConfig(
                hidden_size=config.hidden_size,
                num_hidden_layers=config.txt_encoder.num_layers,
                num_attention_heads=12, type_vocab_size=2
            )
            self.txt_encoder = BertModel.from_pretrained(
                'bert-base-uncased', config=txt_bert_config
            )
        if self.config.txt_encoder.freeze:
            for p in self.txt_encoder.parameters():
                p.requires_grad = False
    
        mm_config = EasyDict(config.mm_encoder)
        mm_config.hidden_size = config.hidden_size
        mm_config.num_attention_heads = 12
        mm_config.dim_loc = config.obj_encoder.dim_loc
        if self.config.mm_encoder.type == 'cmt':
            self.mm_encoder = CMT(mm_config)
        elif self.config.mm_encoder.type == 'mmt':
            self.mm_encoder = MMT(mm_config)

        if self.anchors_mode == 'cot':
            if self.cot_type == "cross":
                self.parallel_embedding = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                                        nn.ReLU(), nn.LayerNorm(config.hidden_size, eps=1e-12), nn.Dropout(config.dropout))
                self.object_language_clf_parallel = nn.Linear(config.hidden_size, self.max_num_anchors+1)
                if config.losses.cot_teacher_loss:
                    self.cot_decoder = CMT(CoTMMConfig())
                else:
                    self.cot_decoder = nn.TransformerDecoder(torch.nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=16, dim_feedforward=512,
                                                                                                    activation="gelu"), num_layers=1)
                self.og3d_head = get_mlp_head(input_size=config.hidden_size, hidden_size=config.hidden_size,
                                               output_size=self.max_num_anchors+1, dropout=config.dropout)
            else:
                raise Exception("Not Implemented !")
        elif self.anchors_mode == "parallel":
            self.og3d_head = get_mlp_head(input_size=config.hidden_size, hidden_size=config.hidden_size,
                                               output_size=self.max_num_anchors+1, dropout=config.dropout)
        else:
            self.og3d_head = get_mlp_head(config.hidden_size, config.hidden_size, 1, dropout=config.dropout)

        if self.distractor_aux_loss_flag and (self.anchors_mode != 'none'):
            self.distractor_aux_head = get_mlp_head(config.hidden_size, config.hidden_size, 1, dropout=config.dropout)
            self.distractor_aux_bce = nn.BCEWithLogitsLoss()

        if self.config.losses.obj3d_clf > 0:
            self.obj3d_clf_head = get_mlp_head(
                config.hidden_size, config.hidden_size, 
                config.num_obj_classes, dropout=config.dropout
            )
        if self.config.losses.obj3d_clf_pre > 0:
            self.obj3d_clf_pre_head = get_mlp_head(
                config.hidden_size, config.hidden_size,
                config.num_obj_classes, dropout=config.dropout
            )
            if self.config.obj_encoder.freeze:
                for p in self.obj3d_clf_pre_head.parameters():
                    p.requires_grad = False
        if self.config.losses.obj3d_reg > 0:
            self.obj3d_reg_head = get_mlp_head(
                config.hidden_size, config.hidden_size, 
                3, dropout=config.dropout
            )
        if self.config.losses.txt_clf > 0:
            if self.predict_lang_anchors:
                if self.is_nr:
                    if self.anchors_ids_type == "GT":
                        self.lang_out = 11
                    else:
                        self.lang_out = 8
                else:
                    self.lang_out = 3
                #self.num_obj_classes = config.num_obj_classes + 1  # +1 to include the no_obj class
            else:
                self.lang_out = 1
            self.num_obj_classes = config.num_obj_classes

            self.language_trans = nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(d_model=self.num_obj_classes, nhead=8, dim_feedforward=512,
                                                                                        activation="gelu"), num_layers=1)
            self.txt_clf_head = get_mlp_head(config.hidden_size, config.hidden_size, self.num_obj_classes*self.lang_out, dropout=config.dropout)

    def prepare_batch(self, batch):
        outs = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                outs[key] = value.to(self.device)
            else:
                outs[key] = value
        return outs
        
    def _create_att_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 1)
        mask = mask.transpose(0, 1).float()
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0))
        return mask

    def forward(
        self, batch: dict, compute_loss=False, is_test=False,
        output_attentions=False, output_hidden_states=False,
    ) -> dict:
        batch = self.prepare_batch(batch)

        if self.config.obj_encoder.freeze or self.config.obj_encoder.freeze_bn:
            freeze_bn(self.obj_encoder)
        # print("--- batch['obj_fts'] before = ", batch['obj_fts'].shape)  # [64, 52, 300] or [64, 52, 1024, 6]
        obj_embeds = self.obj_encoder(batch['obj_fts'])  # [64, 52, 768]
        if self.config.obj_encoder.freeze:
            obj_embeds = obj_embeds.detach()
        if self.config.obj_encoder.use_color_enc:
            obj_embeds = obj_embeds + self.obj_color_encoder(batch['obj_colors'])

        txt_embeds = self.txt_encoder(
            batch['txt_ids'], batch['txt_masks'],
        ).last_hidden_state
        # print("txt_embeds = ", txt_embeds.shape)  # [64, 13, 768]
        if self.config.txt_encoder.freeze:
            txt_embeds = txt_embeds.detach()

        out_embeds = self.mm_encoder(
            txt_embeds, batch['txt_masks'], 
            obj_embeds, batch['obj_locs'], batch['obj_masks'],
            output_attentions=output_attentions, 
            output_hidden_states=output_hidden_states,
        )  # [128, L, 768])
        # print("batch['obj_locs'] = ", batch['obj_locs'].shape)  # [64, 52, 6]
        
        self.reg_term = None
        if self.gaussian_latent and self.anchors_mode != 'none':
            B, N, E = out_embeds['obj_embeds'].shape  # [N, num_cls, E]
            mean, logvar = out_embeds['obj_embeds'][:, :, :int(E/2)], out_embeds['obj_embeds'][:, :, int(E/2):]  # [N, num_cls, E]
            # sampling
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            out_embeds['obj_embeds'] = mean + eps*std
            out_embeds['obj_embeds'] = torch.cat((out_embeds['obj_embeds'], out_embeds['obj_embeds']), dim=-1)
            self.reg_term = (mean**2 + logvar.exp() - logvar).mean() - 1
            # factor = mean.shape[-1] / batch['obj_fts'].shape[2:].numel()  # 1024*6
            factor = 1
            self.reg_term *= factor

        if self.anchors_mode == 'cot':
            if self.cot_type == "cross":
                parallel_embd = self.parallel_embedding(out_embeds['obj_embeds'])  # [B, L, E] --> [B, L, E]
                AUX_LOGITS = self.object_language_clf_parallel(parallel_embd)  # [B, L, E] --> [B, L, anchors_length+1]
                repeated_mask = batch['obj_masks'].unsqueeze(1).repeat(1, self.max_num_anchors+1, 1)
                AUX_LOGITS = AUX_LOGITS.permute(0, 2, 1)  # [B, L, anchors_length+1] --> [B, anchors_length+1, L]
                AUX_LOGITS.masked_fill_(repeated_mask.logical_not(), -float('inf'))
                trg_pass = torch.argmax(AUX_LOGITS, dim=-1)  # [B, anchors_length+1]
                sampled_embd = vector_gather(parallel_embd, trg_pass)  # [B, L, E] --> [B, anchors_length+1, E]
                if self.config.losses.cot_teacher_loss:
                    sampled_embd_mask = torch.ones((sampled_embd.shape[0], sampled_embd.shape[1]), dtype=torch.bool, device=sampled_embd.device)
                    cot_out_embeds = self.cot_decoder(sampled_embd, sampled_embd_mask,
                                                      out_embeds['obj_embeds'], batch['obj_locs'], batch['obj_masks'],
                                                      output_attentions=output_attentions, output_hidden_states=output_hidden_states,) 
                    cot_out = cot_out_embeds['obj_embeds']
                else:
                    cot_out = self.cot_decoder(out_embeds['obj_embeds'].permute(1, 0, 2), sampled_embd.permute(1, 0, 2), 
                                            tgt_key_padding_mask=batch['obj_masks'].logical_not()).permute(1, 0, 2)  # [B, L, E]
                og3d_logits = self.og3d_head(cot_out).permute(0, 2, 1)  # [B, L, E] --> [B, L, anchors_length+1] --> [B, anchors_length+1, L]
                og3d_logits.masked_fill_(repeated_mask.logical_not(), -float('inf'))
            else:
                raise Exception("Not Implemented !")
        elif self.anchors_mode == "parallel":
            AUX_LOGITS = None
            og3d_logits = self.og3d_head(out_embeds['obj_embeds']).permute(0, 2, 1)  # [B, anchors_length+1, L]
            repeated_mask = batch['obj_masks'].unsqueeze(1).repeat(1, self.max_num_anchors+1, 1)
            #og3d_logits.masked_fill_(repeated_mask.logical_not(), -float('inf'))
        else:
            AUX_LOGITS = None
            og3d_logits = self.og3d_head(out_embeds['obj_embeds']).squeeze(2)  # [B, L]
            og3d_logits.masked_fill_(batch['obj_masks'].logical_not(), -float('inf'))
        result = {
            'og3d_logits': og3d_logits,
        }
        result['AUX_LOGITS'] = AUX_LOGITS
        if self.config.losses.cot_teacher_loss:
            result['sampled_embd_mask'] = sampled_embd_mask

        self.distractor_aux_logits = None
        if self.distractor_aux_loss_flag and (self.anchors_mode != 'none'):
            self.distractor_aux_logits = self.distractor_aux_head(out_embeds['obj_embeds']).squeeze(-1)  # [128, L]

        if output_attentions:
            result['all_cross_attns'] = out_embeds['all_cross_attns']
            result['all_self_attns'] = out_embeds['all_self_attns']
            if self.config.losses.cot_teacher_loss:
                result['cot_all_cross_attns'] = cot_out_embeds['all_cross_attns']
                result['cot_all_self_attns'] = cot_out_embeds['all_self_attns']
        if output_hidden_states:
            result['all_hidden_states'] = out_embeds['all_hidden_states']
            if self.config.losses.cot_teacher_loss:
                result['cot_all_hidden_states'] = cot_out_embeds['all_hidden_states']
        
        if self.config.losses.obj3d_clf > 0:
            result['obj3d_clf_logits'] = self.obj3d_clf_head(out_embeds['obj_embeds'])
        if self.config.losses.obj3d_reg > 0:
            result['obj3d_loc_preds'] = self.obj3d_reg_head(out_embeds['obj_embeds'])
        if self.config.losses.obj3d_clf_pre > 0:
            result['obj3d_clf_pre_logits'] = self.obj3d_clf_pre_head(obj_embeds)
        if self.config.losses.txt_clf > 0:
            if self.predict_lang_anchors:
                AUX_LANG_LOGITS = self.txt_clf_head(txt_embeds[:, 0])
                AUX_LANG_LOGITS = AUX_LANG_LOGITS.contiguous().view(-1, self.num_obj_classes, self.lang_out).permute(0, 2, 1)  # [B, trg_seq_length, num_cls]
                LANG_LOGITS = self.language_trans(AUX_LANG_LOGITS.permute(1, 0, 2)).permute(1, 0, 2)  # [trg_seq_length, B, num_cls] --> [B, trg_seq_length, num_cls]
                #LANG_LOGITS = LANG_LOGITS.contiguous().view(-1, self.num_obj_classes*self.lang_out)  # [B, num_cls, trg_seq_length] --> [B, num_cls*trg_seq_length]
                result['txt_clf_logits_aux'] = AUX_LANG_LOGITS
                result['txt_clf_logits'] = LANG_LOGITS
            else:
                result['txt_clf_logits'] = self.txt_clf_head(txt_embeds[:, 0])
                #result['txt_clf_logits_aux'] = None
        
        if compute_loss:
            losses = self.compute_loss(result, batch)
            return result, losses
        else:
            if self.predict_lang_anchors and False:
                result['txt_clf_logits'] = result['txt_clf_logits'].contiguous().view(-1, self.lang_out, self.num_obj_classes)[:, -1]
            if result['AUX_LOGITS'] == None:
                result.pop('AUX_LOGITS')  # delete it from dict
        
        return result

    def compute_loss(self, result, batch):
        losses = {}
        total_loss = 0
        if self.anchors_mode != 'none':
            trg_pass = torch.cat((batch['anchor_objs_idxs'], batch['tgt_obj_idxs'].unsqueeze(-1)), -1)  # [B, trg_seq_length]
            og3d_loss = 0
            for i in range(trg_pass.shape[-1]):
                mul_w = 1
                if i == self.max_num_anchors:  # the main target
                    mul_w = self.tgt_mul_w
                #mul_w = 0  # TODO: Eslam should remove this one
                og3d_loss += F.cross_entropy(result['og3d_logits'][:, i], trg_pass[:, i]) * mul_w
            #trg_pass_reshaped = trg_pass.reshape(-1)  # [B*trg_seq_length]
            """
            LOGITS_reshaped = result['og3d_logits'].reshape(-1, result['og3d_logits'].shape[2])  # [B*trg_seq_length, L]
            og3d_loss = F.cross_entropy(LOGITS_reshaped, trg_pass_reshaped)
            """
            # result['og3d_logits'] = result['og3d_logits'][:, -1, :]
            if self.reg_term is not None:
                total_loss += self.reg_term
            if self.distractor_aux_logits is not None:
                total_loss += self.distractor_aux_bce(self.distractor_aux_logits, batch['distractor_mask'])
            if result['AUX_LOGITS'] != None:
                # print("AUX_LOGITS = ", result['AUX_LOGITS'].shape)  # [64, 2, 80]
                for i in range(trg_pass.shape[-1]):
                    mul_w = 1
                    if i == self.max_num_anchors:  # the main target
                        mul_w = self.tgt_mul_w
                    #mul_w = 0  # TODO: Eslam should remove this one
                    total_loss += F.cross_entropy(result['AUX_LOGITS'][:, i], trg_pass[:, i]) * mul_w
                #total_loss += F.cross_entropy(result['AUX_LOGITS'].reshape(-1, result['AUX_LOGITS'].shape[2]), trg_pass_reshaped)
                #total_loss += 0  # TODO: Eslam should remove this one
            else:
                result.pop('AUX_LOGITS')  # delete it from dict
        else:
            og3d_loss = F.cross_entropy(result['og3d_logits'], batch['tgt_obj_idxs'])
        losses['og3d'] = og3d_loss
        total_loss += og3d_loss

        if self.config.losses.obj3d_clf > 0:
            obj3d_clf_loss = F.cross_entropy(
                result['obj3d_clf_logits'].permute(0, 2, 1), 
                batch['obj_classes'], reduction='none'
            )
            obj3d_clf_loss = (obj3d_clf_loss * batch['obj_masks']).sum() / batch['obj_masks'].sum()
            losses['obj3d_clf'] = obj3d_clf_loss * self.config.losses.obj3d_clf
            total_loss += losses['obj3d_clf']

        if self.config.losses.obj3d_clf_pre > 0:
            obj3d_clf_pre_loss = F.cross_entropy(
                result['obj3d_clf_pre_logits'].permute(0, 2, 1), 
                batch['obj_classes'], reduction='none'
            )
            obj3d_clf_pre_loss = (obj3d_clf_pre_loss * batch['obj_masks']).sum() / batch['obj_masks'].sum()
            losses['obj3d_clf_pre'] = obj3d_clf_pre_loss * self.config.losses.obj3d_clf_pre
            total_loss += losses['obj3d_clf_pre']

        if self.config.losses.obj3d_reg > 0:
            obj3d_reg_loss = F.mse_loss(
                result['obj3d_loc_preds'], batch['obj_locs'][:, :, :3],  reduction='none'
            )
            obj3d_reg_loss = (obj3d_reg_loss * batch['obj_masks'].unsqueeze(2)).sum() / batch['obj_masks'].sum()
            losses['obj3d_reg'] = obj3d_reg_loss * self.config.losses.obj3d_reg
            total_loss += losses['obj3d_reg']

        if self.config.losses.txt_clf > 0:
            if self.predict_lang_anchors:
                AUX_LANG_LOGITS = result['txt_clf_logits_aux']
                LANG_LOGITS = result['txt_clf_logits']   # [N, trg_seq_length, num_cls]
                
                #LANG_LOGITS = LANG_LOGITS.contiguous().permute(0, 2, 1).reshape(-1, self.num_obj_classes)  # [N*trg_seq_length, num_cls]
                LANG_LOGITS = LANG_LOGITS.contiguous().reshape(-1, self.num_obj_classes)  # [N*trg_seq_length, num_cls]
                lang_tgt = torch.cat((batch['anchor_objs_classes'], batch['tgt_obj_classes'].unsqueeze(-1)), dim=-1).reshape(-1).long()  # [N*trg_seq_length]
                txt_clf_loss = F.cross_entropy(LANG_LOGITS, lang_tgt, reduction='mean')
                if AUX_LANG_LOGITS != None:
                    AUX_LANG_LOGITS = AUX_LANG_LOGITS.contiguous().reshape(-1, self.num_obj_classes)  # [N*trg_seq_length, num_cls]
                    txt_clf_loss += F.cross_entropy(AUX_LANG_LOGITS, lang_tgt, reduction='mean')
                
                result['txt_clf_logits_aux'] = AUX_LANG_LOGITS
                result['txt_clf_logits'] = LANG_LOGITS.contiguous().view(-1, self.lang_out, self.num_obj_classes)[:, -1]
            else:
                txt_clf_loss = F.cross_entropy(result['txt_clf_logits'], batch['tgt_obj_classes'], reduction='mean')
            losses['txt_clf'] = txt_clf_loss * self.config.losses.txt_clf
            total_loss += losses['txt_clf']

        losses['total'] = total_loss
        return losses
