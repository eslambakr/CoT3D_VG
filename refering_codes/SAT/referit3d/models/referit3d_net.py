import torch
import random
import argparse
from torch import nn
from collections import defaultdict
import numpy as np 

from pytorch_transformers.modeling_bert import (
    BertLayerNorm, BertEmbeddings, BertEncoder, BertConfig,
    BertPreTrainedModel
)
from .mmt_module import *

from . import DGCNN
from .default_blocks import *
from .utils import get_siamese_features, find_matching_indices, pad_matched_indx, vector_gather
from ..in_out.vocabulary import Vocabulary

try:
    from .backbone.point_net_pp import PointNetPP
except ImportError:
    PointNetPP = None


class MMT_ReferIt3DNet(nn.Module):
    def __init__(self,
                 args,
                 object_encoder,
                 num_class,
                 visudim=128,
                 MMT_HIDDEN_SIZE=192,
                 TEXT_BERT_HIDDEN_SIZE=768,
                 context_2d=None,
                 feat2dtype=None,
                 mmt_mask=None, 
                 class_to_idx=None
                 ):

        super().__init__()

        self.args_mode = args.mode
        self.text_length = args.max_seq_len
        self.context_2d = context_2d
        self.feat2dtype = feat2dtype
        self.mmt_mask = mmt_mask
        
        #Prepare for COT 
        self.bert_pretrain_path = args.bert_pretrain_path
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
                self.lang_out = 2 
            self.n_obj_classes = num_class + 1  # +1 to include the no_obj class
        else:
            self.lang_out = 1
            self.n_obj_classes = num_class
        
        # Encoders for single object
        self.object_encoder = object_encoder
        self.linear_obj_feat_to_mmt_in = nn.Linear(visudim, MMT_HIDDEN_SIZE)
        self.linear_obj_bbox_to_mmt_in = nn.Linear(4, MMT_HIDDEN_SIZE)
        self.obj_feat_layer_norm = BertLayerNorm(MMT_HIDDEN_SIZE)
        self.obj_bbox_layer_norm = BertLayerNorm(MMT_HIDDEN_SIZE)
        self.obj_drop = nn.Dropout(0.1)
        # Encoders for visual 2D objects
        num_class_dim = self.n_obj_classes
        if (args.feat2d.replace('3D',''))=='ROI': featdim = 2048
        elif (args.feat2d.replace('3D',''))=='clsvec': featdim = num_class_dim
        elif (args.feat2d.replace('3D',''))=='clsvecROI': featdim = 2048+num_class_dim
        self.linear_2d_feat_to_mmt_in = nn.Linear(featdim, MMT_HIDDEN_SIZE)
        self.linear_2d_bbox_to_mmt_in = nn.Linear(16, MMT_HIDDEN_SIZE)
        self.obj2d_feat_layer_norm = BertLayerNorm(MMT_HIDDEN_SIZE)
        self.obj2d_bbox_layer_norm = BertLayerNorm(MMT_HIDDEN_SIZE)

        ## encoder for context object
        self.cnt_object_encoder = single_object_encoder(768)
        self.cnt_linear_obj_feat_to_mmt_in = nn.Linear(visudim, MMT_HIDDEN_SIZE)
        self.cnt_linear_obj_bbox_to_mmt_in = nn.Linear(4, MMT_HIDDEN_SIZE)
        self.cnt_feat_layer_norm = BertLayerNorm(MMT_HIDDEN_SIZE)
        self.cnt_bbox_layer_norm = BertLayerNorm(MMT_HIDDEN_SIZE)
        self.context_drop = nn.Dropout(0.1)

        # Encoders for text
        self.text_bert_config = BertConfig(
                 hidden_size=TEXT_BERT_HIDDEN_SIZE,
                 num_hidden_layers=3,
                 num_attention_heads=12,
                 type_vocab_size=2)
        self.text_bert = TextBert.from_pretrained(
            'bert-base-uncased', config=self.text_bert_config,\
            mmt_mask=self.mmt_mask)
        if TEXT_BERT_HIDDEN_SIZE!=MMT_HIDDEN_SIZE:
            self.text_bert_out_linear = nn.Linear(TEXT_BERT_HIDDEN_SIZE, MMT_HIDDEN_SIZE)
        else:
            self.text_bert_out_linear = nn.Identity()

        # Classifier heads
        # Optional, make a bbox encoder
        self.object_clf = None
        if args.obj_cls_alpha > 0:
            print('Adding an object-classification loss.')
            self.object_clf = object_decoder_for_clf(visudim, self.n_obj_classes)

        self.language_clf = None
        if args.lang_cls_alpha > 0:
            print('Adding a text-classification loss.')
            self.language_clf = text_decoder_for_clf(TEXT_BERT_HIDDEN_SIZE, self.n_obj_classes)

        self.mmt_config = BertConfig(
                 hidden_size=MMT_HIDDEN_SIZE,
                 num_hidden_layers=4,
                 num_attention_heads=12,
                 type_vocab_size=2)
        self.mmt = MMT(self.mmt_config,context_2d=self.context_2d,mmt_mask=self.mmt_mask)
        self.mlm_cls = BertLMPredictionHead(self.text_bert.embeddings.word_embeddings.weight, input_size=MMT_HIDDEN_SIZE)
        self.contra_cls = PolluteLinear()
        
        #additional for COT: 
        #ADD LANGUAGE
        lang_out_dim = 768
        if self.is_nr:
            self.parallel_language_embedding = nn.Sequential(nn.Linear(lang_out_dim, 128),
                                                             nn.BatchNorm1d(128),
                                                             nn.ReLU(), 
                                                             nn.Dropout(0.2))
            self.language_clf     = nn.Linear(128, self.n_obj_classes * self.lang_out)
            self.language_trans = nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(d_model=self.n_obj_classes, nhead=7, dim_feedforward=512,
                                                                                         activation="gelu"), num_layers=1)
        else:
            self.language_clf = nn.Sequential(nn.Linear(lang_out_dim, 128),
                                                             nn.BatchNorm1d(128),
                                                             nn.ReLU(), 
                                                             nn.Dropout(0.2),
                                                             nn.Linear(128, self.n_obj_classes * self.lang_out))
            

        # self.language_trans = nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(d_model=self.n_obj_classes, nhead=7, dim_feedforward=512,
        #                                                                                  activation="gelu"), num_layers=1)
        #COT-> CROSS 
        if self.anchors_mode == 'cot':
            if self.cot_type == "cross":
                self.parallel_embedding = nn.Sequential(nn.Linear(MMT_HIDDEN_SIZE, MMT_HIDDEN_SIZE),
                                                        nn.GELU(),
                                                        BertLayerNorm(MMT_HIDDEN_SIZE, eps=1e-12))
                self.object_language_clf_parallel = nn.Linear(MMT_HIDDEN_SIZE, self.ref_out)
                self.object_language_clf = nn.TransformerDecoder(torch.nn.TransformerDecoderLayer(d_model=MMT_HIDDEN_SIZE, nhead=16,
                                                                                                  dim_feedforward=512, activation="gelu"), num_layers=1)
                self.fc_out = MatchingLinear(input_size=MMT_HIDDEN_SIZE, outputdim=self.ref_out)
                
                # CoT for the 2D stream:
                if self.context_2d=='unaligned':
                    self.parallel_embedding_2d = nn.Sequential(nn.Linear(MMT_HIDDEN_SIZE, MMT_HIDDEN_SIZE),
                                                               nn.GELU(),
                                                               BertLayerNorm(MMT_HIDDEN_SIZE, eps=1e-12))
                    self.object_language_clf_parallel_2d = nn.Linear(MMT_HIDDEN_SIZE, self.ref_out)
                    self.object_language_clf_2d = nn.TransformerDecoder(torch.nn.TransformerDecoderLayer(d_model=MMT_HIDDEN_SIZE, nhead=16,
                                                                                                         dim_feedforward=512, activation="gelu"), num_layers=1)
                    self.fc_out_2d = MatchingLinear(input_size=MMT_HIDDEN_SIZE, outputdim=self.ref_out)
                
        else:
            self.matching_cls = MatchingLinear(input_size=MMT_HIDDEN_SIZE)
            if self.context_2d=='unaligned':
                self.matching_cls_2D = MatchingLinear(input_size=MMT_HIDDEN_SIZE)
            
    def __call__(self, batch: dict,  epoch=None) -> dict:
        result = defaultdict(lambda: None)

        # Get features for each segmented scan object based on color and point-cloud
        objects_features = get_siamese_features(self.object_encoder, batch['objects'],
                                                aggregator=torch.stack)  # B X N_Objects x object-latent-dim -> [16,52,768]
        obj_mmt_in = self.obj_feat_layer_norm(self.linear_obj_feat_to_mmt_in(objects_features)) + \
            self.obj_bbox_layer_norm(self.linear_obj_bbox_to_mmt_in(batch['obj_offset'])) 
        if self.context_2d=='aligned':
            obj_mmt_in = obj_mmt_in + \
                self.obj2d_feat_layer_norm(self.linear_2d_feat_to_mmt_in(batch['feat_2d'])) + \
                self.obj2d_bbox_layer_norm(self.linear_2d_bbox_to_mmt_in(batch['coords_2d']))

        obj_mmt_in = self.obj_drop(obj_mmt_in)
        obj_num = obj_mmt_in.size(1)
        obj_mask = _get_mask(batch['context_size'].to(obj_mmt_in.device), obj_num)    ## all proposals are non-empty

        if self.object_clf is not None:
            objects_classifier_features = obj_mmt_in
            result['class_logits'] = get_siamese_features(self.object_clf, objects_classifier_features, torch.stack)

        if self.context_2d=='unaligned':
            context_obj_mmt_in = self.obj2d_feat_layer_norm(self.linear_2d_feat_to_mmt_in(batch['feat_2d'])) + \
                self.obj2d_bbox_layer_norm(self.linear_2d_bbox_to_mmt_in(batch['coords_2d']))
            context_obj_mmt_in = self.context_drop(context_obj_mmt_in)
            context_obj_mask = _get_mask(batch['context_size'].to(context_obj_mmt_in.device), obj_num)    ## all proposals are non-empty
            obj_mmt_in = torch.cat([obj_mmt_in, context_obj_mmt_in],dim=1)
            obj_mask = torch.cat([obj_mask, context_obj_mask],dim=1)

        # Get feature for utterance
        txt_inds = batch["token_inds"] # batch_size, lang_size
        txt_type_mask = torch.ones(txt_inds.shape, device=torch.device('cuda')) * 1.
        txt_mask = _get_mask(batch['token_num'].to(txt_inds.device), txt_inds.size(1))  ## all proposals are non-empty
        txt_type_mask = txt_type_mask.long()

        text_bert_out = self.text_bert(
            txt_inds=txt_inds,
            txt_mask=txt_mask,
            txt_type_mask=txt_type_mask
        )
        txt_emb = self.text_bert_out_linear(text_bert_out) #[16,24,768] -> [B,Language_size, E ]
        # Classify the target instance label based on the text
        # import pdb; pdb.set_trace()
        
        '''
        lang_parallel_embd = self.parallel_language_embedding(text_bert_out[:,0,:]) 
                AUX_LANG_LOGITS    = self.language_clf(lang_parallel_embd)   # [B, num_cls*trg_seq_length]
                # sampled_embd = AUX_LANG_LOGITS.contiguous().view(-1, self.n_obj_classes, self.lang_out).permute(0, 2, 1)  # [N, num_cls, anchors_length+1] --> [N, anchors_length+1, num_cls]
                # LANG_LOGITS = self.language_trans(sampled_embd.permute(1, 0, 2)).permute(1, 2, 0)  # [trg_seq_length, B, num_cls] --> [B, num_cls, trg_seq_length]
                LANG_LOGITS = AUX_LANG_LOGITS
                result['lang_logits']= LANG_LOGITS.contiguous().view(-1, self.n_obj_classes*self.lang_out)
                result['aux_lang_logits'] = None
        '''
        if self.language_clf is not None:
            if self.is_nr:  #Just change the output dimension to make it Number Of anchors + target instead of target only.
                lang_parallel_embd = self.parallel_language_embedding(text_bert_out[:,0,:]) 
                AUX_LANG_LOGITS    = self.language_clf(lang_parallel_embd)   # [B, num_cls*trg_seq_length]
                # sampled_embd = AUX_LANG_LOGITS.contiguous().view(-1, self.n_obj_classes, self.lang_out).permute(0, 2, 1)  # [N, num_cls, anchors_length+1] --> [N, anchors_length+1, num_cls]
                # LANG_LOGITS = self.language_trans(sampled_embd.permute(1, 0, 2)).permute(1, 2, 0)  # [trg_seq_length, B, num_cls] --> [B, num_cls, trg_seq_length]
                result['lang_logits'] = AUX_LANG_LOGITS
                # result['lang_logits']= LANG_LOGITS.contiguous().view(-1, self.n_obj_classes*self.lang_out)
                result['aux_lang_logits'] = None
            else:
                result['aux_lang_logits'] = None
                result['lang_logits'] = self.language_clf(text_bert_out[:,0,:])

        mmt_results = self.mmt(
            txt_emb=txt_emb,
            txt_mask=txt_mask,
            obj_emb=obj_mmt_in,
            obj_mask=obj_mask,
            obj_num=obj_num
        )
        agg_feats = mmt_results['mmt_obj_output']  # [B, num_cls, E]

        if self.args_mode == 'evaluate':
            assert(mmt_results['mmt_seq_output'].shape[1]==(self.text_length+obj_num))
        if self.args_mode != 'evaluate' and self.context_2d=='unaligned':
            assert(mmt_results['mmt_seq_output'].shape[1]==(self.text_length+obj_num*2))
        
        #Changing refer head 
        if self.anchors_mode == "cot":
            if self.cot_type == "cross":
                parallel_embd = self.parallel_embedding(agg_feats)  # [N, num_cls, E] --> [N, num_cls, E]
                AUX_LOGITS = self.object_language_clf_parallel(parallel_embd)  # [N, num_cls, E] --> [N, num_cls, anchors_length+1]
                AUX_LOGITS = AUX_LOGITS.permute(0, 2, 1)  # [N, num_cls, anchors_length+1] --> [N, anchors_length+1, num_cls]
                trg_pass = torch.argmax(AUX_LOGITS, dim=-1)  # [N, trg_seq_length]
                sampled_embd = vector_gather(parallel_embd, trg_pass)  # [N, num_cls, E] --> [N, anchors_length+1, E]
                cot_out = self.object_language_clf(agg_feats.permute(1, 0, 2), sampled_embd.permute(1, 0, 2)).permute(1, 0, 2)  # [N, anchors_length+1, E]
                LOGITS = self.fc_out(cot_out).permute(0, 2, 1)  # [N, anchors_length+1, E] --> [N, anchors_length+1, num_cls]
                result['logits'] = LOGITS
        elif self.anchors_mode == "parallel":
            cot_out = self.object_language_clf(agg_feats)  # [N, 52, E] --> [N, 52, trg_seq_length]
            result['logits'] = cot_out.permute(0, 2, 1)  # [N, 52, trg_seq_length] --> [N, trg_seq_length, 52]
            AUX_LOGITS = None
        else:
            result['logits'] = self.matching_cls(mmt_results['mmt_obj_output'])
            AUX_LOGITS = None
        
        result['aux_logits'] = AUX_LOGITS
        result['mmt_obj_output'] = mmt_results['mmt_obj_output']
        if self.context_2d=='unaligned':
            if self.anchors_mode == "cot":
                if self.cot_type == "cross":
                    agg_feats_2d = mmt_results['mmt_obj_output_2D']
                    parallel_embd_2d = self.parallel_embedding_2d(agg_feats_2d)  # [N, num_cls, E] --> [N, num_cls, E]
                    AUX_LOGITS_2d = self.object_language_clf_parallel_2d(parallel_embd_2d)  # [N, num_cls, E] --> [N, num_cls, anchors_length+1]
                    AUX_LOGITS_2d = AUX_LOGITS_2d.permute(0, 2, 1)  # [N, num_cls, anchors_length+1] --> [N, anchors_length+1, num_cls]
                    trg_pass_2d = torch.argmax(AUX_LOGITS_2d, dim=-1)  # [N, trg_seq_length]
                    sampled_embd_2d = vector_gather(parallel_embd_2d, trg_pass_2d)  # [N, num_cls, E] --> [N, anchors_length+1, E]
                    cot_out_2d = self.object_language_clf_2d(agg_feats_2d.permute(1, 0, 2), sampled_embd_2d.permute(1, 0, 2)).permute(1, 0, 2)  # [N, anchors_length+1, E]
                    LOGITS_2d = self.fc_out_2d(cot_out_2d).permute(0, 2, 1)  # [N, anchors_length+1, E] --> [N, anchors_length+1, num_cls]
                    result['logits_2D'] = LOGITS_2d
            elif self.anchors_mode == "parallel":
                cot_out_2d = self.object_language_clf_2d(agg_feats_2d)  # [N, 52, E] --> [N, 52, trg_seq_length]
                result['logits_2D'] = cot_out_2d.permute(0, 2, 1)  # [N, 52, trg_seq_length] --> [N, trg_seq_length, 52]
                AUX_LOGITS = None
            else:
                result['logits_2D'] = self.matching_cls_2D(mmt_results['mmt_obj_output_2D'])
                AUX_LOGITS = None

            result['mmt_obj_output_2D'] = mmt_results['mmt_obj_output_2D']
        return result

def instantiate_referit3d_net(args: argparse.Namespace, vocab: Vocabulary, n_obj_classes: int, class_to_idx: dict) -> nn.Module:
    """
    Creates a neural listener by utilizing the parameters described in the args
    but also some "default" choices we chose to fix in this paper.

    @param args:
    @param vocab:
    @param n_obj_classes: (int)
    @param class_to_idx: (dict)
    """

    # convenience
    geo_out_dim = args.object_latent_dim
    lang_out_dim = args.language_latent_dim
    mmt_out_dim = args.mmt_latent_dim

    # make an object (segment) encoder for point-clouds with color
    if args.object_encoder == 'pnet_pp':
        object_encoder = single_object_encoder(geo_out_dim)
    else:
        raise ValueError('Unknown object point cloud encoder!')

    if args.model.startswith('mmt') and args.transformer:
        model = MMT_ReferIt3DNet(            
            args=args,
            num_class=n_obj_classes,
            object_encoder=object_encoder,
            visudim=geo_out_dim,
            TEXT_BERT_HIDDEN_SIZE=lang_out_dim,
            MMT_HIDDEN_SIZE=mmt_out_dim,
            context_2d=args.context_2d,
            feat2dtype=args.feat2d,
            mmt_mask=args.mmt_mask, 
            class_to_idx=class_to_idx)
    else:
        raise NotImplementedError('Unknown listener model is requested.')

    return model


## pad at the end; used anyway by obj, ocr mmt encode
def _get_mask(nums, max_num):
    # non_pad_mask: b x lq, torch.float32, 0. on PAD
    batch_size = nums.size(0)
    arange = torch.arange(0, max_num).unsqueeze(0).expand(batch_size, -1)
    non_pad_mask = arange.to(nums.device).lt(nums.unsqueeze(-1))
    non_pad_mask = non_pad_mask.type(torch.float32)
    return non_pad_mask