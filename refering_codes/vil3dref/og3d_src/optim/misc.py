"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Misc lr helper
"""
from torch.optim import Adam, Adamax

from .adamw import AdamW
from .rangerlars import RangerLars


def build_optimizer(model, opts):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    obj_enc_params, txt_enc_params, other_params, cot_decoder_params, anchors_lang_trans_params = {}, {}, {}, {}, {}
    for n, p in param_optimizer:
        if not p.requires_grad: continue
        if 'obj_encoder' in n:
            obj_enc_params[n] = p
        elif 'txt_encoder' in n:
            txt_enc_params[n] = p
        elif 'cot_decoder' in n:
            cot_decoder_params[n] = p
        elif 'language_trans' in n:
            anchors_lang_trans_params[n] = p
        else:
            other_params[n] = p
    
    optimizer_grouped_parameters = []
    init_lrs = []
    for ptype, pdict in [('obj', obj_enc_params), ('txt', txt_enc_params), ('others', other_params), ('cot', cot_decoder_params), ('anchors_lang_trans', anchors_lang_trans_params)]:
        if len(pdict) == 0:
            continue
        init_lr = opts.learning_rate
        if ptype == 'obj':
            init_lr = init_lr * getattr(opts, 'obj_encoder_lr_multi', 1)
        elif ptype == 'txt':
            init_lr = init_lr * getattr(opts, 'txt_encoder_lr_multi', 1)
        elif ptype == 'cot':
            init_lr = init_lr * getattr(opts, 'cot_decoder_lr_multi', 1)
        elif ptype == 'anchors_lang_trans':
            init_lr = init_lr * getattr(opts, 'anchors_lang_trans_lr_multi', 1)
        optimizer_grouped_parameters.extend([
            {'params': [p for n, p in pdict.items()
                        if not any(nd in n for nd in no_decay)],
            'weight_decay': opts.weight_decay, 'lr': init_lr},
            {'params': [p for n, p in pdict.items()
                        if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0, 'lr': init_lr}
        ])
        init_lrs.extend([init_lr] * 2)
        
    # currently Adam only
    if opts.optim == 'adam':
        OptimCls = Adam
    elif opts.optim == 'adamax':
        OptimCls = Adamax
    elif opts.optim == 'adamw':
        OptimCls = AdamW
    elif opts.optim == 'rangerlars':
        OptimCls = RangerLars
    else:
        raise ValueError('invalid optimizer')
    optimizer = OptimCls(optimizer_grouped_parameters,
                         lr=opts.learning_rate, betas=opts.betas)
    return optimizer, init_lrs
