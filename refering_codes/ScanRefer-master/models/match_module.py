import torch
import torch.nn as nn
import einops


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


class MatchModule(nn.Module):
    def __init__(self, num_proposals=256, lang_size=256, hidden_size=128, anchors=None, cot_type=None, max_num_anchors=None, feedcotpath=None):
        super().__init__() 

        self.num_proposals = num_proposals
        self.lang_size = lang_size
        self.hidden_size = hidden_size

        # CoT:
        self.anchors = anchors
        self.cot_type = cot_type
        self.max_num_anchors = max_num_anchors
        self.feedcotpath = feedcotpath
        
        self.fuse = nn.Sequential(
            nn.Conv1d(self.lang_size + 128, hidden_size, 1),
            nn.ReLU()
        )
        
        if self.anchors == "none":
            self.max_num_anchors = 0
            
        self.ref_out = self.max_num_anchors + 1

        self.dropout_rate = 0.15
        if self.anchors == 'cot':
            if self.cot_type == "cross":
                self.parallel_embedding = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(self.dropout_rate))
                self.object_language_clf_parallel = nn.Linear(hidden_size, self.ref_out)
                self.object_language_clf = nn.TransformerDecoder(torch.nn.TransformerDecoderLayer(d_model=hidden_size, nhead=16, dim_feedforward=512,
                                                                                                  activation="gelu"), num_layers=1)
                self.fc_out = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                                        nn.ReLU(), nn.Dropout(self.dropout_rate),
                                                        nn.Linear(hidden_size, self.ref_out))
            elif self.cot_type == "self_cons":
                self.object_language_clf = nn.TransformerDecoder(torch.nn.TransformerDecoderLayer(d_model=hidden_size, nhead=16, dim_feedforward=512,
                                                                                                  activation="gelu"), num_layers=1)
                self.fc_out = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                                        nn.ReLU(), nn.Dropout(self.dropout_rate),
                                                        nn.Linear(hidden_size, self.ref_out))
        else:
            self.match = nn.Sequential(
                nn.Conv1d(hidden_size, hidden_size, 1),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Conv1d(hidden_size, hidden_size, 1),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Conv1d(hidden_size, self.ref_out, 1)
            )
        if self.anchors != 'none' and self.feedcotpath:
            # Language fusion layer for CoT path:
            self.lang_cot_path_fusion = nn.Sequential(nn.Linear(lang_size*2, lang_size), nn.ReLU(), nn.Dropout(self.dropout_rate))

    def forward(self, data_dict):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """

        # unpack outputs from detection branch
        features = data_dict['aggregated_vote_features'] # batch_size, num_proposal, 128
        objectness_masks = data_dict['objectness_scores'].max(2)[1].float().unsqueeze(2) # batch_size, num_proposals, 1

        # unpack outputs from language branch
        lang_feat = data_dict["lang_emb"] # batch_size, lang_size
        if self.feedcotpath:
            lang_feat = self.lang_cot_path_fusion(torch.cat([data_dict["lang_emb"], data_dict["lang_cot_path_emb"]], dim=-1))
        lang_feat = lang_feat.unsqueeze(1).repeat(1, self.num_proposals, 1) # batch_size, num_proposals, lang_size

        # fuse
        features = torch.cat([features, lang_feat], dim=-1) # batch_size, num_proposals, 128 + lang_size
        features = features.permute(0, 2, 1).contiguous() # batch_size, 128 + lang_size, num_proposals

        # fuse features
        features = self.fuse(features) # batch_size, hidden_size, num_proposals
        
        # mask out invalid proposals
        objectness_masks = objectness_masks.permute(0, 2, 1).contiguous() # batch_size, 1, num_proposals
        features = features * objectness_masks

        # match
        if self.anchors == "cot":
            AUX_LOGITS = None
            agg_feats = features.permute(0, 2, 1)  # [N, num_cls, E]
            if self.cot_type == "cross":
                parallel_embd = self.parallel_embedding(agg_feats)  # [N, num_cls, E] --> [N, num_cls, E]
                AUX_LOGITS = self.object_language_clf_parallel(parallel_embd)  # [N, num_cls, E] --> [N, num_cls, anchors_length+1]
                AUX_LOGITS = AUX_LOGITS.permute(0, 2, 1)  # [N, num_cls, anchors_length+1] --> [N, anchors_length+1, num_cls]
                trg_pass = torch.argmax(AUX_LOGITS, dim=-1)  # [N, trg_seq_length]
                sampled_embd = vector_gather(parallel_embd, trg_pass)  # [N, num_cls, E] --> [N, anchors_length+1, E]
                cot_out = self.object_language_clf(agg_feats.permute(1, 0, 2), sampled_embd.permute(1, 0, 2)).permute(1, 0, 2)  # [N, anchors_length+1, E]
                confidences = self.fc_out(cot_out).permute(0, 2, 1)  # [N, anchors_length+1, E] --> [N, anchors_length+1, num_cls]
            elif self.cot_type == "self_cons":
                cot_out = self.object_language_clf(agg_feats.permute(1, 0, 2), agg_feats.permute(1, 0, 2))  # [num_cls, N, E]
                confidences = self.fc_out(cot_out).permute(1, 2, 0)  # [num_cls, N, E] --> [num_cls, N, anchors_length+1] --> [N, anchors_length+1, num_cls]
            data_dict["cluster_ref_aux"] = AUX_LOGITS
        elif self.anchors == "parallel":
            confidences = self.match(features)
            data_dict["cluster_ref_aux"] = None
        else:
            confidences = self.match(features).squeeze(1)  # batch_size, num_proposals
                
        data_dict["cluster_ref"] = confidences

        return data_dict
