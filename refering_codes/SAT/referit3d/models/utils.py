import torch
import torch.nn.functional as F
import numpy as np
import random
import einops

def pad_matched_indx(matched_indices, max_matched_obj_len):
    dummy = np.arange(52)
    for i in range(len(matched_indices)):  # loop on batch size (B)
        if len(matched_indices[i]) < max_matched_obj_len:
            # we need to pad it:
            l = [x for x in dummy if x not in matched_indices[i]]
            matched_indices[i] += random.sample(l, max_matched_obj_len-len(matched_indices[i]))

    return matched_indices, max_matched_obj_len


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

def find_matching_indices(array1, array2, device):
    """
    array1: is the array that contains the indices
    array2: is the array that we will sample from it.
    return: list of list
    """
    all_matching_indices = []
    max_matched_obj_len = 0
    for i in range(array1.shape[0]):  # loop on batch size (B)
        matching_indices = []
        for j in range(array2.shape[1]):  # loop on objects (52)
            if array2[i][j] in array1[i]:
                matching_indices.append(j)
        
        if len(matching_indices) == 0:
            return None, 0
        elif len(matching_indices) > max_matched_obj_len:
            max_matched_obj_len = len(matching_indices)

        all_matching_indices.append(matching_indices)
    
    all_matching_indices, max_matched_obj_len = pad_matched_indx(all_matching_indices, max_matched_obj_len)
    all_matching_indices = torch.from_numpy(np.array(all_matching_indices)).to(device).to(torch.int64)
    return all_matching_indices, max_matched_obj_len

def get_siamese_features(net, in_features, aggregator=None):
    """ Applies a network in a siamese way, to 'each' in_feature independently
    :param net: nn.Module, Feat-Dim to new-Feat-Dim
    :param in_features: B x  N-objects x Feat-Dim
    :param aggregator, (opt, None, torch.stack, or torch.cat)
    :return: B x N-objects x new-Feat-Dim
    """
    independent_dim = 1
    n_items = in_features.size(independent_dim)
    out_features = []
    for i in range(n_items):
        out_features.append(net(in_features[:, i]))
    if aggregator is not None:
        out_features = aggregator(out_features, dim=independent_dim)
    return out_features


def save_state_dicts(checkpoint_file, epoch=None, **kwargs):
    """Save torch items with a state_dict.
    """
    checkpoint = dict()

    if epoch is not None:
        checkpoint['epoch'] = epoch

    for key, value in kwargs.items():
        checkpoint[key] = value.state_dict()

    torch.save(checkpoint, checkpoint_file)


def load_state_dicts(checkpoint_file, map_location=None, **kwargs):
    """Load torch items from saved state_dictionaries.
    """
    if map_location is None:
        checkpoint = torch.load(checkpoint_file)
    else:
        checkpoint = torch.load(checkpoint_file, map_location=map_location)

    for key, value in kwargs.items():
        value.load_state_dict(checkpoint[key])

    epoch = checkpoint.get('epoch')
    if epoch:
        return epoch
