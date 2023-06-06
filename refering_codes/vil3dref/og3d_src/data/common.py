import torch
import random
from einops import repeat


relation_synonyms = {
    "near": ["near", "near to", "close", "closer to", "close to", "besides", "by", "next to", "towards", "along", "alongside", "with"],
    "front": ["opposite", "opposite to", "opposite of", "opposite from", "in front of", "faces", "facing"],
    "far": ["farther", "far from", "farthest from", "farthest", "far", "far away from"],
    "on": ["atop of", "above", "on top", "on top of", "on", "higher", "over", "lying on", "onto"],
    "down": ["below", "down", "beneath", "underneath", "lower", "under", "beneath"],
    "right": ["right on", "right of", "right", "to the right of", "right most", "on the right side of", "on the right of", "right"],
    "left": ["left on", "left of", "left", "on the left of", "on the left side of", "left most", "to the left of", "left"],
    "back": ["beyond", "back", "behind", "on the back of"],
}


def gen_seq_masks(seq_lens, max_len=None):
    """
    Args:
        seq_lens: torch.LongTensor, shape=(N, )
    Returns:
        masks: torch.BoolTensor, shape=(N, L), padded=0
    """
    if max_len is None:
        max_len = max(seq_lens)
    batch_size = len(seq_lens)
    seq_masks = repeat(torch.arange(max_len).long(), 'l -> b l', b=batch_size)
    seq_masks = seq_masks  < seq_lens.unsqueeze(1)
    return seq_masks

def pad_tensors(tensors, lens=None, pad=0, pad_ori_data=False):
    """B x [T, ...] torch tensors"""
    if lens is None:
        lens = [t.size(0) for t in tensors]
    max_len = max(lens)
    bs = len(tensors)
    hid = list(tensors[0].size()[1:])
    size = [bs, max_len] + hid

    dtype = tensors[0].dtype
    output = torch.zeros(*size, dtype=dtype)
    if pad:
        output.data.fill_(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output.data[i, :l, ...] = t.data
        if pad_ori_data:
            rt = (max_len - l) // l + 1
            for j in range(rt):
                s = l + j * l
                e = min(s + l, max_len)
                output.data[i, s: e] = t.data[:e-s]
    return output


def flipcoin(percent=50):
    """
    return Treu or False based on the given percentage.
    """
    return random.randrange(100) < percent