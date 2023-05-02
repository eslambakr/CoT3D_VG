import sng_parser
import stanza
from nltk.tree import Tree
nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
import pdb
import json
from itertools import islice
import numpy as np

def preprocess_prompts(prompts):
    if isinstance(prompts, (list, tuple)):
        return [p.lower().strip().strip(".").strip() for p in prompts]
    elif isinstance(prompts, str):
        return prompts.lower().strip().strip(".").strip()
    else:
        raise NotImplementedError


def get_all_nps(tree, full_sent, tokens=None, highest_only=False, lowest_only=False):
    start = 0
    end = len(tree.leaves())

    idx_map = get_token_alignment_map(tree, tokens)

    def get_sub_nps(tree, left, right):
        if isinstance(tree, str) or len(tree.leaves()) == 1:
            return []
        sub_nps = []
        n_leaves = len(tree.leaves())
        n_subtree_leaves = [len(t.leaves()) for t in tree]
        offset = np.cumsum([0] + n_subtree_leaves)[:len(n_subtree_leaves)]
        assert right - left == n_leaves
        if tree.label() == 'NP' and n_leaves > 1:
            sub_nps.append([" ".join(tree.leaves()), (int(min(idx_map[left])), int(min(idx_map[right])))])
            if highest_only and sub_nps[-1][0] != full_sent: return sub_nps
        for i, subtree in enumerate(tree):
            sub_nps += get_sub_nps(subtree, left=left+offset[i], right=left+offset[i]+n_subtree_leaves[i])
        return sub_nps
    
    all_nps = get_sub_nps(tree, left=start, right=end)
    lowest_nps = []
    for i in range(len(all_nps)):
        span = all_nps[i][1]
        lowest = True
        for j in range(len(all_nps)):
            if i == j: continue
            span2 = all_nps[j][1]
            if span2[0] >= span[0] and span2[1] <= span[1]:
                lowest = False
                break
        if lowest:
            lowest_nps.append(all_nps[i])

    if lowest_only:
        all_nps = lowest_nps

    if len(all_nps) == 0:
        all_nps = []
        spans = []
    else:
        all_nps, spans = map(list, zip(*all_nps))
    if full_sent not in all_nps:
        all_nps = [full_sent] + all_nps
        spans = [(min(idx_map[start]), min(idx_map[end]))] + spans

    return all_nps, spans, lowest_nps


def get_token_alignment_map(tree, tokens):
    if tokens is None:
        return {i:[i] for i in range(len(tree.leaves())+1)}
        
    def get_token(token):
        return token[:-4] if token.endswith("</w>") else token

    idx_map = {}
    j = 0
    max_offset = np.abs(len(tokens) - len(tree.leaves()))
    mytree_prev_leaf = ""
    for i, w in enumerate(tree.leaves()):
        token = get_token(tokens[j])
        idx_map[i] = [j]
        if token == mytree_prev_leaf+w:
            mytree_prev_leaf = ""
            j += 1
        else:
            if len(token) < len(w):
                prev = ""
                while prev + token != w:
                    prev += token
                    j += 1
                    token = get_token(tokens[j])
                    idx_map[i].append(j)
                    # assert j - i <= max_offset
            else:
                mytree_prev_leaf += w
                j -= 1
            j += 1
    idx_map[i+1] = [j]
    return idx_map


def get_all_spans_from_scene_graph(caption):
    caption = caption.strip()
    graph = sng_parser.parse(caption)
    nps = []
    spans = []
    words = caption.split()
    for e in graph['entities']:
        start, end = e['span_bounds']
        if e['span'] == caption: continue
        if end-start == 1: continue
        nps.append(e['span'])
        spans.append(e['span_bounds'])
    for r in graph['relations']:
        start1, end1 = graph['entities'][r['subject']]['span_bounds']
        start2, end2 = graph['entities'][r['object']]['span_bounds']
        start = min(start1, start2)
        end = max(end1, end2)
        if " ".join(words[start:end]) == caption: continue
        nps.append(" ".join(words[start:end]))
        spans.append((start, end))
    
    return [caption] + nps, [(0, len(words))] + spans, None


def single_align(main_seq, seqs, spans, dim=1):
    main_seq = main_seq.transpose(0, dim)
    for seq, span in zip(seqs, spans):
        seq = seq.transpose(0, dim)
        start, end = span[0]+1, span[1]+1
        seg_length = end - start
        main_seq[start:end] = seq[1:1+seg_length]

    return main_seq.transpose(0, dim)


def multi_align(main_seq, seq, span, dim=1):
    seq = seq.transpose(0, dim)
    main_seq = main_seq.transpose(0, dim)
    start, end = span[0]+1, span[1]+1
    seg_length = end - start

    main_seq[start:end] = seq[1:1+seg_length]

    return main_seq.transpose(0, dim)


def align_sequence(main_seq, seqs, spans, dim=1, single=False):
    aligned_seqs = []
    if single:
        return [single_align(main_seq, seqs, spans, dim=dim)]
    else:
        for seq, span in zip(seqs, spans):
            aligned_seqs.append(multi_align(main_seq.clone(), seq, span, dim=dim))
        return aligned_seqs


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


# prompts = "facing the window the arm chair in the upper right corner."
prompts = "the monitor that is upper the printer"

prompts = preprocess_prompts(prompts)

print("prompts: ", prompts)

# Parser Type
parser_type = 'scene_graph'
if parser_type == 'constituency':
                            doc = nlp(prompts[0])
                            mytree = Tree.fromstring(str(doc.sentences[0].constituency))
                            tokens = model.cond_stage_model.tokenizer.tokenize(prompts[0])
                            nps, spans, noun_chunk = get_all_nps(mytree, prompts[0])
elif parser_type == 'scene_graph':
    nps, spans, noun_chunk = get_all_spans_from_scene_graph(prompts[0].split("\t")[0])
else:
    raise NotImplementedError

#conjunction

print("nps: ", nps)
print("spans: ", spans)
print("noun_chunk: ", noun_chunk)