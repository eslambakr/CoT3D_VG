import pathlib
import os.path as osp
import pandas as pd
import numpy as np
from ast import literal_eval

from .vocabulary import build_vocab, Vocabulary
from .pt_datasets.utils import get_logical_pth_lang, clean_paraphrased
from ..utils import read_lines, unpickle_data
from ..data_generation.nr3d import decode_stimulus_string
from .pt_datasets.listening_dataset import get_consecutive_identical_elements


def scannet_official_train_val(valid_views=None, verbose=True):
    """
    :param valid_views: None or list like ['00', '01']
    :return:
    """
    pre_fix = osp.split(pathlib.Path(__file__).parent.absolute())[0]
    train_split = osp.join(pre_fix, 'data/scannet/splits/official/v2/scannetv2_train.txt')
    train_split = read_lines(train_split)
    test_split = osp.join(pre_fix, 'data/scannet/splits/official/v2/scannetv2_val.txt')
    test_split = read_lines(test_split)

    if valid_views is not None:
        train_split = [sc for sc in train_split if sc[-2:] in valid_views]
        test_split = [sc for sc in test_split if sc[-2:] in valid_views]

    if verbose:
        print('#train/test scans:', len(train_split), '/', len(test_split))

    scans_split = dict()
    scans_split['train'] = set(train_split)
    scans_split['test'] = set(test_split)
    return scans_split


def objects_counter_percentile(scan_ids, all_scans, prc):
    all_obs_len = list()
    for scan_id in all_scans:
        if scan_id in scan_ids:
            all_obs_len.append(len(all_scans[scan_id].three_d_objects))
    return np.percentile(all_obs_len, prc)


def mean_color(scan_ids, all_scans):
    mean_rgb = np.zeros((1, 3), dtype=np.float32)
    n_points = 0
    for scan_id in scan_ids:
        color = all_scans[scan_id].color
        mean_rgb += np.sum(color, axis=0)
        n_points += len(color)
    mean_rgb /= n_points
    return mean_rgb


def load_scanrefer_referential_data(args, referit_csv, scans_split):
    """
    :param args:
    :param referit_csv:
    :param scans_split:
    :return:
    """
    # Load the two files (train and val):
    scanrefer_data_train = pd.read_csv(referit_csv)
    print("The length of the loaded training file is : ", len(scanrefer_data_train))
    scanrefer_data_val = pd.read_csv(referit_csv.replace('train', 'val'))
    print("The length of the loaded val file is : ", len(scanrefer_data_val))

    # Create the new split:
    scans_split['train'] =  set(scanrefer_data_train['scene_id'])
    scans_split['test'] =  set(scanrefer_data_val['scene_id'])
    print("The length of the training scenes is : ", len(scans_split['train']))
    print("The length of the val scenes is : ", len(scans_split['test']))

    # Merge the two files:
    scanrefer_data = pd.concat([scanrefer_data_train, scanrefer_data_val], axis=0).reset_index()
    print("The length of the loaded train & val files together is : ", len(scanrefer_data))

    # Rename the columns to match the Referit names:
    scanrefer_data.rename(columns = {'scene_id':'scan_id'}, inplace = True)
    scanrefer_data.rename(columns = {'object_id':'target_id'}, inplace = True)
    scanrefer_data.rename(columns = {'description':'utterance'}, inplace = True)
    scanrefer_data.rename(columns = {'token':'tokens'}, inplace = True)
    scanrefer_data.rename(columns = {'object_name':'instance_type'}, inplace = True)

    # Clean path, tokens and anchor_ids --> convert them to list:
    scanrefer_data.path = scanrefer_data['path'].apply(literal_eval)
    scanrefer_data.anchor_ids = scanrefer_data['anchor_ids'].apply(literal_eval)
    scanrefer_data.tokens = scanrefer_data['tokens'].apply(literal_eval)

    # Drop the last item as we add the tgt at the end by mistake.
    scanrefer_data.anchor_ids = scanrefer_data['anchor_ids'].apply(lambda x: x[:-1])

    # Remove duplicated anchors from the path and sample anchor_id to assign it:
    if args.remove_repeated_anchors:
        filtered_path = []
        filtered_anchor_ids = []
        anchors_id_temp = []
        anchors_id_temp_all_data = scanrefer_data['anchor_ids'].values
        for i, path in enumerate(scanrefer_data['path']):  # Loop on each example
            anchors_path = path[:-1]  # exclude the target for now
            # print("anchors_path = ", anchors_path)  ['cabinet', 'cabinet', 'door', 'desk', 'table'] ['39', '54', '7', '39', '35']
            anchors_id_temp = anchors_id_temp_all_data[i]
            anchors_id_temp = [int(A) for A in anchors_id_temp]
            assert len(anchors_path) == len(anchors_id_temp)
            anchor_name_id_list = []
            for j, anchor_id in enumerate(anchors_id_temp):
                anchor_name_id_list.append((anchors_path[j], anchor_id))
            # e.g.: [('nightstand', 20), ('bed', 14), ('bed', 13)] --> [('nightstand', [20]), ('bed', [14, 13])]
            #print("Before --> ", anchor_name_id_list)
            anchor_name_id_list = get_consecutive_identical_elements(anchor_name_id_list)
            #print("After --> ", anchor_name_id_list) 
            # Sample the first anchor_id if there are more than one, A.K.A remove duplicates:
            for j in range(len(anchor_name_id_list)):
                anchor_name_id_list[j] = (anchor_name_id_list[j][0], anchor_name_id_list[j][1][0])  # sample the first one
            
            # Save the filtered path and anchors_ids:
            anchors_path = [dumy_item[0] for dumy_item in anchor_name_id_list]
            filtered_path.append(anchors_path + [path[-1]])
            filtered_anchor_ids.append([dumy_item[1] for dumy_item in anchor_name_id_list])

        scanrefer_data['path'] = pd.Series(filtered_path)
        scanrefer_data['anchor_ids'] = pd.Series(filtered_anchor_ids)

    # Add the 'num_anchors' data to the pandas data frame
    num_anchors = scanrefer_data['anchor_ids'].apply(lambda x: len(x))
    scanrefer_data['num_anchors'] = num_anchors
    print("------------- Max number of anchors: ", max(num_anchors))

    keys = ['tokens', 'instance_type', 'scan_id', 'target_id', 'utterance', 'path', 'anchor_ids', 'num_anchors']
    scanrefer_data = scanrefer_data[keys]

    # Add the is_train data to the pandas data frame (needed in creating data loaders for the train and test)
    is_train = scanrefer_data.scan_id.apply(lambda x: x in scans_split['train'])
    scanrefer_data['is_train'] = is_train

    # Add the 'dataset'
    data_name = ['nr3d']*len(is_train)
    scanrefer_data['dataset'] = pd.Series(data_name)

    # Trim data based on token length
    train_token_lens = scanrefer_data.tokens[is_train].apply(lambda x: len(x))
    print('{}-th percentile of token length for remaining (training) data'
          ' is: {:.1f}'.format(95, np.percentile(train_token_lens, 95)))
    n_original = len(scanrefer_data)
    scanrefer_data = scanrefer_data[scanrefer_data.tokens.apply(lambda x: len(x) <= args.max_seq_len)]
    scanrefer_data.reset_index(drop=True, inplace=True)
    print('Dropping utterances with more than {} tokens, {}->{}'.format(args.max_seq_len, n_original, len(scanrefer_data)))

    return scanrefer_data, scans_split


def load_referential_data(args, referit_csv, scans_split):
    """
    :param args:
    :param referit_csv:
    :param scans_split:
    :return:
    """
    referit_data = pd.read_csv(referit_csv)

    is_nr = True if 'nr' in args.referit3D_file else False
    if is_nr and (args.anchors != 'none'):
        if args.anchors_ids_type == "pseudoWneg" or args.anchors_ids_type == "pseudoWneg_old" or args.anchors_ids_type == "ourPathGTids":
            anchor_path_name = "path"
        elif args.anchors_ids_type == "pseudoWOneg":
            anchor_path_name = "our_neg_anchor_names"
        elif args.anchors_ids_type == "GT":
            anchor_path_name = "true_gt_anchor_names"
        referit_data = get_logical_pth_lang(referit_data, key=anchor_path_name)
    
    if args.textaug_paraphrase_percentage:
        referit_data = clean_paraphrased(referit_data)

    if args.mentions_target_class_only:
        n_original = len(referit_data)
        referit_data = referit_data[referit_data['mentions_target_class']]
        referit_data.reset_index(drop=True, inplace=True)
        print('Dropping utterances without explicit '
              'mention to the target class {}->{}'.format(n_original, len(referit_data)))

    keys = ['tokens', 'instance_type', 'scan_id', 'dataset', 'target_id', 'utterance', 'stimulus_id']
    if is_nr:
        added_keys = ['path', 'anchor_ids', 'num_anchors', 'paraphrases']
        if args.anchors_ids_type == "pseudoWOneg":
            added_keys += ['our_neg_anchor_names', 'ours_with_neg_ids']
        elif args.anchors_ids_type == "ourPathGTids":
            added_keys += ['our_gt_id']
        elif args.anchors_ids_type == "GT":
            keys = ['tokens', 'instance_type', 'scan_id', 'dataset', 'target_id', 'utterance', 'stimulus_id', 'true_gt_anchor_names', 'true_gt_id']
            added_keys = []
    else:
        added_keys = ['anchors_types', 'anchor_ids']
    keys += added_keys
    referit_data = referit_data[keys]
    referit_data.tokens = referit_data['tokens'].apply(literal_eval)

    # Add the is_train data to the pandas data frame (needed in creating data loaders for the train and test)
    is_train = referit_data.scan_id.apply(lambda x: x in scans_split['train'])
    referit_data['is_train'] = is_train

    # Trim data based on token length
    train_token_lens = referit_data.tokens[is_train].apply(lambda x: len(x))
    print('{}-th percentile of token length for remaining (training) data'
          ' is: {:.1f}'.format(95, np.percentile(train_token_lens, 95)))
    n_original = len(referit_data)
    referit_data = referit_data[referit_data.tokens.apply(lambda x: len(x) <= args.max_seq_len)]
    referit_data.reset_index(drop=True, inplace=True)
    print('Dropping utterances with more than {} tokens, {}->{}'.format(args.max_seq_len, n_original, len(referit_data)))

    # do this last, so that all the previous actions remain unchanged
    if args.augment_with_sr3d is not None:
        print('Adding Sr3D as augmentation.')
        sr3d = pd.read_csv(args.augment_with_sr3d)
        sr3d.tokens = sr3d['tokens'].apply(literal_eval)
        is_train = sr3d.scan_id.apply(lambda x: x in scans_split['train'])
        sr3d['is_train'] = is_train
        sr3d = sr3d[is_train]
        # Eslam:
        keys = ['tokens', 'instance_type', 'scan_id', 'dataset', 'target_id', 'utterance', 'stimulus_id', 'anchors_types', 'anchor_ids', 'is_train']
        sr3d = sr3d[keys]
        # sr3d = sr3d[referit_data.columns]
        print('Dataset-size before augmentation:', len(referit_data))
        referit_data = pd.concat([referit_data, sr3d], axis=0)
        referit_data.reset_index(inplace=True, drop=True)
        print('Dataset-size after augmentation:', len(referit_data))

    context_size = referit_data[~referit_data.is_train].stimulus_id.apply(lambda x: decode_stimulus_string(x)[2])
    print('(mean) Random guessing among target-class test objects {:.4f}'.format( (1 / context_size).mean() ))

    return referit_data


def load_scan_related_data(preprocessed_scannet_file, verbose=True, add_pad=True, add_no_obj=False):
    _, all_scans = unpickle_data(preprocessed_scannet_file)
    if verbose:
        print('Loaded in RAM {} scans'.format(len(all_scans)))

    instance_labels = set()
    for scan in all_scans:
        idx = np.array([o.object_id for o in scan.three_d_objects])
        instance_labels.update([o.instance_label for o in scan.three_d_objects])
        assert np.all(idx == np.arange(len(idx)))  # assert the list of objects-ids -is- the range(n_objects).
                                                   # because we use this ordering when we sample objects from a scan.
    all_scans = {scan.scan_id: scan for scan in all_scans}  # place scans in dictionary


    class_to_idx = {}
    i = 0
    for el in sorted(instance_labels):
        class_to_idx[el] = i
        i += 1

    if verbose:
        print('{} instance classes exist in these scans'.format(len(class_to_idx)))

    # Add the pad class needed for object classification
    if add_pad:
        class_to_idx['pad'] = len(class_to_idx)

    if add_no_obj:
        class_to_idx['no_obj'] = len(class_to_idx)

    scans_split = scannet_official_train_val()

    return all_scans, scans_split, class_to_idx


def compute_auxiliary_data(referit_data, all_scans, args):
    """Given a train-split compute useful quantities like mean-rgb, a word-vocabulary.
    :param referit_data: pandas Dataframe, as returned from load_referential_data()
    :param all_scans:
    :param args:
    :return:
    """
    # Vocabulary
    if args.vocab_file:
        vocab = Vocabulary.load(args.vocab_file)
        print(('Using external, provided vocabulary with {} words.'.format(len(vocab))))
    else:
        train_tokens = referit_data[referit_data.is_train]['tokens']
        vocab = build_vocab([x for x in train_tokens], args.min_word_freq)
        print(('Length of vocabulary, with min_word_freq={} is {}'.format(args.min_word_freq, len(vocab))))

    if all_scans is None:
        return vocab

    # Mean RGB for the training
    training_scan_ids = set(referit_data[referit_data['is_train']]['scan_id'])
    print('{} training scans will be used.'.format(len(training_scan_ids)))
    mean_rgb = mean_color(training_scan_ids, all_scans)

    # Percentile of number of objects in the training data
    prc = 90
    obj_cnt = objects_counter_percentile(training_scan_ids, all_scans, prc)
    print('{}-th percentile of number of objects in the (training) scans'
          ' is: {:.2f}'.format(prc, obj_cnt))

    # Percentile of number of objects in the testing data
    prc = 99
    testing_scan_ids = set(referit_data[~referit_data['is_train']]['scan_id'])
    obj_cnt = objects_counter_percentile(testing_scan_ids, all_scans, prc)
    print('{}-th percentile of number of objects in the (testing) scans'
          ' is: {:.2f}'.format(prc, obj_cnt))
    return mean_rgb, vocab


def trim_scans_per_referit3d_data(referit_data, scans):
    # remove scans not in referit_data
    in_r3d = referit_data.scan_id.unique()
    to_drop = []
    for k in scans:
        if k not in in_r3d:
            to_drop.append(k)
    for k in to_drop:
        del scans[k]
    print('Dropped {} scans to reduce mem-foot-print.'.format(len(to_drop)))
    return scans


##
## Are below necessary? Refactor. Maybe send to a _future_ package
## I think I wrote them to extract the classes of only the training data, but we rejected this idea.
##

# def object_classes_of_scans(scan_ids, all_scans, verbose=False):
#     """ get the object classes (e.g., chair, table...) that the specified scans contain.
#     :param scan_ids: a list of strings
#     :param all_scans: a dictionary holding ScannetScan objects
#     :return: a dictionary mapping all present object classes (string) to a unique int
#     """
#     object_classes = set()
#     for scan_id, scan in all_scans.items():
#         if scan_id in scan_ids:
#             object_classes.update([s.instance_label for s in scan.three_d_objects])
#
#     if verbose:
#         print('{} object classes were found.'.format(len(object_classes)))
#     return object_classes
#
#
# def object_class_to_idx_dictionary(object_classes, add_pad=False):
#     class_to_idx = {m: i for i, m in enumerate(sorted(list(object_classes)))}
#     if add_pad:
#         class_to_idx['pad'] = len(class_to_idx)
#     return class_to_idx
