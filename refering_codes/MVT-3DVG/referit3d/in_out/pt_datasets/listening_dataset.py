import numpy as np
import h5py
import os
import torch
import pandas as pd
import json
from random import sample
import torchvision.transforms as T
from torch.utils.data import Dataset
from functools import partial
from .utils import dataset_to_dataloader, max_io_workers, relation_synonyms
from .pc_transforms import ChromaticTranslation, RandomSymmetry, RandomNoise, Random3AxisRotation

# the following will be shared on other datasets too if not, they should become part of the ListeningDataset
# maybe make SegmentedScanDataset with only static functions and then inherit.
from .utils import check_segmented_object_order, sample_scan_object, pad_samples, objects_bboxes
from .utils import instance_labels_of_context, mean_rgb_unit_norm_transform, flipcoin
from ...data_generation.nr3d import decode_stimulus_string
from transformers import DistilBertTokenizer, DistilBertModel
from ..three_d_object import ThreeDObject


class ListeningDataset(Dataset):
    def __init__(self, references, scans, vocab, max_seq_len, points_per_object, max_distractors,
                 class_to_idx=None, object_transformation=None, visualization=False,
                 anchors_mode="cot", max_anchors=2, predict_lang_anchors=False, 
                 shuffle_objects=None, pc_transforms=None, textaug_paraphrase_percentage=None, shuffle_objects_percentage=None,
                 target_aug_percentage=None, is_train=None, distractor_aux_loss_flag=False):

        self.references = references
        self.scans = scans
        self.max_seq_len = max_seq_len
        self.points_per_object = points_per_object
        self.max_distractors = max_distractors
        self.max_context_size = self.max_distractors + 1  # to account for the target.
        self.class_to_idx = class_to_idx
        self.visualization = visualization
        self.object_transformation = object_transformation
        self.anchors_mode = anchors_mode
        self.predict_lang_anchors = predict_lang_anchors
        self.shuffle_objects = shuffle_objects
        self.shuffle_objects_percentage = shuffle_objects_percentage
        self.pc_transforms = pc_transforms
        self.textaug_paraphrase_percentage = textaug_paraphrase_percentage
        self.target_aug_percentage = target_aug_percentage
        self.distractor_aux_loss_flag = distractor_aux_loss_flag
        self.is_train = is_train
        with open('/home/abdelrem/3d_codes/CoT3D_VG/extract_anchors/unique_rel_map_dict_opposite.json') as f:
            self.opposite_dict = json.load(f)
        if self.anchors_mode != 'none' or self.predict_lang_anchors:
            self.max_anchors = max_anchors
        else:
            self.max_anchors = 0
        if not check_segmented_object_order(scans):
            raise ValueError

    def __len__(self):
        return len(self.references)

    def get_anchor_ids(self, anchor_ids):
        """
        Convert anchor ids from string to int
        """
        #print("anchor_ids = ", anchor_ids)
        # Check if anchor_ids is in the format "[3, 26]"
        if anchor_ids.startswith('[') and anchor_ids.endswith(']'):
            # Extract the numbers from the string
            numbers = anchor_ids[1:-1].split(',')
            # Convert the numbers to integers
            anchor_ids = []
            for num in numbers:
                if num != '':
                    anchor_ids.append(int(num.strip()))
        else:
            # If anchor_ids is not in the expected format, assume it's a single number
            anchor_ids = [int(anchor_ids)]
        return anchor_ids

    def get_reference_data(self, index):
        target_augmented_flag = False
        ref = self.references.loc[index]
        scan_id = ref['scan_id']
        scan = self.scans[ref['scan_id']]
        target = scan.three_d_objects[ref['target_id']]
        # Get Anchors
        anchors = None
        path = None
        if self.anchors_mode != 'none' or self.predict_lang_anchors:
            path = ref['path']
            self.anchors_ids = self.get_anchor_ids(ref['anchor_ids'])
            if flipcoin(self.target_aug_percentage) and (len(path)==2) and self.is_train and (type(ref['relation'])==str):  # swap target with anchor
                path.reverse()
                target = scan.three_d_objects[self.anchors_ids[0]]  # [0] as we are sure it is only one anchor
                self.anchors_ids = [ref['target_id']]
                #mask = self.unique_rel_df[self.unique_rel_df.utterance.str.lower().isin([ref['utterance']])]
                relation = sample(relation_synonyms[self.opposite_dict[ref['relation']]], 1)[0]
                #relation = sample(relation_synonyms[self.opposite_dict[mask.relation.values[0]]], 1)[0]
                tokens = path[-1] + " " + relation + " " + path[0]
                print(ref['relation'])
                target_augmented_flag = True

            anchors = []
            for anchor_id in self.anchors_ids:
                if anchor_id == -1:  # handle the case where we don't have a GT box for the class (For Nr3D only)
                    #anchors.append(-1)
                    continue
                anchors.append(scan.three_d_objects[anchor_id])
            
            # Handle the case where we set max_anchors number to low number (< true path) --> Trim the path:
            if len(self.anchors_ids) > self.max_anchors:
                self.anchors_ids = self.anchors_ids[:self.max_anchors]
                # TODO: Eslam: but here the path not matching the anchors. Should I exclude the *?
                trimmed_path = path[:self.max_anchors]  # Get max_num of anchor
                trimmed_path.append(path[-1])  # Adding the target 
                path = trimmed_path
                anchors = anchors[:self.max_anchors]

            self.anchors_len = len(anchors)+1 if len(anchors) < self.max_anchors else len(anchors)  # +1 as we will add only one empty anchor later

        # sega_update: 使用原始的token
        #tokens = np.array(self.vocab.encode(ref['tokens'], self.max_seq_len), dtype=np.long)
        if not target_augmented_flag:
            if self.textaug_paraphrase_percentage and self.is_train:
                if flipcoin(percent=self.textaug_paraphrase_percentage) and (len(ref['paraphrases'])):
                    tokens = sample(ref['paraphrases'], 1)[0]
                else:
                    ori_tokens = ref['tokens']
                    tokens = " ".join(ori_tokens)
            else:
                ori_tokens = ref['tokens']
                tokens = " ".join(ori_tokens)
        # tokens = self.vocab(sen).input_ids
        # print(len(tokens))
        # tokens = np.array(tokens)
        # tokens = np.array([102]*(self.max_seq_len + 2 + self.max_context_size * 2))
        # tokens[:min(self.max_seq_len + 2, len(emb))] = emb[:min(self.max_seq_len + 2, len(emb))]
        is_nr3d = ref['dataset'] == 'nr3d'

        return scan, target, tokens, is_nr3d, scan_id, anchors, path

    def prepare_distractors(self, scan, target, anchors):
        target_label = target.instance_label

        # First add all objects with the same instance-label as the target
        distractors = [o for o in scan.three_d_objects if
                       (o.instance_label == target_label and (o != target))]
        already_included = [target_label]

        # Add anchors' distractors:
        if self.anchors_mode != 'none' or self.predict_lang_anchors:
            anchor_labels = [anchor.instance_label for anchor in anchors]
            anchors_distractors = []
            for anchor in anchors:
                anchor_distractors = [o for o in scan.three_d_objects if
                                      (o.instance_label == anchor.instance_label and (o != anchor))]
                if len(anchor_distractors):
                    already_included.append(anchor.instance_label)
                anchors_distractors = anchors_distractors + anchor_distractors
            distractors = distractors + anchors_distractors

        # Then all more objects up to max-number of distractors
        clutter = [o for o in scan.three_d_objects if o.instance_label not in already_included]
        np.random.shuffle(clutter)

        distractors.extend(clutter)
        distractors = distractors[:(self.max_distractors - self.anchors_len)]
        np.random.shuffle(distractors)

        return distractors

    def __getitem__(self, index):
        res = dict()
        scan, target, tokens, is_nr3d, scan_id, anchors, path = self.get_reference_data(index)  
        # Make a context of distractors
        context = self.prepare_distractors(scan, target, anchors)
            
        if self.anchors_mode != 'none' or self.predict_lang_anchors:
            # "replace" is false to make sure the positions are unique
            poses = np.random.choice(range(len(context) + 1), 1+self.anchors_len, replace=False)  # +1 for the target
            poses.sort()
            # Add target object in 'context' list
            target_pos = poses[0]
            context.insert(target_pos, target)
            # Add anchors in 'context' list
            anchors_pos = poses[1:]
            #print("---- anchors_pos = ", anchors_pos)
            
            for anchor_i, anchor in enumerate(anchors):
                context.insert(anchors_pos[anchor_i], anchor)
            # pad with dummy anchor which will be replaced by zeros later.
            if len(anchors) < self.anchors_len:
                context.insert(anchors_pos[-1], ThreeDObject(context[0].scan, context[0].object_id, context[0].points, context[0].instance_label))
                context[anchors_pos[-1]].instance_label = 'no_obj'
        else:
            # Add target object in 'context' list
            target_pos = np.random.randint(len(context) + 1)
            anchors_pos = None
            context.insert(target_pos, target)

        res['class_labels'] = instance_labels_of_context(context, self.max_context_size, self.class_to_idx)
        # sample point/color for them
        context_len = len(context)
        if (self.shuffle_objects is not None) and (flipcoin(percent=self.shuffle_objects_percentage)) and self.is_train:  # Shuffling objects optionally
            # Iterate over object labels
            samples = []
            for k, object_label in enumerate(res['class_labels'][:context_len]):
                class_key = 'objects_class_%d' % object_label
                # sample a random pointcloud
                # from the same object class
                if class_key in self.shuffle_objects:
                    n_objects = len(self.shuffle_objects[class_key])
                if class_key not in self.shuffle_objects or n_objects == 0:
                    samples += [sample_scan_object(context[k], self.points_per_object)]
                    continue
                rnd_index = np.random.randint(n_objects)
                sampled_obj = self.shuffle_objects[class_key][rnd_index]
                assert sampled_obj.shape == (2048, 6)
                # Downsample pointcloud to points_per_object if needed
                if sampled_obj.shape[0] > self.points_per_object:
                    sampled_obj = sampled_obj[np.random.choice(sampled_obj.shape[0],
                                                               self.points_per_object,
                                                               replace=False), :]
                samples += [sampled_obj]
            samples = np.array(samples)
        else:
            samples = np.array([sample_scan_object(o, self.points_per_object) for o in context])

        assert samples.shape == (context_len, self.points_per_object, 6)

        if self.anchors_mode != 'none' or self.predict_lang_anchors:
            # pad with non-existing anchor
            if len(anchors) < self.anchors_len:
                samples[anchors_pos[-1]] = np.zeros((1, samples.shape[1], samples.shape[2]), dtype=samples.dtype)

        # mark their classes
        # res['ori_labels'],
        res['scan_id'] = scan_id
        box_info = np.zeros((self.max_context_size, 4))
        box_info[:len(context),0] = [o.get_bbox().cx for o in context]
        box_info[:len(context),1] = [o.get_bbox().cy for o in context]
        box_info[:len(context),2] = [o.get_bbox().cz for o in context]
        box_info[:len(context),3] = [o.get_bbox().volume() for o in context]
        box_corners = np.zeros((self.max_context_size, 8, 3))
        box_corners[:len(context)] = [o.get_bbox().corners for o in context]
        if self.anchors_mode != 'none' or self.predict_lang_anchors:
            # pad with non-existing anchor
            if len(anchors) < self.anchors_len:
                box_info[anchors_pos[-1]] = np.zeros((1, 4))
                box_corners[anchors_pos[-1]] = np.zeros((1, 8, 3))

        if self.object_transformation is not None:
            samples = self.object_transformation(samples)

        res['context_size'] = len(samples)

        if (self.pc_transforms is not None) and (flipcoin(percent=20)) and self.is_train:
            for sample in samples:
                sample = torch.from_numpy(sample)
                sample = self.pc_transforms(sample)

        # take care of padding, so that a batch has same number of N-objects across scans.
        res['objects'] = pad_samples(samples, self.max_context_size)

        # Get a mask indicating which objects have the same instance-class as the target.
        target_class_mask = np.zeros(self.max_context_size, dtype=np.bool)
        target_class_mask[:len(context)] = [target.instance_label == o.instance_label for o in context]

        res['target_class'] = self.class_to_idx[target.instance_label]
        res['target_pos'] = target_pos
        res['target_class_mask'] = target_class_mask
        # Anchors
        if self.anchors_mode != 'none' or self.predict_lang_anchors:
            # Anchors classes for language branch only
            res['anchor_classes'] = np.zeros(self.max_anchors)
            if is_nr3d:
                for anchor_i in range(len(path)-1):  # -1 exclude the target
                    res['anchor_classes'][anchor_i] = self.class_to_idx[path[anchor_i]]
                # pad with non-existing anchor
                for anchor_i in range(len(path)-1, self.max_anchors):
                    res['anchor_classes'][anchor_i] = self.class_to_idx['no_obj']
            else:
                for anchor_i, anchor in enumerate(anchors):
                    res['anchor_classes'][anchor_i] = self.class_to_idx[anchor.instance_label]
                # pad with non-existing anchor
                for anchor_i in range(len(anchors), self.max_anchors):
                    res['anchor_classes'][anchor_i] = self.class_to_idx['no_obj']
            # Anchors poses for referring head
            anchors_pos_with_no_obj = []
            i = 0
            for anchor_id in self.anchors_ids:
                if anchor_id == -1:
                    anchors_pos_with_no_obj.append(anchors_pos[-1])
                else:
                    anchors_pos_with_no_obj.append(anchors_pos[i])
                    i += 1
            res['anchors_pos'] = np.array(anchors_pos_with_no_obj + [anchors_pos[-1]]*(self.max_anchors-len(self.anchors_ids)))

        if self.distractor_aux_loss_flag:
            res['distractor_mask'] = np.zeros(len(res['objects']))
            for obj_i, o in enumerate(context):
                if (o.instance_label==target.instance_label) and o != target:
                    res['distractor_mask'][obj_i] = 1.0

        res['tokens'] = tokens
        res['is_nr3d'] = is_nr3d
        res['box_info'] = box_info
        res['box_corners'] = box_corners

        if self.visualization:
            distrators_pos = np.zeros((6))  # 6 is the maximum context size we used in dataset collection
            object_ids = np.zeros((self.max_context_size))
            j = 0
            for k, o in enumerate(context):
                if o.instance_label == target.instance_label and o.object_id != target.object_id:
                    distrators_pos[j] = k
                    j += 1
            for k, o in enumerate(context):
                object_ids[k] = o.object_id
            res['utterance'] = self.references.loc[index]['utterance']
            res['stimulus_id'] = self.references.loc[index]['stimulus_id']
            res['distrators_pos'] = distrators_pos
            res['object_ids'] = object_ids
            res['target_object_id'] = target.object_id

        return res


def make_data_loaders(args, referit_data, vocab, class_to_idx, scans, mean_rgb, transform_obj=True):
    max_anchors = args.max_num_anchors if 'nr' in args.referit3D_file else 2
    print("max_anchors = ", max_anchors)
    shuffle_mode = args.visaug_shuffle_mode
    pc_augment = args.visaug_pc_augment
    assert shuffle_mode in ["none", "scannet", "3DCoMPaT"]
    n_workers = args.n_workers
    if n_workers == -1:
        n_workers = max_io_workers()

    data_loaders = dict()
    is_train = referit_data['is_train']
    splits = ['train', 'test']

    pc_transforms = None
    if pc_augment:
        pc_transforms = T.Compose([
            RandomSymmetry(axis=[True, True, False], p=0.05),
            RandomNoise(sigma=0.01, clip=0.05, p=0.05),
            Random3AxisRotation(rot_x=5, rot_y=5, rot_z=5, p=0.05),
            ChromaticTranslation(p=0.05),
        ])

    object_transformation = None
    if transform_obj:
        object_transformation = partial(mean_rgb_unit_norm_transform, mean_rgb=mean_rgb,
                                        unit_norm=args.unit_sphere_norm)

    for split in splits:
        mask = is_train if split == 'train' else ~is_train
        d_set = referit_data[mask]
        if split == 'train':
            extend = 1
            org_training_len = len(d_set)
            if args.target_aug_percentage:
                unique_rel_df = pd.read_csv("/home/abdelrem/3d_codes/CoT3D_VG/extract_anchors/nr3d_cot_unique_rel_anchor_data.csv")
                print("unique_rel_df: ", len(unique_rel_df))
                print("d_set before: ", len(d_set))
                d_set = pd.merge(d_set, unique_rel_df, how='left', on=['utterance'], suffixes=('', '_y'))
                print("d_set after: ", len(d_set))
            if args.train_data_percent < 1:
                # Filter the samples which don't contain the max_num_anchors
                d_set = d_set[d_set['num_anchors'] <= max_anchors]
                if args.target_aug_percentage and (args.train_data_percent==0.1) and (max_anchors==1):
                    unique_rel_df = pd.read_csv("/home/abdelrem/3d_codes/CoT3D_VG/extract_anchors/nr3d_cot_unique_rel_anchor_data.csv")
                    d_set = pd.merge(d_set, unique_rel_df, how='inner', on=['utterance'], suffixes=('', '_y'))
                extend = org_training_len/len(d_set)
                print("-------- org_training_len = ", org_training_len, "   ", len(d_set))
            d_set = d_set.sample(frac=args.train_data_percent*extend)
            print("-------- d_set = ", len(d_set))
            if args.train_data_repeatation > 1:
                d_set_repeated = pd.DataFrame(np.repeat(d_set.values, args.train_data_repeatation, axis=0))
                d_set_repeated.columns = d_set.columns
                d_set = d_set_repeated
        d_set.reset_index(drop=True, inplace=True)

        max_distractors = args.max_distractors if split == 'train' else args.max_test_objects - 1
        ## this is a silly small bug -> not the minus-1.

        # if split == test remove the utterances of unique targets
        if split == 'test':
            def multiple_targets_utterance(x):
                _, _, _, _, distractors_ids = decode_stimulus_string(x.stimulus_id)
                return len(distractors_ids) > 0

            multiple_targets_mask = d_set.apply(multiple_targets_utterance, axis=1)
            d_set = d_set[multiple_targets_mask]
            d_set.reset_index(drop=True, inplace=True)
            print("length of dataset before removing non multiple test utterances {}".format(len(d_set)))
            print("removed {} utterances from the test set that don't have multiple distractors".format(
                np.sum(~multiple_targets_mask)))
            print("length of dataset after removing non multiple test utterances {}".format(len(d_set)))

            assert np.sum(~d_set.apply(multiple_targets_utterance, axis=1)) == 0

        shuffle_source = None
        if shuffle_mode != "none" and split == "train":
            hdf5_file = os.path.join(args.visaug_extracted_obj_path, '%s.hdf5' % shuffle_mode)
            shuffle_source = h5py.File(hdf5_file, 'r')

        dataset = ListeningDataset(references=d_set,
                                   scans=scans,
                                   vocab=vocab,
                                   max_seq_len=args.max_seq_len,
                                   points_per_object=args.points_per_object,
                                   max_distractors=max_distractors,
                                   class_to_idx=class_to_idx,
                                   object_transformation=object_transformation,
                                   visualization=args.mode == 'evaluate',
                                   max_anchors=max_anchors,
                                   anchors_mode=args.anchors,
                                   predict_lang_anchors=args.predict_lang_anchors,
                                   shuffle_objects=shuffle_source,
                                   shuffle_objects_percentage=args.shuffle_objects_percentage,
                                   pc_transforms=pc_transforms,
                                   textaug_paraphrase_percentage=args.textaug_paraphrase_percentage,
                                   target_aug_percentage=args.target_aug_percentage,
                                   is_train=split=='train',
                                   distractor_aux_loss_flag=args.distractor_aux_loss_flag)

        seed = None
        if split == 'test':
            seed = args.random_seed

        data_loaders[split] = dataset_to_dataloader(dataset, split, args.batch_size, n_workers, seed=seed)

    return data_loaders
