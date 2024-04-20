import warnings
import numpy as np
import multiprocessing as mp
from torch.utils.data import DataLoader
import json 
import random
relation_synonyms = {
    "near": ["near", "near to", "close", "closer to", "close to", "besides", "by", "next to", "towards", "along", "alongside", "with"],
    "front": ["opposite", "opposite to", "opposite of", "opposite from", "in front of", "faces", "facing"],
    "far": ["farther", "far from", "farthest from", "farthest", "far", "far away from"],
    "on": ["atop of", "above", "on top", "on top of", "on", "higher", "over", "lying on", "onto"],
    "down": ["below", "down", "beneath", "underneath", "lower", "under", "beaneath"],
    "right": ["right on", "right of", "right", "to the right of", "right most", "on the right side of", "on the right of", "right"],
    "left": ["left on", "left of", "left", "on the left of", "on the left side of", "left most", "to the left of", "left"],
    "back": ["beyond", "back", "behind", "on the back of"],
}
def create_sr3d_classes_2_idx(json_pth):
    with open(json_pth) as json_file:
        data = json.load(json_file)

    instance_labels = set()
    instance_labels.update([k for k in data])

    class_to_idx = {}
    i = 0
    for el in sorted(instance_labels):
        class_to_idx[el] = i
        i += 1

    class_to_idx['pad'] = len(class_to_idx)
    class_to_idx['no_obj'] = len(class_to_idx)

    return class_to_idx

def get_logical_pth_lang(data_dict, key):
    """
    Convert the string into readable list
    """
    for idx, row in enumerate(data_dict[key]):
        clean_objs = []
        if "'" in row:
            objects = row.split(',')
            for obj in objects:
                obj = obj.strip().split("'")[1]
                if '*' in obj:
                    obj = obj[1:]
                clean_objs.append(obj)
        data_dict[key][idx] = clean_objs

    return data_dict


def clean_paraphrased(data_dict):
    """
    Convert the string into readable list
    """
    for idx, row in enumerate(data_dict["paraphrases"]):
        clean_objs = []
        if '"' in row:
            objects = row.split(',')
            for obj in objects:
                obj = obj.strip().split('"')[1]
                clean_objs.append(obj)
        data_dict["paraphrases"][idx] = clean_objs

    return data_dict


def flipcoin(percent=50):
    """
    return Treu or False based on the given percentage.
    """
    return random.randrange(100) < percent

def max_io_workers():
    """ number of available cores -1."""
    n = max(mp.cpu_count() - 1, 1)
    print('Using {} cores for I/O.'.format(n))
    return n


def dataset_to_dataloader(dataset, split, batch_size, n_workers, pin_memory=False, seed=None):
    """
    :param dataset:
    :param split:
    :param batch_size:
    :param n_workers:
    :param pin_memory:
    :param seed:
    :return:
    """
    batch_size_multiplier = 1 if split == 'train' else 2
    b_size = int(batch_size_multiplier * batch_size)

    drop_last = False
    if split == 'train' and len(dataset) % b_size == 1:
        print('dropping last batch during training')
        drop_last = True

    shuffle = split == 'train'

    worker_init_fn = lambda x: np.random.seed(seed)
    if split == 'test':
        if type(seed) is not int:
            warnings.warn('Test split is not seeded in a deterministic manner.')

    data_loader = DataLoader(dataset,
                             batch_size=b_size,
                             num_workers=n_workers,
                             shuffle=shuffle,
                             drop_last=drop_last,
                             pin_memory=pin_memory,
                             worker_init_fn=worker_init_fn)
    return data_loader


def sample_scan_object(object, n_points):
    sample = object.sample(n_samples=n_points)
    return np.concatenate([sample['xyz'], sample['color']], axis=1)


def pad_samples(samples, max_context_size, padding_value=1):
    n_pad = max_context_size - len(samples)

    if n_pad > 0:
        shape = (max_context_size, samples.shape[1], samples.shape[2])
        temp = np.ones(shape, dtype=samples.dtype) * padding_value
        temp[:samples.shape[0], :samples.shape[1]] = samples
        samples = temp

    return samples


def check_segmented_object_order(scans):
    """ check all scan objects have the three_d_objects sorted by id
    :param scans: (dict)
    """
    for scan_id, scan in scans.items():
        idx = scan.three_d_objects[0].object_id
        for o in scan.three_d_objects:
            if not (o.object_id == idx):
                print('Check failed for {}'.format(scan_id))
                return False
            idx += 1
    return True


def objects_bboxes(context):
    b_boxes = []
    for o in context:
        bbox = o.get_bbox(axis_aligned=True)

        # Get the centre
        cx, cy, cz = bbox.cx, bbox.cy, bbox.cz

        # Get the scale
        lx, ly, lz = bbox.lx, bbox.ly, bbox.lz

        b_boxes.append([cx, cy, cz, lx, ly, lz])

    return np.array(b_boxes).reshape((len(context), 6))


def instance_labels_of_context(context, max_context_size, label_to_idx=None, add_padding=True):
    """
    :param context: a list of the objects
    :return:
    """
    instance_labels = [i.instance_label for i in context]

    if add_padding:
        n_pad = max_context_size - len(context)
        instance_labels.extend(['pad'] * n_pad)

    if label_to_idx is not None:
        instance_labels = np.array([label_to_idx[x] for x in instance_labels])

    return instance_labels


def mean_rgb_unit_norm_transform(segmented_objects, mean_rgb, unit_norm, epsilon_dist=10e-6, inplace=True):
    """
    :param segmented_objects: K x n_points x 6, K point-clouds with color.
    :param mean_rgb:
    :param unit_norm:
    :param epsilon_dist: if max-dist is less than this, we apply not scaling in unit-sphere.
    :param inplace: it False, the transformation is applied in a copy of the segmented_objects.
    :return:
    """
    if not inplace:
        segmented_objects = segmented_objects.copy()

    # adjust rgb
    segmented_objects[:, :, 3:6] -= np.expand_dims(mean_rgb, 0)

    # center xyz
    if unit_norm:
        xyz = segmented_objects[:, :, :3]
        mean_center = xyz.mean(axis=1)
        xyz -= np.expand_dims(mean_center, 1)
        max_dist = np.max(np.sqrt(np.sum(xyz ** 2, axis=-1)), -1)
        max_dist[max_dist < epsilon_dist] = 1  # take care of tiny point-clouds, i.e., padding
        xyz /= np.expand_dims(np.expand_dims(max_dist, -1), -1)
        segmented_objects[:, :, :3] = xyz
        mean_center = np.concatenate((mean_center, np.expand_dims(max_dist, 1)),axis=1)

    return segmented_objects, mean_center
