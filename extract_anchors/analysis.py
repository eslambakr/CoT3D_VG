import numpy as np
from tqdm import tqdm
import csv
from referit3d.in_out.scannet_scan import ScannetScan, ScannetDataset
import pandas as pd
import string
from extract_objs_from_description import ExtractObjsFromDescription
import random
import json
import multiprocessing as mp
import ast


def read_referring_data_scv(file_path):
    df = pd.read_csv(file_path)
    return df


def read_cls_json(file_name):
    f = open(file_name)
    ins_sem_cls_map = json.load(f)
    return ins_sem_cls_map
        

def scannet_loader(scan_id):
    """Helper function to load the scans in memory.
    :param scan_id:
    :return: the loaded scan.
    """
    global scannet

    # print("scan_id = ", scan_id)
    scan_i = ScannetScan(scan_id=scan_id, scannet_dataset=scannet, apply_global_alignment=False, load_dense=load_dense)
    if load_dense:
        scan_i.load_point_clouds_of_all_objects_dense()
    else:
        scan_i.load_point_clouds_of_all_objects()

    return scan_i


if __name__ == '__main__':
    dataset = "ScanRefer"  # three options are supported [Nr3D, Sr3D, ScanRefer]
    if dataset == "Nr3D":
        df = read_referring_data_scv(file_path="nr3D_with_TRUE_GT.csv")
        scan_ids = df.scan_id
    elif dataset == "Sr3D":
        df = read_referring_data_scv(file_path="/home/abdelrem/3d_codes/scannet_dataset/scannet/sr3d.csv")
        scan_ids = df.scan_id
    elif dataset == "ScanRefer":
        df = read_referring_data_scv(file_path="/home/abdelrem/3d_codes/CoT3D_VG/refering_codes/ScanRefer-master/data/merged_train_scanrefer_cot.csv")
        scan_ids = df.scene_id
    # Configurations:
    # ---------------
    load_dense = False
    scannet_dataset_path = "/home/abdelrem/3d_codes/scannet_dataset/"
    # scannet_dataset_path = "/media/eslam/0d208863-5cdb-4a43-9794-3ca8726831b3/3D_visual_grounding/dataset/"

    # Read the scan related information
    top_scan_dir = scannet_dataset_path + "scannet/scans"
    idx_to_semantic_class_file = '../automatic_loc_module/referit3d/data/mappings/scannet_idx_to_semantic_class.json'
    instance_class_to_semantic_class_file = '../automatic_loc_module/referit3d/data/mappings/scannet_instance_class_to_semantic_class.json'
    axis_alignment_info_file = '../automatic_loc_module/referit3d/data/scannet/scans_axis_alignment_matrices.json'
    scannet = ScannetDataset(top_scan_dir,
                             idx_to_semantic_class_file,
                             instance_class_to_semantic_class_file,
                             axis_alignment_info_file)
    # Loop on the whole scenes and load them once:
    all_scenes_paths = list(np.unique(np.array(scan_ids)))
    all_scenes_paths = list(np.unique(np.array(all_scenes_paths)))
    scenes_dict = {}
    all_scan_ids = all_scenes_paths
    n_items = len(all_scan_ids)
    n_processes = min(mp.cpu_count(), n_items)
    pool = mp.Pool(n_processes)
    chunks = int(n_items / n_processes)

    print("Start Loading the Scans in the memory ..............")
    for i, data in enumerate(pool.imap(scannet_loader, all_scan_ids, chunksize=chunks)):
        scenes_dict[all_scan_ids[i]] = data
    pool.close()
    pool.join()
    print("Finish Loading the Scans in the memory ..............")
    
    # Get percentage of unique targets in Nr3D:
    unique_tgt_count = 0
    for i in tqdm(range(len(df))):  # Loop on the scenes
        if dataset == "Nr3D" or dataset == "Sr3D":
            target_name = df.instance_type[i]
            scan_id = df.scan_id[i]
        elif dataset == "ScanRefer":
            target_name = df.object_name[i]
            scan_id = df.scene_id[i]
        
        tgt_count_per_sample = 0
        for obj_3d in scenes_dict[scan_id].three_d_objects:
            if (target_name == obj_3d.instance_label):
                tgt_count_per_sample +=1
        if tgt_count_per_sample <= 1:
            unique_tgt_count +=1
    print("Percentage of Unique targets: ", 100*(unique_tgt_count/len(df)))

    # Get percentage of unique anchors in Nr3D:
    unique_anchors_count, total_num_anchors = 0, 0
    for i in tqdm(range(len(df))):  # Loop on the scenes
        if dataset == "Nr3D":
            anchors_names = ast.literal_eval(df.true_gt_anchor_names[i])[:-1]  # to exclude the tgt
            scan_id = df.scan_id[i]
        elif dataset == "Sr3D":
            anchors_names = ast.literal_eval(df.anchors_types[i])
            scan_id = df.scan_id[i]
        elif dataset == "ScanRefer":
            anchors_names = ast.literal_eval(df.path[i])[:-1]  # to exclude the tgt
            scan_id = df.scene_id[i]

        for j in range(len(anchors_names)):  # Loop on the anchors in one sample
            total_num_anchors += 1
            anchor_name = anchors_names[j]
            anchor_count_per_sample = 0
            for obj_3d in scenes_dict[scan_id].three_d_objects:
                if (anchor_name == obj_3d.instance_label):
                    anchor_count_per_sample +=1
            if anchor_count_per_sample <= 1:
                unique_anchors_count +=1
    print("Percentage of Unique anchors: ", 100*(unique_anchors_count/total_num_anchors))
    # -------------------------------------------------------------------------------------------------
    #                                          Extract anchors' names
    # -------------------------------------------------------------------------------------------------