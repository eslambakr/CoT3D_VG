from tqdm import tqdm
from referit3d.in_out.scannet_scan import ScannetScan, ScannetDataset
import pandas as pd
import string
from extract_objs_from_description import ExtractObjsFromDescription
import glob
import numpy as np
from benchmark_auto_obj_extraction_module_sr3d import read_referring_data_scv
import multiprocessing as mp
import math
import json


def get3d_box_from_pcs(pc):
    """
    Given point-clouds that represent object or scene return the 3D dimension of the 3D box that contains the PCs.
    """
    w = pc[:, 0].max() - pc[:, 0].min()
    l = pc[:, 1].max() - pc[:, 1].min()
    h = pc[:, 2].max() - pc[:, 2].min()
    return w, l, h


def get3d_box_center_from_pcs(pc):
    """
    Given point-clouds that represent object or scene return the 3D center of the 3D box that contains the PCs.
    """
    w, l, h = get3d_box_from_pcs(pc)
    return np.array([pc[:, 0].max() - w / 2, pc[:, 1].max() - l / 2, pc[:, 2].max() - h / 2])


def extract_target_loc_from_pred_objs_from_description(pred_objs_list, target_class):
    indices = [c for c, x in enumerate(pred_objs_list) if x == target_class]  # find indices of the target class
    if len(indices) == 1:
        return indices[0]
    else:  # multiple targets have been found.
        # TODO: Eslam: think about a way to find which one is the target.
        print("XXX for now will return the first occurrence")
        return indices[0]  # for now will return the first occurrence


def scannet_loader(scan_id):
    """Helper function to load the scans in memory.
    :param scan_id:
    :return: the loaded scan.
    """
    global scannet

    print("scan_id = ", scan_id)
    scan_i = ScannetScan(scan_id=scan_id, scannet_dataset=scannet, apply_global_alignment=False, load_dense=load_dense)
    if load_dense:
        scan_i.load_point_clouds_of_all_objects_dense()
    else:
        scan_i.load_point_clouds_of_all_objects()

    return scan_i


if __name__ == '__main__':
    df = read_referring_data_scv(file_path="./data/sr3d.csv")
    scan_ids = df.scan_id
    # Configurations:
    # ---------------
    load_dense = False
    # scannet_dataset_path = "/home/eslam/scannet_dataset/"
    # # scannet_dataset_path = "/media/eslam/0d208863-5cdb-4a43-9794-3ca8726831b3/3D_visual_grounding/dataset/"

    # # Read the scan related information
    # top_scan_dir = scannet_dataset_path + "scannet/scans"
    # idx_to_semantic_class_file = './referit3d/data/mappings/scannet_idx_to_semantic_class.json'
    # instance_class_to_semantic_class_file = './referit3d/data/mappings/scannet_instance_class_to_semantic_class.json'
    # axis_alignment_info_file = './referit3d/data/scannet/scans_axis_alignment_matrices.json'
    # scannet = ScannetDataset(top_scan_dir,
    #                          idx_to_semantic_class_file,
    #                          instance_class_to_semantic_class_file,
    #                          axis_alignment_info_file)
    # # Loop on the whole scenes and load them once:
    # all_scenes_paths = glob.glob(top_scan_dir+"/*")
    # all_scenes_paths = list(np.unique(np.array(scan_ids)))
    # scenes_dict = {}
    # all_scan_ids = all_scenes_paths
    # n_items = len(all_scan_ids)
    # n_processes = min(mp.cpu_count(), n_items)
    # pool = mp.Pool(n_processes)
    # chunks = int(n_items / n_processes)

    # for i, data in enumerate(pool.imap(scannet_loader, all_scan_ids, chunksize=chunks)):
    #     scenes_dict[all_scan_ids[i]] = data
    # pool.close()
    # pool.join()

    # scences_dict = {}
    # for i in tqdm(range(len(all_scan_ids))):  # Loop on the scenes
    #     scan_id = all_scan_ids[i]
    #     dummy_scene = scenes_dict[scan_id]
    #     obj_list = []
    #     for obj in dummy_scene.three_d_objects:
    #         obj_points = dummy_scene.pc[obj.points]
    #         min_z = obj_points[:, 2].min()
    #         max_z = obj_points[:, 2].max()
    #         min_x = obj_points[:, 0].min()
    #         p1 = [min_x, obj_points[:, 1][list(obj_points[:, 0]).index(min_x)], min_z]
    #         p2 = [min_x, obj_points[:, 1][list(obj_points[:, 0]).index(min_x)], max_z]
    #         max_x = obj_points[:, 0].max()
    #         p3 = [max_x, obj_points[:, 1][list(obj_points[:, 0]).index(max_x)], min_z]
    #         p4 = [max_x, obj_points[:, 1][list(obj_points[:, 0]).index(max_x)], max_z]
    #         max_y = obj_points[:, 1].max()
    #         p5 = [obj_points[:, 0][list(obj_points[:, 1]).index(max_y)], max_y, min_z]
    #         p6 = [obj_points[:, 0][list(obj_points[:, 1]).index(max_y)], max_y, max_z]
    #         min_y = obj_points[:, 1].min()
    #         p7 = [obj_points[:, 0][list(obj_points[:, 1]).index(min_y)], min_y, min_z]
    #         p8 = [obj_points[:, 0][list(obj_points[:, 1]).index(min_y)], min_y, max_z]
    #         conrners = [p1, p2, p3, p4, p5, p6, p7, p8]
    #         obj_list.append({
    #             "scan_id": all_scan_ids[i], "object_id": str(obj.object_id).zfill(3),
    #             "object_unique_id": all_scan_ids[i].split("scene")[-1].split("_")[0] + str(obj.object_id).zfill(3),
    #             "instance_type": obj.instance_label,
    #             "x1": str(p1[0]), "y1": str(p1[1]), "z1": str(p1[2]), "x2": str(p2[0]), "y2": str(p2[1]), "z2": str(p2[2]),
    #             "x3": str(p3[0]), "y3": str(p3[1]), "z3": str(p3[2]), "x4": str(p4[0]), "y4": str(p4[1]), "z4": str(p4[2]),
    #             "x5": str(p5[0]), "y5": str(p5[1]), "z5": str(p5[2]), "x6": str(p6[0]), "y6": str(p6[1]), "z6": str(p6[2]),
    #             "x7": str(p7[0]), "y7": str(p7[1]), "z7": str(p7[2]), "x8": str(p8[0]), "y8": str(p8[1]), "z8": str(p8[2])
    #         })

    #         # https://stackoverflow.com/questions/61888228/convert-3d-box-vertices-to-center-dimensions-and-rotation
    #         angle = math.atan2(p3[1] - p1[1], p3[0] - p1[0])
    #         cx = (p1[0] + p4[0]) / 2
    #         cy = (p1[1] + p4[1]) / 2
    #         cz = (p1[2] + p4[2]) / 2
    #         lx, ly, lz = p4[0] - p1[0], p4[1] - p1[1], p4[2] - p1[2]
    #         box = [cx, cy, cz, lx, ly, lz, angle]
    #     scences_dict[all_scan_ids[i]] = obj_list

    # with open("3d_objs_vertices.json", "w") as write_file:
    #     json.dump(scences_dict, write_file)

    # sdafsdf = asfsdf
    gt_objs_name_all_scenes = []
    gt_utternaces_all_scenes = []
    pred_objs_name_all_scenes = []
    counter = 0
    unique_anchor_counter = 0
    target_counter = 0
    hard_counter = 0
    # Create our obj retrieval module:
    obj_extractor = ExtractObjsFromDescription("./data/scannet_instance_class_to_semantic_class.json")

    for i in tqdm(range(len(scan_ids))):  # Loop on the scenes
        scan_id = scan_ids[i]
        if "_00" in scan_id or True:
            # Run our obj retrieval module:
            pred_objs_name = obj_extractor.extract_objs_from_description(utterance=df.utterance[i])
            pred_objs_name_all_scenes.append(pred_objs_name)

            # Extract easy objs; prior knowledge (the target and the unique objects):
            pred_anchor = [None] * len(pred_objs_name)
            possible_anchors_dict = {}  # clear the dictionary for each scene
            for pred_obj_idx, pred_obj_name in enumerate(pred_objs_name):  # Loop on the predicted objects in the utterance
                counter += 1
                # Get all the possible objects that exist in the scene and match the predicted obj class from the text:
                if not (pred_obj_name in possible_anchors_dict.keys()):
                    possible_anchors_dict[pred_obj_name] = []  # initialize the list once
                for obj_3d in scenes_dict[scan_id].three_d_objects:
                    if pred_obj_name == obj_3d.instance_label:
                        possible_anchors_dict[pred_obj_name].append(obj_3d)

                # Exclude the target:
                if pred_obj_name == df.instance_type[i]:
                    # Detect the target location from the predicted objects from the utterance:
                    target_idx = extract_target_loc_from_pred_objs_from_description(pred_objs_list=pred_objs_name,
                                                                                    target_class=df.instance_type[i])
                    if pred_obj_idx == target_idx:  # make sure it is the target not text-distractor
                        for anchor_id, anchor in enumerate(possible_anchors_dict[pred_obj_name]):
                            if anchor.object_id == df.target_id[i]:
                                target_counter += 1
                                target_anchor = possible_anchors_dict[pred_obj_name][anchor_id]
                                pred_anchor[pred_obj_idx] = target_anchor
                                del possible_anchors_dict[pred_obj_name][anchor_id]
                                break
                        continue

                if len(possible_anchors_dict[pred_obj_name]) == 0:
                    print("XXX  Error the obj not found!!!")
                elif len(possible_anchors_dict[pred_obj_name]) == 1:  # Unique object
                    pred_anchor[pred_obj_idx] = possible_anchors_dict[pred_obj_name][0]
                    unique_anchor_counter += 1

            """
            # Assign the hard objs (Several objects) using the geometry info:
            remaining_indices = [c for c, x in enumerate(pred_anchor) if x is None]  # find indices of hard objs
            objs_center = [None] * len(pred_objs_name)
            # Get relationships words:
            sub_phrases, sub_phrases_start_obj_loc, sub_phrases_end_obj_loc = obj_extractor.get_phrases_between_2_objs(
                ip_sentence=df.utterance[i], objs_name=pred_objs_name)
            pred_relationship_word_per_phrase = obj_extractor.get_relationship_between_2_objs(sub_phrases)
            # Loop on the remaining objects:
            for idx in remaining_indices:
                hard_counter += 1
                pred_obj_name = pred_objs_name[idx]
                # check unique obj may be after the target removal the obj become unique.
                if len(possible_anchors_dict[pred_obj_name]) == 1:  # Unique object
                    pred_anchor[idx] = possible_anchors_dict[pred_obj_name][0]
                else:  # Several objs
                    # 1- Get the center of each object:
                    # 1.1-unassigned objs of same class center
                    unassigned_anchors_center = []
                    for anchor in possible_anchors_dict[pred_obj_name]:  # unassigned objs
                        obj_pc = scenes_dict[scan_id].pc[anchor.points]
                        unassigned_anchors_center.append(get3d_box_center_from_pcs(obj_pc))
                        w, l, h = get3d_box_from_pcs(obj_pc)
                    # 1.2-assigned objs center
                    known_indices = [c for c, x in enumerate(pred_anchor) if x is not None]  # find indices of hard objs
                    for known_idx in known_indices:
                        obj_pc = scenes_dict[scan_id].pc[pred_anchor[known_idx].points]
                        objs_center[known_idx] = get3d_box_center_from_pcs(obj_pc)

                    # 2- Get the relationship between objects:


                    # 3- Predict the anchor based on the relationship:
            """
            """
            df.instance_type[i]
            df.reference_type[i]
            df.target_id[i]
            df.utterance[i]
            """

        #break
    print("The unique ratio is:", (unique_anchor_counter/counter)*100, "%")
    print("The unique+tgt ratio is:", ((unique_anchor_counter+target_counter)/counter)*100, "%")
