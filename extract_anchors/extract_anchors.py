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
from extraction_utils import *


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
        # print("XXX for now will return the first occurrence")
        return indices[0]  # for now will return the first occurrence


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
    df = read_referring_data_scv(file_path="nr3d_cot_ref_paraphrases.csv")
    scan_ids = df.scan_id
    # Configurations:
    # ---------------
    load_dense = False
    scannet_dataset_path = "/home/ahmems0a/repos/CoT3D_VG/automatic_loc_module/"
    # scannet_dataset_path = "/media/eslam/0d208863-5cdb-4a43-9794-3ca8726831b3/3D_visual_grounding/dataset/"

    # Read the scan related information
    top_scan_dir = scannet_dataset_path + "scannet/scans"
    idx_to_semantic_class_file = './referit3d/data/mappings/scannet_idx_to_semantic_class.json'
    instance_class_to_semantic_class_file = './referit3d/data/mappings/scannet_instance_class_to_semantic_class.json'
    axis_alignment_info_file = './referit3d/data/scannet/scans_axis_alignment_matrices.json'
    scannet = ScannetDataset(top_scan_dir,
                             idx_to_semantic_class_file,
                             instance_class_to_semantic_class_file,
                             axis_alignment_info_file)
    # Loop on the whole scenes and load them once:
    all_scenes_paths = glob.glob(top_scan_dir+"/*")
    all_scenes_paths = list(np.unique(np.array(scan_ids)))
    scenes_dict = {}
    all_scan_ids = all_scenes_paths
    n_items = len(all_scan_ids)
    n_processes = min(mp.cpu_count(), n_items)
    pool = mp.Pool(n_processes)
    chunks = int(n_items / n_processes)

    for i, data in enumerate(pool.imap(scannet_loader, all_scan_ids, chunksize=chunks)):
        scenes_dict[all_scan_ids[i]] = data
    pool.close()
    pool.join()

    gt_objs_name_all_scenes = []
    gt_utternaces_all_scenes = []
    pred_objs_name_all_scenes = []
    counter = 0
    unique_anchor_counter = 0
    empty_relation_counter = 0
    empty_anchor_counter = 0
    no_box_counter = 0
    target_mismatch_target = 0
    correct_counter = 0
    target_counter = 0
    unique_anchor_counter2 = 0
    empty_relation_counter2 = 0
    empty_anchor_counter2 = 0
    hard_counter = 0
    empty_anchor_flag = False
    empty_relation_flag = False
    unique_anchor_flag = False

    no_anchor_counter_per_scene = 0
    all_scene_anchors = {}
    # Create our obj retrieval module:
    # obj_extractor = ExtractObjsFromDescription("./data/scannet_instance_class_to_semantic_class.json")
    relations_df = pd.read_csv("relations_parsed.csv")
    relations_df['used_obj'] = [False] * len(relations_df)
    relations_df['used_sub'] = [False] * len(relations_df)


    with open('mapped_relation.json') as f:
        mapped_relation = json.load(f)
    relation_fn_dict = {
        'near': nearest_3dobject,
        'far': farthest_3dobject,
        'right': right_3dobject,
        'left': left_3dobject,
        'on': on_3dobject,
        'down': under_3dobject,
        'front': front_3dobject,
        'back': back_3dobject
    }
    reverse_relation_fn_dict = {
        'near': nearest_3dobject,
        'far': farthest_3dobject,
        'right': left_3dobject,
        'left': right_3dobject,
        'on': under_3dobject,
        'down': on_3dobject,
        'front': back_3dobject,
        'back': front_3dobject
    }
    # df['used'] = [False] * len(df)
    for i in tqdm(range(len(scan_ids))):  # Loop on the scenes
    # i = 50
        scan_id = scan_ids[i]
        if "_00" in scan_id or True:
            empty_anchor_flag = False
            empty_relation_flag = False
            unique_anchor_flag = False
            # Run our obj retrieval module:
            pred_objs_name = extract_objs_from_description(df.utterance[i], df.instance_type[i], df.path[i])
            # if len(pred_objs_name[-1]) >0 and  df.instance_type[i] != pred_objs_name[-1]:
            #     target_mismatch_target += 1
            # print(pred_objs_name)
            subtract = False
            trgt_idx = 9999
            # possible_anchors_dict = {}  # clear the dictionary for each scene
            all_scene_anchors[i] = [-1] * (len(pred_objs_name))
            # print(len(all_scene_anchors[i]))
            # if len(pred_objs_name) == 1 and pred_objs_name[0] == df.instance_type[i]:
            #     # all_scene_anchors[i].append(-1)
            #     no_anchor_counter_per_scene += 1

            pred_objs_name_all_scenes.append(pred_objs_name)
            # Extract easy objs; prior knowledge (the target and the unique objects):
            pred_anchor = [None] * len(pred_objs_name)
            possible_anchors_dict = {}  # clear the dictionary for each scene
            for pred_obj_idx, pred_obj_name in enumerate(pred_objs_name):  # Loop on the predicted objects in the utterance
                counter += 1
                if '*' == pred_obj_name[0]:
                    no_box_counter += 1
                    continue
                # Get all the possible objects that exist in the scene and match the predicted obj class from the text:
                if not (pred_obj_name in possible_anchors_dict.keys()):
                    possible_anchors_dict[pred_obj_name] = []  # initialize the list once
                    for obj_3d in scenes_dict[scan_id].three_d_objects:
                        # print(obj_3d.instance_label, obj_3d.object_id)
                        if (pred_obj_name == obj_3d.instance_label): # or (pred_obj_name in obj_3d.instance_label.split()) or (obj_3d.instance_label in pred_obj_name.split()):
                            possible_anchors_dict[pred_obj_name].append(obj_3d)
                # print("*"*10)
                # Exclude the target:
                if pred_obj_name == df.instance_type[i]: # or pred_obj_name in df.instance_type[i] or df.instance_type[i] in pred_obj_name:
                    # Detect the target location from the predicted objects from the utterance:
                    obj_name = df.instance_type[i]
                    if pred_obj_name != df.instance_type[i]:
                        obj_name = pred_obj_name
                    target_idx = extract_target_loc_from_pred_objs_from_description(pred_objs_list=pred_objs_name,
                                                                                    target_class=obj_name)
                    # print("target found at", target_idx)
                    if pred_obj_idx == target_idx:  # make sure it is the target not text-distractor
                        for anchor_id, anchor in enumerate(possible_anchors_dict[pred_obj_name]):
                            if anchor.object_id == df.target_id[i]:
                                target_counter += 1
                                subtract = True
                                trgt_idx = target_idx
                                target_anchor = possible_anchors_dict[pred_obj_name][anchor_id]
                                pred_anchor[pred_obj_idx] = target_anchor
                                del possible_anchors_dict[pred_obj_name][anchor_id]
                                print("target",pred_obj_name,len(all_scene_anchors[i]), pred_obj_idx)
                                del all_scene_anchors[i][pred_obj_idx]
                                break
                        continue
                # import pdb; pdb.set_trace()
                # print(pred_obj_name, len(possible_anchors_dict[pred_obj_name]))
                if len(possible_anchors_dict[pred_obj_name]) == 0:
                    print("XXX  Error the obj not found",pred_obj_name, "!!!")
                    empty_anchor_counter += 1
                    empty_anchor_flag = True
                    continue
                elif len(possible_anchors_dict[pred_obj_name]) == 1:  # Unique object
                    # import pdb; pdb.set_trace()
                    pred_anchor[pred_obj_idx] = possible_anchors_dict[pred_obj_name][0]
                    anch_id = pred_obj_idx
                    if subtract and pred_obj_idx > trgt_idx:
                        anch_id = max(pred_obj_idx -1, 0)
                    print("found unique", pred_obj_name)
                    all_scene_anchors[i][anch_id] = possible_anchors_dict[pred_obj_name][0].object_id
                    unique_anchor_flag = True
                    unique_anchor_counter += 1

            # Assign the hard objs (Several objects) using the geometry info:
            remaining_indices = [c for c, x in enumerate(pred_anchor) if x is None]  # find indices of hard objs
            objs_center = [None] * len(pred_objs_name)
            # Loop on the remaining objects:
            for idx in remaining_indices:
                hard_counter += 1
                pred_obj_name = pred_objs_name[idx]
                if pred_obj_name[0] == '*':
                    continue

                # check unique obj may be after the target removal the obj become unique.
                if len(possible_anchors_dict[pred_obj_name]) <= 1:
                    print("in the second unique case")
                    continue
            
                # 1- Get the center of each object:
                # 1.1-unassigned objs of same class center
                unassigned_anchors_center = []
                for anchor in possible_anchors_dict[pred_obj_name]:  # unassigned objs
                    obj_pc = scenes_dict[scan_id].pc[anchor.points]
                    unassigned_anchors_center.append(np.mean(obj_pc, axis=0))
                    # w, l, h = get3d_box_from_pcs(obj_pc)
                # 1.2-assigned objs center
                known_indices = [c for c, x in enumerate(pred_anchor) if x is not None]  # find indices of hard objs
                known_centers = []
                for known_idx in known_indices:
                    obj_pc = scenes_dict[scan_id].pc[pred_anchor[known_idx].points]
                    known_centers.append((get3d_box_center_from_pcs(obj_pc), pred_anchor[known_idx].instance_label))
                # import pdb; pdb.set_trace()
                # 2- Get the relationship between objects:
                # get relationships containing the current obj:
                relations = get_relationship_conatining_obj(i, relations_df, pred_obj_name)
                if len(relations) == 0:
                    print("Error: no relations found for", pred_obj_name)
                    # all_scene_anchors[i].append(-1)
                    empty_relation_counter += 1
                    empty_relation_flag = True
                    rand_idx = np.random.randint(0,len(possible_anchors_dict[pred_obj_name]))
                    pred_anchor[idx] = possible_anchors_dict[pred_obj_name][rand_idx]
                    # del unassigned_anchors_center[rand_idx]
                    anch_id = idx
                    if subtract and pred_obj_idx > trgt_idx:
                        anch_id = max(idx-1, 0)
                    all_scene_anchors[i][anch_id] = pred_anchor[idx].object_id
                deleted_objs = 0
                found = False
                # loop over the relations and get which one of the relations has a known object in the second value of the tuple
                for relation_tuple in relations:
                    if found:
                        break
                    relation, obj2, flag = relation_tuple
                    
                    # Get the closest relation mapping from mapped_relation using get_closest_relation_mapping
                    closest_relation = get_closest_relation_mapping(relation, mapped_relation.keys())
                    obj2_centers = [c[0] for c in known_centers if c[1] == obj2]
                    # print("HERE", closest_relation, len(obj2_centers))
                    if closest_relation == '' or len(obj2_centers) == 0:
                        print("Error: relation failed", relation, len(obj2_centers))
                        # all_scene_anchors[i].append(-1)
                        empty_relation_counter += 1
                        rand_idx = np.random.randint(0,len(possible_anchors_dict[pred_obj_name]))
                        pred_anchor[idx] = possible_anchors_dict[pred_obj_name][rand_idx]
                        # del unassigned_anchors_center[rand_idx]
                        anch_id = idx
                        if subtract and pred_obj_idx > trgt_idx:
                            anch_id = max(idx-1, 0)
                        all_scene_anchors[i][anch_id] = pred_anchor[idx].object_id

                        empty_relation_flag = True
                        continue
                    
                    relation_mapped = mapped_relation[closest_relation] 
                    for known_center in obj2_centers:
                        if len(unassigned_anchors_center) == 0:
                            break
                        if flag:
                            approx_anchor = relation_fn_dict[relation_mapped](known_center, unassigned_anchors_center)
                        else:
                            approx_anchor = reverse_relation_fn_dict[relation_mapped](known_center, unassigned_anchors_center)
                        # get the index of the chosen center with regards to unassigned_anchors_center:
                        if approx_anchor is not None:
                            chosen_center_idx = np.argmin(np.linalg.norm(np.array(unassigned_anchors_center) - np.array(approx_anchor), axis=1))
                            del unassigned_anchors_center[chosen_center_idx]
                            chosen_center_idx = chosen_center_idx + deleted_objs
                            deleted_objs += 1
                            found = True
                            pred_anchor[idx] = possible_anchors_dict[pred_obj_name][chosen_center_idx]
                            anch_id = idx
                            if subtract and pred_obj_idx > trgt_idx:
                                anch_id = max(idx-1, 0)
                            all_scene_anchors[i][anch_id] = possible_anchors_dict[pred_obj_name][chosen_center_idx].object_id
                            correct_counter += 1
                            break
                        else:
                            if relation_mapped != 'far':
                                approx_anchor = relation_fn_dict['near'](known_center, unassigned_anchors_center)
                                # print(unassigned_anchors_center)
                                chosen_center_idx = np.argmin(np.linalg.norm(np.array(unassigned_anchors_center) - np.array(approx_anchor), axis=1))
                                del unassigned_anchors_center[chosen_center_idx]
                                found = True
                                chosen_center_idx = chosen_center_idx + deleted_objs
                                deleted_objs += 1
                                pred_anchor[idx] = possible_anchors_dict[pred_obj_name][chosen_center_idx]
                                anch_id = idx
                                if subtract and pred_obj_idx > trgt_idx:
                                    anch_id = max(idx-1, 0)
                                all_scene_anchors[i][anch_id] = possible_anchors_dict[pred_obj_name][chosen_center_idx].object_id
                                correct_counter += 1
                                break
                            else:
                                empty_relation_counter += 1
                                empty_relation_flag = True
                    
                                
            for h, anchor in enumerate(pred_anchor):
                if pred_anchor is None:
                    all_scene_anchors[i].insert(h, -1)
            # break
            if unique_anchor_flag:
                unique_anchor_counter2 += 1 
            if empty_anchor_flag:
                empty_anchor_counter2 += 1
            if empty_relation_flag:
                empty_relation_counter2 += 1           

            #break
    
    with open('all_scene_anchors_normal_objs.json', 'w') as fp:
        json.dump(all_scene_anchors, fp)
    print("The total num of objects is:", counter)
    print("The unique ratio is:", (unique_anchor_counter/counter)*100, "%")
    print("The no box ratio is:", (no_box_counter/counter)*100, "%")
    print("The empty ratio is:", (empty_anchor_counter/counter)*100, "%")
    print("The empty relation ratio is:", (empty_relation_counter/counter)*100, "%")
    print("The target ratio is:", (target_counter/counter)*100, "%")
    print("The unique+tgt ratio is:", ((unique_anchor_counter+target_counter)/counter)*100, "%")
    print("The correct ratio is:", (correct_counter/counter)*100, "%")


    print("*"*10)
    print("The target mismatch counter is:", target_mismatch_target)
    print("The no anchors  ratio is:", (no_anchor_counter_per_scene/len(df))*100, "%")
    print("The empty relation per scene ratio is:", (empty_relation_counter2/len(df))*100, "%")
    print("The empty per scene  ratio is:", (empty_anchor_counter2/len(df))*100, "%")
    print("The unique per scene ratio is::", (unique_anchor_counter2/len(df))*100, "%")

