import numpy as np
from tqdm import tqdm
import csv
from referit3d.in_out.scannet_scan import ScannetScan, ScannetDataset
import pandas as pd
import string
from extract_objs_from_description import ExtractObjsFromDescription
import xlsxwriter
import random

def read_referring_data_scv(file_path):
    df = pd.read_csv(file_path)
    return df


def save_in_csv(lst, saving_name):
    # Save output in csv:
    keys = lst[0].keys()
    with open(saving_name, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(lst)


if __name__ == '__main__':
    df = read_referring_data_scv(file_path="./data/nr3d.csv")
    scan_ids = df.scan_id
    gt_objs_name_all_scenes = []
    gt_utternaces_all_scenes = []
    pred_objs_name_all_scenes = []
    unique_counter = 0
    num_objs_per_scene = []
    # Create our obj retrieval module:
    obj_extractor = ExtractObjsFromDescription("./data/scannet_instance_class_to_semantic_class.json",
                                               coloring_type="[]")

    for i in tqdm(range(len(scan_ids))):
        scan_id = scan_ids[i]
        if "_00" in scan_id:
            unique_counter = unique_counter + 1
            # Get Ground-Truth anchors and target objects:
            pred_objs_name, objs_start_loc, objs_end_loc, \
                colored_utterance, adapted_utterance = obj_extractor.extract_objs_from_description(
                utterance=df.utterance[i])            
            objects_relations_pred =  obj_extractor.get_objects_relations_pred(df.utterance[i].lower(),pred_objs_name)
            for objects_relation_pred in objects_relations_pred:
                pred_objs_name_all_scenes.append({"org_utterance": df.utterance[i].lower(),
                                                "pred_obj_name_1": objects_relation_pred[0],
                                                "relation"        : objects_relation_pred[1],
                                                "pred_obj_name_2": objects_relation_pred[2]
                                                })
    print("Number of _00 items are: ", unique_counter)

    # Save the predicted objects in CSV file for manual verification:
    save_in_csv(lst=pred_objs_name_all_scenes, saving_name="./data/pred_objs_total_opt1.csv")

    pred_objs_name_all_scenes = random.sample(pred_objs_name_all_scenes, 3000)
    save_in_csv(lst=pred_objs_name_all_scenes, saving_name="./data/pred_objs_3000_opt1.csv")
