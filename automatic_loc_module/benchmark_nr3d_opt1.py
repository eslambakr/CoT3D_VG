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
                
            objects_relations_1 = obj_extractor.get_objects_relations_pred(utterance=df.utterance[i])
            sub_phrases, sub_phrases_start_obj_loc, sub_phrases_end_obj_loc, objs_name = obj_extractor.get_phrases_between_2_objs(ip_sentence=df.utterance[i]\
                , objs_name=pred_objs_name)
            
            conflict_flag , pred_relationship_word_per_phrase = obj_extractor.get_relationship_between_2_objs(sub_phrases)
            
            total_relations = set()
            res_relations = []
            for _, object_relation in enumerate(objects_relations_1):
                total_relations.add((object_relation[0], object_relation[2]))
                res_relations.append(object_relation)
            
            for j, sub_phrase in enumerate(sub_phrases):
                if((objs_name[sub_phrases_start_obj_loc[j]], objs_name[sub_phrases_end_obj_loc[j]]) in total_relations or pred_relationship_word_per_phrase[j] == None):
                    continue 
                res_relations.append([objs_name[sub_phrases_start_obj_loc[j]], pred_relationship_word_per_phrase[j], objs_name[sub_phrases_end_obj_loc[j]]])
            # import pdb; pdb.set_trace()
            for objects_relation_pred in res_relations:
                pred_objs_name_all_scenes.append({"org_utterance": df.utterance[i].lower(),
                                                "object": objects_relation_pred[0],
                                                "relation"        : objects_relation_pred[1],
                                                "subject": objects_relation_pred[2],
                                                "id": i
                                                })
                
    print("Number of _00 items are: ", unique_counter)

    # Save the predicted objects in CSV file for manual verification:
    save_in_csv(lst=pred_objs_name_all_scenes, saving_name="./data/pred_objs_total_opt1.csv")

    pred_objs_name_all_scenes = random.sample(pred_objs_name_all_scenes, 3000)
    save_in_csv(lst=pred_objs_name_all_scenes, saving_name="./data/pred_objs_3000_opt1.csv")
