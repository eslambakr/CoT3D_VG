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

            # Run our obj retrieval module:
            pred_objs_name, objs_start_loc, objs_end_loc, \
                colored_utterance, adapted_utterance = obj_extractor.extract_objs_from_description(
                utterance=df.utterance[i])
            num_objs_per_scene.append(len(pred_objs_name))

            #Relationship between objects
            sub_phrases, sub_phrases_start_obj_loc, sub_phrases_end_obj_loc, objs_name = obj_extractor.get_phrases_between_2_objs(\
                ip_sentence=df.utterance[i].lower(), objs_name=pred_objs_name)

            
            
            #Extract the relationship words in sentences
            pred_relationship_word_per_phrase = obj_extractor.get_relationship_between_2_objs(sub_phrases)
            
            start_obj_name = []
            end_obj_name   = []
            for i, sub_phrase in enumerate(sub_phrases):
                # print(i, "--> phrase:", sub_phrase, ", start_obj:", objs_name[sub_phrases_start_obj_loc[i]],
                #     ", end_obj:", objs_name[sub_phrases_end_obj_loc[i]], ", relation:", pred_relationship_word_per_phrase[i])
                start_obj_name.append(objs_name[sub_phrases_start_obj_loc[i]])
                end_obj_name.append(objs_name[sub_phrases_end_obj_loc[i]])
            
            pred_objs_name_all_scenes.append({"org_utterance": adapted_utterance,
                                              "colored_utterance": colored_utterance,
                                              "pred_objs_name": pred_objs_name,
                                              "phrases" : sub_phrases,
                                              "start_obj_name" : start_obj_name,
                                              "end_obj_name" : end_obj_name,
                                              "pred_relationship_word_per_phrase" : pred_relationship_word_per_phrase})
    print("Average number of objs per utterance: ", sum(num_objs_per_scene) / len(num_objs_per_scene))
    print("Number of _00 items are: ", unique_counter)

    # Save the predicted objects in CSV file for manual verification:
    pred_objs_name_all_scenes = random.sample(pred_objs_name_all_scenes, 240)
    save_in_csv(lst=pred_objs_name_all_scenes, saving_name="./data/pred_objs_for_manual_verification.csv")
