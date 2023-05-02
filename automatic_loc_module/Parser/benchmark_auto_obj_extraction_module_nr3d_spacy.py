import numpy as np
from tqdm import tqdm
import csv
# from referit3d.in_out.scannet_scan import ScannetScan, ScannetDataset
import pandas as pd
import string
# from extract_objs_from_description import ExtractObjsFromDescription
import random
from scipy_parser import SpacyParser 

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
    df = read_referring_data_scv(file_path="../data/nr3d.csv")
    scan_ids = df.scan_id
    gt_objs_name_all_scenes = []
    gt_utternaces_all_scenes = []
    pred_objs_name_all_scenes = []
    unique_counter = 0
    num_objs_per_scene = []
    mx_len = -1
    sng_parser = SpacyParser()
    # Create our obj retrieval module:
    # obj_extractor = ExtractObjsFromDescription("./data/scannet_instance_class_to_semantic_class.json",
    #                                            coloring_type="[]")
    objects = []
    subjects = []
    relations = []
    captions = []
    ids = []
    for i in tqdm(range(len(scan_ids))):
        scan_id = scan_ids[i]
        if "_00" in scan_id:
            unique_counter = unique_counter + 1
            # Get Ground-Truth anchors and target objects:
            caption = df.utterance[i]
            caption = ' '.join(caption.replace(',', ' , ').split())

            # some error or typo in ScanRefer.
            caption = ' '.join(caption.replace("'m", "am").split())
            caption = ' '.join(caption.replace("'s", "is").split())
            caption = ' '.join(caption.replace("2-tiered", "2 - tiered").split())
            caption = ' '.join(caption.replace("4-drawers", "4 - drawers").split())
            caption = ' '.join(caption.replace("5-drawer", "5 - drawer").split())
            caption = ' '.join(caption.replace("8-hole", "8 - hole").split())
            caption = ' '.join(caption.replace("7-shaped", "7 - shaped").split())
            caption = ' '.join(caption.replace("2-door", "2 - door").split())
            caption = ' '.join(caption.replace("3-compartment", "3 - compartment").split())
            caption = ' '.join(caption.replace("computer/", "computer /").split())
            caption = ' '.join(caption.replace("3-tier", "3 - tier").split())
            caption = ' '.join(caption.replace("3-seater", "3 - seater").split())
            caption = ' '.join(caption.replace("4-seat", "4 - seat").split())
            caption = ' '.join(caption.replace("theses", "these").split())
            
            # some error or typo in NR3D.
            # if anno['dataset'] == 'nr3d':
            caption = ' '.join(caption.replace('.', ' .').split())
            caption = ' '.join(caption.replace(';', ' ; ').split())
            caption = ' '.join(caption.replace('-', ' ').split())
            caption = ' '.join(caption.replace('"', ' ').split())
            caption = ' '.join(caption.replace('?', ' ').split())
            caption = ' '.join(caption.replace("*", " ").split())
            caption = ' '.join(caption.replace(':', ' ').split())
            caption = ' '.join(caption.replace('$', ' ').split())
            caption = ' '.join(caption.replace("#", " ").split())
            caption = ' '.join(caption.replace("/", " / ").split())
            caption = ' '.join(caption.replace("you're", "you are").split())
            caption = ' '.join(caption.replace("isn't", "is not").split())
            caption = ' '.join(caption.replace("thats", "that is").split())
            caption = ' '.join(caption.replace("doesn't", "does not").split())
            caption = ' '.join(caption.replace("doesnt", "does not").split())
            caption = ' '.join(caption.replace("itis", "it is").split())
            caption = ' '.join(caption.replace("left-hand", "left - hand").split())
            caption = ' '.join(caption.replace("[", " [ ").split())
            caption = ' '.join(caption.replace("]", " ] ").split())
            caption = ' '.join(caption.replace("(", " ( ").split())
            caption = ' '.join(caption.replace(")", " ) ").split())
            caption = ' '.join(caption.replace("wheel-chair", "wheel - chair").split())
            caption = ' '.join(caption.replace(";s", "is").split())
            caption = ' '.join(caption.replace("tha=e", "the").split())
            caption = ' '.join(caption.replace("it’s", "it is").split())
            caption = ' '.join(caption.replace("’s", " is").split())
            caption = ' '.join(caption.replace("isnt", "is not").split())
            caption = ' '.join(caption.replace("Don't", "Do not").split())
            caption = ' '.join(caption.replace("arent", "are not").split())
            caption = ' '.join(caption.replace("cant", "can not").split())
            caption = ' '.join(caption.replace("you’re", "you are").split())
            caption = ' '.join(caption.replace('!', ' !').split())
            caption = ' '.join(caption.replace('id the', ' , the').split())
            caption = ' '.join(caption.replace('youre', 'you are').split())

            caption = ' '.join(caption.replace("'", ' ').split())

            if caption[0] == "'":
                caption = caption[1:]
            if caption[-1] == "'":
                caption = caption[:-1]
            
            # anno['utterance'] = caption

            # text parsing

            graph_node, graph_edge = sng_parser.parse(caption)
            
            
            unique_objs = set()
            for edge in graph_edge:
                if  True: #edge['subject'].lower() != "it" and edge['object'].lower() != "it":
                    objects.append(edge['object'])
                    subjects.append(edge['subject'])
                    relations.append(edge['relation'])
                    captions.append(caption)
                    unique_objs.add(edge['object'])
                    unique_objs.add(edge['subject'])
                    ids.append(i)
            obj_len = len(unique_objs)
            mx_len = max(mx_len, obj_len)
            # if obj_len >10 :
            #     print(caption)
            #     for obj in unique_objs:
            #         print(obj)
            #     assert False
            num_objs_per_scene.append(obj_len)

            # pred_objs_name_all_scenes.append({"caption": caption for object in,
            #                                 "object": objects,
            #                                 "relation": relations,
            #                                 "subject" : subjects})
    data = {"id": ids,
            "caption": captions,
            "object": objects,
            "relation": relations,
            "subject" : subjects}
    print("Average number of objs per utterance: ", sum(num_objs_per_scene) / len(num_objs_per_scene))
    print("Number of _00 items are: ", unique_counter)
    print("Max number of objects in a sentence is: ", mx_len)

    # Save the predicted objects in CSV file for manual verification:
    # pred_objs_name_all_scenes = random.sample(pred_objs_name_all_scenes, 240)
    new_df = pd.DataFrame(data)

    # save DataFrame to CSV file
    new_df.to_csv("./relations.csv", index=False)
