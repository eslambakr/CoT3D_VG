import json 
import pandas as pd 
import numpy as np
import argparse
from geometry_module import extract_scene_info
import tqdm

def main():
    parser = argparse.ArgumentParser(description='Extract the 3D location of the referred object in the scene')
    parser.add_argument('--file_path', type=str, default="/home/mohama0e/3D_Codes/CoT3D_VG/automatic_loc_module/new_final_scanrefer++.csv")
    parser.add_argument('--idx_to_semantic_class_file', type=str, default='../automatic_loc_module/referit3d/data/mappings/scannet_idx_to_semantic_class.json')
    parser.add_argument('--instance_class_to_semantic_class_file', type=str, default='../automatic_loc_module/referit3d/data/mappings/scannet_instance_class_to_semantic_class.json')
    parser.add_argument('--axis_alignment_info_file', type=str, default='../automatic_loc_module/referit3d/data/scannet/scans_axis_alignment_matrices.json')
    parser.add_argument('--file_type', type=str, default='nr3d', help='nr3d or scanrefer')
    parser.add_argument('--phrase_refer_path', type=str, default='scanrefer++.json', help='nr3d or scanrefer')
    parser.add_argument('scanrefer_file_path', type=str, default='/home/mohama0e/scanRefer.csv')
    args = parser.parse_args()
    
    scanrefer = json.load(open(args.phrase_refer_path))
    scenes_dict, relation_fn_dict, reverse_relation_fn_dict, mapped_relation,df = extract_scene_info(args)
    final_labels = []
    unfound_scan_ids = []
    total_not_found = 0
    for i in tqdm(range(len(scanrefer))):
        dt_dict = scanrefer[i]
        scan_id = dt_dict['scene_id']
        dt_id   = dt_dict['labeled_id']
        # if(len(updated_label_list) > 1):
            #Now, use the box_ids
        if scan_id not in scenes_dict.keys():
            unfound_scan_ids.append(scan_id)
            continue 
        found = False
        for obj_3d in scenes_dict[scan_id].three_d_objects:
            truth_label = obj_3d.instance_label
            GT_id       = obj_3d.object_id
            if(str(GT_id)== dt_id):
                final_labels.append(truth_label)
                found = True
                break
        if not found:
            total_not_found +=1
            
    original_scanrefer = pd.read_csv(args.scanrefer_file_path)

    all_data = []
    mapper = {}
    box_ids= {}
    utterance = {}
    uni_keys = {}
    pos_keys = {}
    j = 0
    #Create ScanRefer dataset
    for i in tqdm(range(len(scanrefer))):
        ann_id     = scanrefer[i]['ann_id']
        scene_id   = scanrefer[i]['scene_id']
        object_id  = scanrefer[i]['object_id']
        box_id     = scanrefer[i]['labeled_id']
        description= scanrefer[i]['description']
        pos_start  = scanrefer[i]['position_start']
        key = ann_id + '_' + scene_id + '_' + object_id
        # if scene_id in unfound_scan_ids:
        #     continue
        if key not in mapper.keys():
            mapper[key] = []
            box_ids[key] = []
            utterance[key] = ""
            uni_keys[key] = ""
            pos_keys[key] = []
        utterance[key] = description
        uni_keys[key] = scene_id+ "_" + description
        mapper[key].append(final_labels[j])
        pos_keys[key].append(pos_start)
        j+=1
        box_ids[key].append(box_id)            
        
        all_data = {"path": list(mapper.values())}
        df_path  = pd.DataFrame(all_data)
        all_data = {"anchor_ids": list(box_ids.values())}
        df_anchor_ids = pd.DataFrame(all_data)
        all_data = {"uni_keys": list(uni_keys.values())}
        df_scene_ids = pd.DataFrame(all_data)
        all_data     = {"pos_keys": list(pos_keys.values())}
        pos_keys_df  = pd.DataFrame(all_data)
        
        #Get a csv File with the following columns: "assignment_id", "utterance", "anchors", box_ids
        assignment_ids = mapper.keys()

        #Get the data for each assignment id in a dataframe 
        final_df = pd.DataFrame({'assignment_id': assignment_ids, 'utterance': utterance.values()})
        final_df['path'] = df_path['path']
        final_df['anchor_ids'] = df_anchor_ids['anchor_ids']
        final_df['uni_keys']  = df_scene_ids['uni_keys']
        final_df['pos_keys']  = pos_keys_df['pos_keys']
        
        #create Unique Keys
        original_scanrefer['uni_keys'] = original_scanrefer.apply(lambda row: str(row['stimulus_id']) + '_' + str(row['utterance']), axis=1)
        
        #join scanrefer and final_df according to the utterance 
        #     """

        merged_df = pd.merge(original_scanrefer,final_df, on='uni_keys')
        
        new_path = {}
        new_boxes= {}

        for i in range(len(merged_df)):
        
            
            target_name = merged_df.iloc[i]['instance_type']
            target_id   = merged_df.iloc[i]['target_id']
            
            path = []
            box  = []
            
            current_path = merged_df.iloc[i]['path']
            current_boxes= merged_df.iloc[i]['anchor_ids']
            checker = {}
            
            for j in range(len(current_path)):
                inp = current_path[j] + "_" + str(current_boxes[j])
                if inp not in checker:
                    checker[inp] = 1
                else:
                    checker[inp] +=1
            for j in range(len(current_path)):
                if current_path[j] != target_name and str(current_boxes[j]) != str(target_id):
                    path.append(current_path[j])
                    box.append(current_boxes[j])
                elif current_path[j] == target_name and str(current_boxes[j]) == str(target_id) and checker[current_path[j] + "_" + str(current_boxes[j])] > 1:
                    path.append(current_path[j])
                    box.append(current_boxes[j])
                    checker[current_path[j] + "_" + str(current_boxes[j])] -=1
                
            path.append(target_name)
            box.append(target_id)
            
            new_path[str(i)] = path
            new_boxes[str(i)] = box
            
            
        new_boxes
        all_data = {"anchor_ids": list(new_boxes.values())}
        anchor_ids  = pd.DataFrame(all_data)
        all_data = {"path": list(new_path.values())}
        path     = pd.DataFrame(all_data)
        merged_df['path'] = path['path']
        merged_df['anchor_ids'] = anchor_ids['anchor_ids']
        exact_match_found = 0
        substring_match_found = 0
        not_found_at_all = 0
        total = 0
        new_path = []

        for i in range(len(merged_df)):
            final_path = []
            final_anchors= []
            paths = merged_df.iloc[i]['path']
            anchors= merged_df.iloc[i]['anchor_ids']
            utterance= merged_df.iloc[i]['utterance_x']
            add_path = []
            for j in range(len(paths) - 1):
                pos = merged_df.iloc[i]['pos_keys'][j]
                path= paths[j]
                anchor=anchors[j]
                if path in utterance:
                    substring_match_found += 1
                    #find the path in utterance index 
                    idx = utterance.index(path)
                    add_path.append([int(idx), path, anchor])
                else:
                    not_found_at_all += 1
                    add_path.append([int(pos), path, anchor])
                total +=1
                
            #sort_descendingly by index first one in add_path
            add_path.sort(key=lambda x: x[0], reverse=False)
            for k in range(len(add_path)):
                final_path.append(add_path[k][1])
                final_anchors.append(add_path[k][2])
            final_path.append(paths[-1])
            final_anchors.append(anchors[-1])
            merged_df.at[i,'path'] = final_path
            merged_df.at[i,'anchor_ids'] = final_anchors
            
        merged_df.to_csv('final_scanrefer++.csv', index=False)