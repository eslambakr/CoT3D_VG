from tqdm import tqdm
from referit3d.in_out.scannet_scan import ScannetScan, ScannetDataset
import pandas as pd
import string
from extract_objs_from_description import ExtractObjsFromDescription


def read_referring_data_scv(file_path):
    df = pd.read_csv(file_path)
    return df


if __name__ == '__main__':
    df = read_referring_data_scv(file_path="./data/sr3d.csv")
    scan_ids = df.scan_id
    gt_objs_name_all_scenes = []
    gt_utternaces_all_scenes = []
    pred_objs_name_all_scenes = []
    unique_counter = 0
    num_objs_per_scene = []
    # Create our obj retrieval module:
    obj_extractor = ExtractObjsFromDescription("./data/scannet_instance_class_to_semantic_class.json")

    for i in tqdm(range(len(scan_ids))):
        scan_id = scan_ids[i]
        if True or ("_00" in scan_id):
            unique_counter = unique_counter + 1
            # Get Ground-Truth anchors and target objects:
            gt_objs_name = [df.instance_type[i]]
            anchors = df.anchors_types[i][1:-1].split(',')
            for anchor in anchors:
                gt_objs_name.append(anchor.translate(str.maketrans('', '', string.punctuation)).strip())
            gt_objs_name_all_scenes.append(gt_objs_name)
            gt_utternaces_all_scenes.append(df.utterance[i])

            # Run our obj retrieval module:
            pred_objs_name = obj_extractor.extract_objs_from_description(utterance=df.utterance[i])
            num_objs_per_scene.append(len(pred_objs_name))
            pred_objs_name_all_scenes.append(pred_objs_name)

    print("Average number of objs per utterance: ", sum(num_objs_per_scene)/len(num_objs_per_scene))
    print("Number of _00 items are: ", unique_counter)
    # Calculate the Acc:
    true_pos = 0
    false_pos = 0
    for i in tqdm(range(len(pred_objs_name_all_scenes))):
        for pred_obj in pred_objs_name_all_scenes[i]:
            true_pos_flag = False
            for gt_obj in gt_objs_name_all_scenes[i]:
                if pred_obj in gt_obj:
                    true_pos_flag = True
                    true_pos += 1
                    break
            if not true_pos_flag:
                print(pred_obj, "------>", gt_utternaces_all_scenes[i])
                false_pos += 1

    false_neg = 0
    for i in tqdm(range(len(gt_objs_name_all_scenes))):
        for gt_obj in gt_objs_name_all_scenes[i]:
            false_neg_flag = True
            for pred_obj in pred_objs_name_all_scenes[i]:
                if gt_obj in pred_obj:
                    false_neg_flag = False
                    break
            if false_neg_flag:
                print(gt_obj, "-->", gt_utternaces_all_scenes[i])
                false_neg += 1

    precision = true_pos / (true_pos+false_pos)  # top, center, side, back
    # trash can, laundry hamper, bathroom stall door, paper towel dispenser, coffee table, kitchen cabinet,
    # office chair, copier, kitchen cabinets, end table, kitchen counter, file cabinet, oven
    recall = true_pos / (true_pos+false_neg)
    print("precision: ", precision * 100, "%")
    print("recall: ", recall * 100, "%")
