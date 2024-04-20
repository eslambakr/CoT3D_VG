
import numpy as np
# from scannet_classes import SCANNET_OBJECTS
import json
#from scipy_parser import SpacyParser 
from extract_objs_from_description import ExtractObjsFromDescription
import re

SINGULAR_UNINFLECTED = ['gas', 'asbestos', 'womens', 'childrens', 'sales', 'physics']

SINGULAR_SUFFIX = [
    ('people', 'person'),
    ('men', 'man'),
    ('wives', 'wife'),
    ('menus', 'menu'),
    ('us', 'us'),
    ('ss', 'ss'),
    ('is', 'is'),
    ("'s", "'s"),
    ('ies', 'y'),
    ('ies', 'y'),
    ('es', 'e'),
    ('s', '')
]

convert_words_to_standards = {"television": "tv", "couches": "couch", "shelves":"shelf"\
                                           ,"bookcase":"bookshelf", "its": "it", "toilet seat" : "toilet",
                                           "doorway":"door", "trashcan":"trash can", "trashcans":"trash can",  
                                           "photos":"photo", 'barstool': 'chair',
                                           "back pack":"backpack", "tub":"bathtub"}
reverse_convert_words_to_standards = {v: k for k, v in convert_words_to_standards.items()}

#sng_parser = SpacyParser()
obj_extractor = ExtractObjsFromDescription("../automatic_loc_module/data/scannet_instance_class_to_semantic_class.json",
                                               coloring_type="[]")
def singularize_word(word):
    for ending in SINGULAR_UNINFLECTED:
        if word.lower().endswith(ending):
            return word
    for suffix, singular_suffix in SINGULAR_SUFFIX:
        if word.endswith(suffix):
            return word[:-len(suffix)] + singular_suffix
    return word

SCANNET_OBJECTS = list(json.load(open('../automatic_loc_module/data/scannet_instance_class_to_semantic_class.json', 'r')).keys())
def match_obj_name(object_, target):
    best_match = 'nan'
    best_len = 10000

    if object_ in convert_words_to_standards:
        object_ = convert_words_to_standards[object_]
    if object_.lower() == 'it' or 'one' in object_.lower():
        object_ = target
    for obj in SCANNET_OBJECTS:
        
        if object_ == 'nan':
            continue
        if object_ == obj:
            return obj
        if singularize_word(object_) == obj:
            return obj
        else:
            for st in object_.split(" "):
                st_ = singularize_word(st)
                object_2 = object_.replace(st, st_)
            if object_2 == obj:
                best_match = obj
        if best_match == 'nan':
            if len(object_.split(" ")) == 1:
                object_2 = singularize_word(object_2)
            else:
                object_2 = object_
            #Do substring matching
            annotated_object = object_2.lower().split(" ")
            temp = annotated_object

            for idx, obj_ in enumerate(annotated_object):
                temp[idx] = singularize_word(obj_)
            annotated_object = temp
            # print(annotated_object)
            scannet_object = obj.lower().split(" ")
            temp = scannet_object
            for idx, obj_ in enumerate(scannet_object):
                temp[idx] = singularize_word(obj_)
            scannet_object = temp
            
            if len(annotated_object) > len(scannet_object) : 
                for left in range(len(scannet_object)):
                    for right in range(left, len(scannet_object)): 
                        if len(" ".join(scannet_object[left:right+1])) > 0 and " ".join(scannet_object[left:right+1]) in object_2  and len(obj) <= best_len and len(obj) >= len(object_2.split(" ")[-1]):
                            best_match = obj
                            best_len = len(obj)
            else: #len(annoted_object) <= len(scannet_object)
                # import pdb; pdb.set_trace()
                for left in range(len(annotated_object)):
                    for right in range(left, len(annotated_object)): 
                        if len(" ".join(annotated_object[left:right+1])) > 0 and " ".join(annotated_object[left:right+1]) in obj and len(obj) <= best_len and len(obj) >= len(object_2.split(" ")[-1]):
                            best_match = obj
                            best_len = len(obj)
                            
            if str(best_match) == 'nan':
                
                #Do the greedy part 
                last  = singularize_word(object_2.split(" ")[-1].split('/')[-1].lower())
                if last == obj:
                    best_match = obj
                else:
                    first = singularize_word(object_2.split(" ")[0].split('/')[0].lower())
                    if first == obj:
                        best_match = first
    return best_match

def clean_caption(caption):
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

    return caption
import ast

def extract_objs_from_description(caption, target, anchors):
    # Extract the numbers from the string
    for i in range(len(anchors)):
        if anchors[i] == "*book rack":
            anchors[i] = "bookshelf"
    
    # objs_ls, _, _, _, _ = obj_extractor.extract_objs_from_description(utterance=caption)            
    # objs = []
    # for obj in objs_ls:
    #     val = match_obj_name(obj, target)
    #     if val != 'nan':
    #         objs.append(val)
    return anchors

def get_relationship_conatining_obj(id, relations_df, obj, target):

    relations = relations_df[relations_df['id'] == id].drop_duplicates(subset=['object', 'subject', 'relation']).copy()
    # print(relations)
    valid_relations = []
    for idx in range(len(relations)):
        relation = relations.iloc[idx]
        # if relation['used_obj'] == True:
        #     print('it is true')
        #     continue
        # obj_name = match_obj_name(relation['object'].lower(),'')
        # sub_name = match_obj_name(relation['subject'].lower(),'')
        obj_name = str(relation['object'])
        if obj_name == "it":
            obj_name = target
        sub_name = str(relation['subject'])
        if sub_name == "it":
            sub_name = target
        # if (obj_name == obj or obj_name in obj or obj in obj_name) and not relation['used_obj']:
        if (obj_name == obj) and not relation['used_obj']:
            valid_relations.append((relation['relation'], sub_name,True))
            # print(relation)
            relations_df.loc[relation.name, 'used_obj'] = True
        # elif (sub_name==obj or sub_name in obj or obj in sub_name) and not relation['used_sub']:
        elif (sub_name==obj) and not relation['used_sub']:
            # print(relation)
            valid_relations.append((relation['relation'], obj_name,False))
            relations_df.loc[relation.name, 'used_sub'] = True
    return valid_relations

def get_closest_relation_mapping(target_string, RELATIONS):
    target_string = target_string.lower()
    words_pred = []
    for rel_word in RELATIONS:
        # if rel_word in sub_phrase.split(" "):
        if re.search(r'\b%s\b' % (re.escape(rel_word.lower())), target_string) is not None:
            words_pred.append(rel_word)
    if len(words_pred) == 0:
            # search the other way around:
        for rel_word in RELATIONS:
            if re.search(r'\b%s\b' % (re.escape(target_string)), rel_word.lower()) is not None:
                words_pred.append(rel_word)
    max_str = ''
    max_len = 0
    if len(words_pred) == 0:
        # get the closest match by finding if the target is a substring of any of the relations:
        for rel_word in RELATIONS:
            if target_string in rel_word.lower() or rel_word.lower() in target_string:
                words_pred.append(rel_word)
    for word in words_pred:
        if len(word) > len(max_str):
                max_str = word
    return max_str


def nearest_3dobject(obj, objs_list):
    # obj is a tuple of (x, y, z) representing the center of the object
    # each object in objs_list is a tuple of (x, y, z) representing the center of the object
    # return the object in objs_list that is nearest to obj using euclidean distance
    # return None if objs_list is empty
    if len(objs_list) == 0:
        return None
    else:
        return min(objs_list, key=lambda x: np.linalg.norm(x - obj))
def farthest_3dobject(obj, objs_list):
    # obj is a tuple of (x, y, z) representing the center of the object
    # each object in objs_list is a tuple of (x, y, z) representing the center of the object
    # return the object in objs_list that is farthest to obj using euclidean distance
    # return None if objs_list is empty
    if len(objs_list) == 0:
        return None
    else:
        return max(objs_list, key=lambda x: np.linalg.norm(x - obj))

def right_3dobject(obj, objs_list):
    # obj is a tuple of (x, y, z) representing the center of the object
    # each object in objs_list is a tuple of (x, y, z) representing the center of the object
    # return the object in objs_list that is on the on the nearest right of to obj on the x axis
    # return None if objs_list is empty
    if len(objs_list) == 0:
        return None
    else:
        return min(objs_list, key=lambda x: x[0] - obj[0])

def left_3dobject(obj, objs_list):
    # Sort objects in objs_list based on their distance from obj on the x-axis
    if len(objs_list) == 0:
        return None
    objs_list_sorted = sorted(objs_list, key=lambda x: abs(x[0]-obj[0]))
    
    # Find the first object in objs_list_sorted that is to the left of obj on the x-axis
    for o in objs_list_sorted:
        if o[0] < obj[0]:
            return o
    
    # If no object is found to the left of obj on the x-axis, return the closest object on the x-axis
    return objs_list_sorted[0]

def on_3dobject(obj, objs_list):
    if len(objs_list) == 0:
        return None
    # Sort objects in objs_list based on their distance from obj on the z-axis
    objs_list_sorted = sorted(objs_list, key=lambda x: abs(x[2]-obj[2]))
    
    # Find the first object in objs_list_sorted that is above obj on the z-axis
    for o in objs_list_sorted:
        if o[2] > obj[2]:
            return o
    
    # If no object is found above obj on the z-axis, return the closest object on the z-axis
    return objs_list_sorted[0]

def under_3dobject(obj, objs_list):
    if len(objs_list) == 0:
        return None
    # Sort objects in objs_list based on their distance from obj on the z-axis
    objs_list_sorted = sorted(objs_list, key=lambda x: abs(x[2]-obj[2]))
    
    # Find the first object in objs_list_sorted that is below obj on the z-axis
    for o in objs_list_sorted:
        if o[2] < obj[2]:
            return o
    
    # If no object is found below obj on the z-axis, return the closest object on the z-axis
    return objs_list_sorted[0]

def front_3dobject(obj, objs_list):
    if len(objs_list) == 0:
        return None
    # Find the object in objs_list that is on the nearest front of obj on the y axis
    front_obj = None
    front_distance = float('inf')
    for o in objs_list:
        if o[1] < obj[1]:  # skip objects behind obj on the y axis
            continue
        distance = abs(o[1]-obj[1])
        if distance < front_distance:
            front_obj = o
            front_distance = distance
    
    return front_obj


def back_3dobject(obj, objs_list):    
    if len(objs_list) == 0:
        return None
    # Find the object in objs_list that is on the nearest back of obj on the y axis
    back_obj = None
    back_distance = float('inf')
    for o in objs_list:
        if o[1] > obj[1]:  # skip objects in front of obj on the y axis
            continue
        distance = abs(o[1]-obj[1])
        if distance < back_distance:
            back_obj = o
            back_distance = distance
    
    return back_obj