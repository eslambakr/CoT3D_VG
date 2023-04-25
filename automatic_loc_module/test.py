import spacy
import json
import string
import time
from spacy.matcher import Matcher
from spacy import displacy
import re
import pandas as pd
import numpy as np
from copy import deepcopy


class Colour:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


class ExtractObjsFromDescription:
    def __init__(self, scannet_inst_cls_to_sem_cls_pth, coloring_type):
        ins_sem_cls_map_dict = self.read_cls_json(scannet_inst_cls_to_sem_cls_pth)
        self.cls_names = list(ins_sem_cls_map_dict.keys())
        self.nlp = spacy.load("en_core_web_md")
        self.exclude_word_list = ["side", "back", "center", "top"]
        self.include_word_list = ["it", "its", "it's"]
        # Define relationships words:
        self.above_words = ['aboard', 'above', 'on', 'onto', 'over', 'up', 'upper']
        self.below_words = ['low', 'lower', 'lowest', 'below', 'beneath', 'down', 'under', 'underneath']
        self.far_words = ['far', 'farthest', 'furthest', 'away', 'ahead']
        self.near_words = ['nearest', 'holding', 'holds', 'supporting', 'supports', 'side', 'closest', 'along',
                           'alongside', 'near', 'nearby', 'apart', 'around', 'beside', 'inside', 'into', 'within',
                           'aside', 'astride', 'at', 'corner', 'through', 'together', 'toward',
                           'by', 'next', 'close', 'about', 'round']
        self.between_words = ['amid', 'amidst', 'among', 'amongst', 'between', 'middle', 'center']
        self.other_words = ['across', 'adjacent', 'against', 'behind', 'beyond', 'opposite', 'out', 'outside',
                            'left', 'right', 'front', 'back', 'on the right', 'on the left']
        self.plural_letters = ['s', 'ss', 'sh', 'ch', 'x', 'z']
        self.coloring_type = coloring_type

    def read_cls_json(self, file_name):
        f = open(file_name)
        ins_sem_cls_map = json.load(f)
        return ins_sem_cls_map

    def get_nouns_by_matching_cls(self, utterance, nouns):
        temp_utterance = deepcopy(utterance)
        # Remove the detected nouns so far from the utterance:
        for noun in nouns:
            try:
                temp_utterance = re.sub(r"\b%s\b" % noun, "", temp_utterance)  # remove whole word only
            except: # if the noun is not a whole word
                temp_utterance = temp_utterance.replace(noun, "")
        # Try to detect the rest of them if some nouns are missed from the first step:
        # Get the whole text combinations
        words = temp_utterance.split(" ")
        # Option #1:
        for c_idx, c_word in enumerate(words):
            combined_word = ""
            correct_word = None
            for f_idx in range(c_idx, len(words)):
                combined_word = combined_word + words[f_idx] + " "
                # Check word by word. if not empty list means word is correct
                if self.nouns_cls_alignment_exact_matching([combined_word.strip()]):
                    correct_word = combined_word.strip()
                    words[f_idx] = "removed"  # remove the word to not consider it again in the future
                else:
                    break
            if correct_word:
                nouns += [correct_word]

        # Option #2:
        """
        phrases = []
        for c_idx, c_word in enumerate(words):
            combined_word = ""
            for f_idx in range(c_idx, len(words)):
                combined_word += words[f_idx]
                combined_word += " "
                phrases.append(combined_word.strip())
        # Remove duplicates:
        phrases = list(dict.fromkeys(phrases))
        pred_counts = self.nouns_cls_alignment_exact_matching(phrases)
        nouns += pred_counts
        """

        return nouns

    def get_nouns(self, ip_sentence):
        nouns = self.get_nouns_spacy(ip_sentence)
        nouns = self.get_nouns_by_matching_cls(ip_sentence, nouns)
        return nouns

    def get_nouns_spacy(self, ip_sentence):
        doc = self.nlp(ip_sentence)
        nouns = []
        for word_idx, token in enumerate(doc):
            if token.pos_ == "NOUN":
                nouns.append(token.text)
            # handel corner cases: (add user defined objs)
            elif token.text == "bin" or token.text == "copier" or token.text == "bookshelf" or token.text == "stall" \
                    or token.text == "oven" or token.text == "cabinets":
                nouns.append(token.text)

            # Handel corner cases:
            if (token.text == "hamper") and (doc[word_idx - 1].text == "laundry"):
                if (len(nouns) >= 1) and nouns[-1] == "hamper":
                    if (len(nouns) >= 2) and nouns[-2] == "laundry":
                        nouns.pop()
                    nouns.pop()
                elif (len(nouns) >= 1) and nouns[-1] == "laundry":
                    nouns.pop()
                nouns.append("laundry hamper")
            elif (token.text == "trash") and ((word_idx + 1) < len(doc)) and (doc[word_idx + 1].text == "can"):
                if (len(nouns) >= 1) and nouns[-1] == "trash":
                    nouns.pop()
                nouns.append("trash can")
            elif (token.text == "can") and (doc[word_idx - 1].text == "trash"):
                if (len(nouns) >= 1) and nouns[-1] == "can":
                    nouns.pop()
            elif (token.text == "bin") and (doc[word_idx - 1].text == "storage") and (
                    doc[word_idx - 2].text == "plastic"):
                if (len(nouns) >= 1) and nouns[-1] == "bin":
                    if (len(nouns) >= 2) and nouns[-2] == doc[word_idx - 1].text:
                        if (len(nouns) >= 3) and nouns[-3] == doc[word_idx - 2].text:
                            nouns.pop()
                        nouns.pop()
                    nouns.pop()
                elif (len(nouns) >= 1) and nouns[-1] == doc[word_idx - 1].text:
                    if (len(nouns) >= 2) and nouns[-2] == doc[word_idx - 2].text:
                        nouns.pop()
                    nouns.pop()
                elif (len(nouns) >= 1) and nouns[-1] == doc[word_idx - 2].text:
                    nouns.pop()
                nouns.append(doc[word_idx - 2].text + " " + doc[word_idx - 1].text + " " + token.text)
            elif (token.text == "bin") and (
                    (doc[word_idx - 1].text == "compost") or (doc[word_idx - 1].text == "diaper")
                    or (doc[word_idx - 1].text == "mail") or (doc[word_idx - 1].text == "plastic")
                    or (doc[word_idx - 1].text == "recycling") or (doc[word_idx - 1].text == "storage")
                    or (doc[word_idx - 1].text == "trash")):
                if (len(nouns) >= 1) and nouns[-1] == "bin":
                    if (len(nouns) >= 2) and nouns[-2] == doc[word_idx - 1].text:
                        nouns.pop()
                    nouns.pop()
                elif (len(nouns) >= 1) and nouns[-1] == doc[word_idx - 1].text:
                    nouns.pop()
                nouns.append(doc[word_idx - 1].text + " " + token.text)
            # print(token.text, token.pos_)
            # print("HERE" , nouns)
        print(nouns)
        return nouns

    def get_nouns_loc_in_txt(self, ip_sentence, nouns):
        """
        Get the locations of the input nouns from the input sentence.
        :return: list of locations. the list length should equal the length of the nouns.
        """
        new_string = ip_sentence.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
        # import pdb; pdb.set_trace()
        nouns_loc = []
        words = new_string.split(" ")
        for noun in nouns:
            if noun in words:
                temp_noun = noun
            elif noun.translate(str.maketrans('', '', string.punctuation)) in words:  # handle the case of it, its, it's
                temp_noun = noun.translate(str.maketrans('', '', string.punctuation))
            else:  # this means the name is already combined
                temp_noun = noun.split(" ")[0]
            loc = words.index(temp_noun)
            nouns_loc.append(loc)
            # remove the word to handel the case where two repeated words are mentioned
            words[loc] = "Nan"
        return nouns_loc

    def obj_names_refinement(self, ip_sentence, nouns):
        """
        merge names for the same object together
        :return:
        """
        # Get nouns locations:
        nouns_loc = self.get_nouns_loc_in_txt(ip_sentence, nouns)

        # If there are two consecutive nouns, then merge them. (e.g., office chair)
        combined_nouns = []
        i = 0
        while i < len(nouns_loc):
            combine_idx = i
            for j in range(i + 1, len(nouns_loc)):
                if nouns_loc[j] - nouns_loc[i] == j - i:  # forward combination only
                    combine_idx = j
                else:
                    break

            combined_noun = ""
            combine_flag = False
            for k in range(i, combine_idx + 1):
                combine_flag = True
                if combined_noun == "":  # handle the first time
                    combined_noun = nouns[k]
                else:
                    # check if the combined word will match class or not. If no, then keep the word as it is.
                    temp_noun = combined_noun + " " + nouns[k]
                    if self.nouns_cls_alignment(nouns=[temp_noun]) == [temp_noun]:
                        combined_noun = combined_noun + " " + nouns[k]
                    else:
                        combined_noun = nouns[k]
            if combine_flag:
                combined_nouns.append(combined_noun.strip())
            i = combine_idx + 1

        return combined_nouns

    def nouns_cls_alignment(self, nouns):
        """
        Align the detected nouns with the predefined classes of the data set.
        :return:
        aligned_nouns: the nouns after alignment.
        """
        obj_nouns = []
        # Keep objects only: (remove other names)
        for noun in nouns:
            found = False
            for cls_name in self.cls_names:
                if noun in cls_name:
                    found = True
                    obj_nouns.append(noun)
                    break
                # Handle the plural letters
                for plural_letter in self.plural_letters:
                    if noun in cls_name+plural_letter:
                        found = True
                        obj_nouns.append(noun)
                        break
                if found:
                    break
                # Handle the spaces:
                if (noun.replace(" ", "") in cls_name) or (noun in cls_name.replace(" ", "")):
                    obj_nouns.append(noun)
                    break

        return obj_nouns

    def nouns_cls_alignment_exact_matching(self, nouns, check_substring = False):
        """
        Align the detected nouns with the predefined classes of the data set.
        :return:
        aligned_nouns: the nouns after alignment.
        """
        
        obj_nouns = []
        sorted_cls_names = sorted(self.cls_names)
        from bisect import bisect_left
        def BinarySearch(a, x):
            i = bisect_left(a, x)
            if i != len(a) and a[i] == x:
                return True
            else:
                return False
        for noun in nouns:
            # if check_substring:
            #     import pdb; pdb.set_trace()
            found = False
            
            if noun == "side" or noun == "set":
                continue
            # import pdb; pdb.set_trace()
            #Do Exact Match with a help of Binary Search
            if(BinarySearch(sorted_cls_names, noun) == True):
                obj_nouns.append(noun)
                continue    
            
            for cls_name in self.cls_names:
                
                # # Exact matching:
                # if noun == cls_name:
                #     obj_nouns.append(noun)
                #     break
                # Handle the plural letters:
                for plural_letter in self.plural_letters:
                    if noun == cls_name+plural_letter:
                        found = True
                        obj_nouns.append(noun)
                        break
                if found:
                    break
                # Handle the spaces:
                if (noun.replace(" ", "") == cls_name) or (noun == cls_name.replace(" ", "")):
                    obj_nouns.append(noun)
                    break
                if (check_substring):
                    found = False
                    substring_cls_name = cls_name.split(" ") 
                    #Check if the noun is a substring of the class name
                    for cls_name_substring in substring_cls_name:
                        if noun == cls_name_substring:
                            obj_nouns.append(cls_name)
                            found = True
                            break
                    if (found):
                        break
        return obj_nouns

    def exclude_user_defined_nouns(self, nouns):
        for i, noun in enumerate(nouns):
            if len(noun.split(" ")) == 1:
                if noun in self.exclude_word_list:
                    del nouns[i]

        return nouns

    def include_user_defined_nouns(self, utterance, nouns):
        for i, word in enumerate(self.include_word_list):
            if word in utterance.split(" "):
                nouns.append(word)

        return nouns

    def exclude_symbols(self, utterance):
        utterance = utterance.replace("/", " or ")
        utterance = utterance.replace("-", " ")
        utterance = utterance.replace(".", " ")
        utterance = utterance.replace("=", " ")
        utterance = utterance.replace("'", " ")
        utterance = utterance.replace(",", " ")
        utterance = utterance.replace("’", " ")
        utterance = utterance.replace("[", " ")
        utterance = utterance.replace("]", " ")
        utterance = utterance.replace("(", " ")
        utterance = utterance.replace(")", " ")
        return utterance

    def remove_misspelling(self, utterance):
        new_string = ' '.join([w for w in utterance.split() if len(w) > 1])  # remove one character word.
        return new_string

    def add_the(self, utterance):
        if utterance.split(" ")[0] in self.cls_names:  # check if the first word is an object
            # add the to be detected as a noun by the spacy toolkit.
            utterance = "the " + utterance
        return utterance

    def get_start_and_end_loc_of_objs(self, ip_sentence, nouns):
        """
        Save the location of each word
        :param ip_sentence:
        :param nouns:
        :return:
        """
        objs_start_loc = self.get_nouns_loc_in_txt(ip_sentence=ip_sentence, nouns=nouns)
        objs_end_loc = []
        for obj_idx, pred_obj in enumerate(nouns):
            objs_end_loc.append(objs_start_loc[obj_idx] - 1 + len(pred_obj.split(" ")))

        # Create mask for the location of objects in the utterance:
        pred_objs_mask = np.zeros(len(ip_sentence.split(" ")))
        for obj_idx, pred_obj in enumerate(nouns):
            pred_objs_mask[objs_start_loc[obj_idx]:objs_end_loc[obj_idx] + 1] = 1
        return objs_start_loc, objs_end_loc, pred_objs_mask

    def add_nouns_loc_to_org_utterance(self, ip_sentence, pred_objs_mask):
        # Add colors to the detected objects:
        colored_utterance = ""
        for word_idx, word in enumerate(ip_sentence.split(" ")):
            if pred_objs_mask[word_idx]:
                if self.coloring_type == "colors":
                    colored_utterance += Colour.RED + str(word) + Colour.END + " "
                else:
                    colored_utterance += "[" + str(word) + "] "
            else:
                colored_utterance += word + " "
        return colored_utterance

    def extract_objs_from_description(self, utterance):
        print(utterance)
        org_utterance = utterance.lower()
        # Exclude symbols:
        utterance = self.exclude_symbols(org_utterance)
        utterance = self.remove_misspelling(utterance)
        # Add The:
        temp_utterance = self.add_the(utterance)
        # Extract nouns of the utterance using spacy:
        nouns = self.get_nouns(ip_sentence=temp_utterance)
        print("get_nouns = ", nouns)
        # Align the detected nouns with the predefined classes of the data set:
        obj_nouns = self.nouns_cls_alignment(nouns)
        print("nouns_cls_alignment = ", obj_nouns)
        # Merge names for the same object together:
        obj_nouns_refined = self.obj_names_refinement(utterance, obj_nouns)
        print("obj_names_refinement = ", obj_nouns_refined)
        # Align the detected nouns again with the predefined classes of the data set after refining them:
        obj_nouns_refined = self.nouns_cls_alignment_exact_matching(obj_nouns_refined,  check_substring = True)
        print("nouns_cls_alignment_exact_matching = ", obj_nouns_refined)
        # Exclude user defined nouns:
        obj_nouns_refined = self.exclude_user_defined_nouns(obj_nouns_refined)
        # Add defined nouns:
        obj_nouns_refined = self.include_user_defined_nouns(utterance=utterance, nouns=obj_nouns_refined)
        # Save the location of each word:
        import pdb; pdb.set_trace()
        objs_start_loc, objs_end_loc, pred_objs_mask = self.get_start_and_end_loc_of_objs(ip_sentence=utterance,
                                                                                          nouns=obj_nouns_refined)
        # Add colors to the detected objects:
        colored_utterance = self.add_nouns_loc_to_org_utterance(ip_sentence=utterance,
                                                                pred_objs_mask=pred_objs_mask)

        return obj_nouns_refined, objs_start_loc, objs_end_loc, colored_utterance, utterance

   



if __name__ == '__main__':
    """
    near acomp (adjectival complement)
    to prep (prepositional modifier)
    behind prep
    farthest acomp
    from prep 
    above prep
    """
    # utterance = "find the office chair that is near to the other office chair which behind the copier."
    # utterance = "the monitor that is upper the printer"
    # utterance = "Wall lamp between two posters and behind the table"
    # utterance   = "LOOK DOWN ON ROOM FROM CEILING, CHOOSE BOX LOCATED IN THE CENTER OF THE ROOM, ITS ALSO THE LARGEST BOX TO SELECT."
    # utterance   = "LOOK DOWN ON THE ROOM AS YOU IF YOUR LOOKING IN FROM THE CEILING OF THE ROOM. CHOOSE THE TABLE THAT HAS ONLY 2 CHAIRS AROUND IT. THE OTHER TABLE HAS 4 CHAIRS."
    utterance   = "The inner set of curtains"
    # choose the trash can that is in the center of the door and the whiteboard
    # print(utterance)
    obj_extractor = ExtractObjsFromDescription("./data/scannet_instance_class_to_semantic_class.json",
                                               coloring_type="colors")
    objs_name, objs_start_loc, objs_end_loc,\
        colored_utterance, adapted_utterance = obj_extractor.extract_objs_from_description(utterance=utterance)
        
    tgt_from_txt = obj_extractor.get_subject_phrase(ip_sentence=utterance.lower())
    print("colored_utterance = ", colored_utterance)
    print("objs_name = ", objs_name)
    print("tgt_from_txt = ", tgt_from_txt)

    # Get relationships words:
    print("Extract the phrases between two objects:")
    sub_phrases, sub_phrases_start_obj_loc, sub_phrases_end_obj_loc = obj_extractor.get_phrases_between_2_objs(
        ip_sentence=utterance.lower(), objs_name=objs_name)
    print("Extract the relationship between two objects:")
    # import pdb; pdb.set_trace()
    pred_relationship_word_per_phrase = obj_extractor.get_relationship_between_2_objs(sub_phrases)
    for i, sub_phrase in enumerate(sub_phrases):
        print(i, "--> phrase:", sub_phrase, ", start_obj:", objs_name[sub_phrases_start_obj_loc[i]],
              ", end_obj:", objs_name[sub_phrases_end_obj_loc[i]], ", relation:", pred_relationship_word_per_phrase[i])


# if __name__ == '__main__':
    
#     def read_referring_data_scv(file_path):
#         df = pd.read_csv(file_path)
#         return df

#     df = read_referring_data_scv(file_path="./data/nr3d.csv")
#     scan_ids = df.scan_id
#     gt_objs_name_all_scenes = []
#     gt_utternaces_all_scenes = []
#     pred_objs_name_all_scenes = []
#     unique_counter = 0
#     num_objs_per_scene_without_matching = []
#     num_objs_per_scene_with_matching = []
#     # Create our obj retrieval module:
#     obj_extractor = ExtractObjsFromDescription("./data/scannet_instance_class_to_semantic_class.json",
#                                                coloring_type="[]")
#     from tqdm import tqdm
#     num_examples_differ = 0
#     object_diffs = []
#     for i in tqdm(range(len(scan_ids))):
#         scan_id = scan_ids[i]
#         if "_00" in scan_id:
#             unique_counter = unique_counter + 1
#             # Get Ground-Truth anchors and target objects:

#             # Run our obj retrieval module:
#             refined_data_without_matching, refined_data_with_matching, utterance= obj_extractor.extract_objs_from_description(
#                 utterance=df.utterance[i])
#             if len(refined_data_without_matching) != len(refined_data_with_matching):
#                 num_examples_differ = num_examples_differ + 1
#                 object_diffs.append(len(refined_data_without_matching) - len(refined_data_with_matching))
#                 pred_objs_name_all_scenes.append({"org_utterance": df.utterance[i], "colored_utterance": utterance,
#                                               "pred_objs_name_without_matching": refined_data_without_matching,
#                                               "pred_objs_name_with_matching": refined_data_with_matching})
#             num_objs_per_scene_without_matching.append(len(refined_data_without_matching))
#             num_objs_per_scene_with_matching.append(len(refined_data_with_matching))
#             """
#             pred_objs_name_all_scenes.append({"adapted_utterance": adapted_utterance, "org_utterance": df.utterance[i],
#                                               "colored_utterance": colored_utterance, "missed_objs": [],
#                                               "pred_objs_name": pred_objs_name})
#                                               """
#             # pred_objs_name_all_scenes.append({"org_utterance": df.utterance[i], "colored_utterance": utterance,
#             #                                   "pred_objs_name_without_matching": refined_data_without_matching,
#             #                                   "pred_objs_name_with_matching": refined_data_with_matching})

#     #Know the percentage in With matching with "without matching"
#     print("Average number of objs per utterance WITH: ", sum(num_objs_per_scene_with_matching) / len(num_objs_per_scene_with_matching))
#     print("Average number of objs per utterance WITHOUT: ", sum(num_objs_per_scene_without_matching) / len(num_objs_per_scene_without_matching))
    
#     #Percantage for the number of objects in the secene with matching compare without matching
#     avg = sum(num_objs_per_scene_with_matching) / sum(num_objs_per_scene_without_matching)
#     print("Percentage of objects in the secene with matching compare without matching: ", avg)
    
#     print("Number of examples that differ: ", num_examples_differ)
    
#     print("Number of _00 items are: ", unique_counter)  

#     # Save the predicted objects in CSV file for manual verification:
#     # pred_objs_name_all_scenes = random.sample(pred_objs_name_all_scenes, 240)
#     import csv
#     def save_in_csv(lst, saving_name):
#         # Save output in csv:
#         keys = lst[0].keys()
#         with open(saving_name, 'w', newline='') as output_file:
#             dict_writer = csv.DictWriter(output_file, keys)
#             dict_writer.writeheader()
#             dict_writer.writerows(lst)

#     save_in_csv(lst=pred_objs_name_all_scenes, saving_name="./data/pred_objs_for_manual_verification_with_without_matching.csv")