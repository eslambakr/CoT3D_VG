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
        self.nlp = spacy.load("en_core_web_sm")
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
            temp_utterance = re.sub(r"\b%s\b" % noun, "", temp_utterance)  # remove whole word only
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
            # print(token.text, token.pos_)
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

        return nouns

    def get_nouns_loc_in_txt(self, ip_sentence, nouns):
        """
        Get the locations of the input nouns from the input sentence.
        :return: list of locations. the list length should equal the length of the nouns.
        """
        new_string = ip_sentence.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
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

    def nouns_cls_alignment_exact_matching(self, nouns):
        """
        Align the detected nouns with the predefined classes of the data set.
        :return:
        aligned_nouns: the nouns after alignment.
        """
        obj_nouns = []
        for noun in nouns:
            found = False
            for cls_name in self.cls_names:
                # Exact matching:
                if noun == cls_name:
                    obj_nouns.append(noun)
                    break
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
        obj_nouns_refined = self.nouns_cls_alignment_exact_matching(obj_nouns_refined)
        print("nouns_cls_alignment_exact_matching = ", obj_nouns_refined)
        # Exclude user defined nouns:
        obj_nouns_refined = self.exclude_user_defined_nouns(obj_nouns_refined)
        # Add defined nouns:
        obj_nouns_refined = self.include_user_defined_nouns(utterance=utterance, nouns=obj_nouns_refined)
        # Save the location of each word:
        objs_start_loc, objs_end_loc, pred_objs_mask = self.get_start_and_end_loc_of_objs(ip_sentence=utterance,
                                                                                          nouns=obj_nouns_refined)
        # Add colors to the detected objects:
        colored_utterance = self.add_nouns_loc_to_org_utterance(ip_sentence=utterance,
                                                                pred_objs_mask=pred_objs_mask)

        return obj_nouns_refined, objs_start_loc, objs_end_loc, colored_utterance, utterance

    def get_subject_phrase(self, ip_sentence):
        doc = self.nlp(ip_sentence)
        for token in doc:
            if "subj" in token.dep_:
                subtree = list(token.subtree)
                start = subtree[0].i
                end = subtree[-1].i + 1
                return doc[start:end]

    def get_phrases_between_2_objs(self, ip_sentence, objs_name):
        nouns_loc = self.get_nouns_loc_in_txt(ip_sentence=ip_sentence, nouns=objs_name)
        sub_phrases = []
        sub_phrases_start_obj_loc = []
        sub_phrases_end_obj_loc = []
        for objs_name_idx, obj_loc_idx in enumerate(range(len(nouns_loc) - 1)):
            start = nouns_loc[obj_loc_idx] + 1  # +1 to skip the object name itself
            end = nouns_loc[obj_loc_idx + 1]
            sub_phrase = utterance.split(" ")[start: end]
            sub_phrase = ''.join([s + " " for s in sub_phrase])
            sub_phrases.append(sub_phrase)
            if 'and' in sub_phrase:
                sub_phrases_start_obj_loc.append(0)
            else:
                sub_phrases_start_obj_loc.append(objs_name_idx)
            sub_phrases_end_obj_loc.append(objs_name_idx + 1)

        return sub_phrases, sub_phrases_start_obj_loc, sub_phrases_end_obj_loc

    def check_relationship_conflict(self, word_lists):
        """
        Check the conflict. If we find multiple relation words in the same sub-phrase
        :return:
        conflict_flag: indicates whether there is a conflict or not. based on two rules
                        1) more than relationship type is found.
                        2) more than one relationship word is found in the same type.
        relationship_word: the unique one in case there is no conflict, or the one after solving the conflict.
        """
        empty_counter = 0
        non_empty_lists = []
        for wd_list in word_lists:
            if not wd_list:
                empty_counter += 1  # empty list
            else:
                non_empty_lists.append(wd_list)
        if len(word_lists) - empty_counter == 1:
            # all the lists are empty except one, thus, we are okay.
            if len(non_empty_lists[0]) == 1:
                # means that in this type of relationships only one relation word is found
                conflict_flag = False
                return conflict_flag, non_empty_lists[0][0]
        elif len(word_lists) - empty_counter == 0:  # means no relationship word is detected
            # set the predicted relationship word as None to delete this phrase and candidate objs in the parent function
            return False, None
        else:
            conflict_flag = True
            # Fuse relationships words together:
            pred_relationship_words = sum(non_empty_lists, [])  # fuse the words together
            # Solve the conflict by getting the longest relationship word:
            pred_rel_wd = max(pred_relationship_words, key=len)
            return conflict_flag, pred_rel_wd

    def get_relationship_between_2_objs(self, sub_phrases):
        pred_relationship_word_per_phrase = []
        for i, sub_phrase in enumerate(sub_phrases):
            near_words_pred, far_words_pred, up_words_pred, low_words_pred, bet_words_pred, \
                others_words_pred = [], [], [], [], [], []
            for rel_word in self.near_words:
                # if rel_word in sub_phrase.split(" "):
                if re.search(r'\b%s\b' % (re.escape(rel_word)), sub_phrase) is not None:
                    near_words_pred.append(rel_word)
            for rel_word in self.far_words:
                if re.search(r'\b%s\b' % (re.escape(rel_word)), sub_phrase) is not None:
                    far_words_pred.append(rel_word)
            for rel_word in self.above_words:
                if re.search(r'\b%s\b' % (re.escape(rel_word)), sub_phrase) is not None:
                    up_words_pred.append(rel_word)
            for rel_word in self.below_words:
                if re.search(r'\b%s\b' % (re.escape(rel_word)), sub_phrase) is not None:
                    low_words_pred.append(rel_word)
            for rel_word in self.between_words:
                if re.search(r'\b%s\b' % (re.escape(rel_word)), sub_phrase) is not None:
                    bet_words_pred.append(rel_word)
            for rel_word in self.other_words:
                if re.search(r'\b%s\b' % (re.escape(rel_word)), sub_phrase) is not None:
                    others_words_pred.append(rel_word)

            # Check the conflict. If we find multiple relationship words in the same sub-phrase try to solve it:
            conflict_flag, pred_relationship_word = self.check_relationship_conflict(
                [near_words_pred, far_words_pred, up_words_pred, low_words_pred, bet_words_pred, others_words_pred])
            pred_relationship_word_per_phrase.append(pred_relationship_word)

        return pred_relationship_word_per_phrase

    def analyis_sentence(self, ip_sentence):
        doc = self.nlp(ip_sentence)
        for token in doc:
            print(token.text, token.dep_)
        print("ent: ", doc.ents)
        # displacy.serve(doc, style="dep")


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
    utterance = "Wall lamp between two posters and behind the table"
    print(utterance)
    obj_extractor = ExtractObjsFromDescription("./data/scannet_instance_class_to_semantic_class.json",
                                               coloring_type="colors")
    objs_name, objs_start_loc, objs_end_loc,\
        colored_utterance, adapted_utterance = obj_extractor.extract_objs_from_description(utterance=utterance)
    tgt_from_txt = obj_extractor.get_subject_phrase(ip_sentence=utterance)
    print("colored_utterance = ", colored_utterance)
    print("objs_name = ", objs_name)
    print("tgt_from_txt = ", tgt_from_txt)

    # Get relationships words:
    print("Extract the phrases between two objects:")
    sub_phrases, sub_phrases_start_obj_loc, sub_phrases_end_obj_loc = obj_extractor.get_phrases_between_2_objs(
        ip_sentence=utterance, objs_name=objs_name)
    print("Extract the relationship between two objects:")
    pred_relationship_word_per_phrase = obj_extractor.get_relationship_between_2_objs(sub_phrases)
    for i, sub_phrase in enumerate(sub_phrases):
        print(i, "--> phrase:", sub_phrase, ", start_obj:", objs_name[sub_phrases_start_obj_loc[i]],
              ", end_obj:", objs_name[sub_phrases_end_obj_loc[i]], ", relation:", pred_relationship_word_per_phrase[i])
