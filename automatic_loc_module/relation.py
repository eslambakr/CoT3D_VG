import spacy
from scannet_classes import RELATIONS
# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Define the sentence
sentence  = "facing the window the arm chair in the upper right corner.".replace(".","")
#Removing any punctuation 
# Parse the sentence using spaCy
doc = nlp(sentence)
objects = ["window", "arm chair"]

#Match RELATIONS with the sentence

matched_relations = []

#Sort RELATIONS with the longer sentences 

SORTED_REOLATIONS = sorted(RELATIONS, key=len, reverse=True)
#Sentence Mask will be zero with length of sentence
sentence_mask = [0] * len(sentence.split(" "))

for rel in SORTED_REOLATIONS:
    if rel in sentence:
        temp_sentence = sentence_mask
        rel_mask = [0] * len(sentence.split(" "))
        used_before = False
        for word in rel.split(" "):
            indx = sentence.split(" ").index(word)
            if sentence_mask[indx] == 1:
                used_before = True
                break
            temp_sentence[indx] = 1
            rel_mask[indx] = 1     
        
        if not used_before: 
            matched_relations.append([rel, rel_mask])
            sentence_mask = temp_sentence
            



#We have multiple cases: 
#   1. Only two objects in the sentence with one or more relation!
#   2. More than two objects in the sentence with one or more relation!
Objects_relations = []
if len(objects) == 2:
    for rel_list in matched_relations:
        rel = rel_list[0]
        mask= rel_list[1]
        if mask[-1] == 1:
            #at the end or the beginning of the sentence
            Objects_relations.append([objects[-2], rel, objects[-1]])
        elif mask[0] == 1:
            Objects_relations.append([objects[0], rel, objects[1]])

print(Objects_relations)
