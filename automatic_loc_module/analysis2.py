import pandas as pd
import numpy as np

df = pd.read_csv('total_correct.csv')
df.shape


from scannet_classes import SCANNET_OBJECTS
import string
from difflib import SequenceMatcher as SM
#Using apply function to get the class name
annoted_wrong = []
for i in range(1):
    #Get each object
    # for j in range(1,9):
    #     feature = "Answer.object" + str(j)
    #     new_feature = "Answer.object" + str(j) + "_preprocessed"
    #     object_ = str(df.loc[i,feature]).lower()
    #     object_ = object_.translate(str.maketrans('', '', string.punctuation))
    #     if object_ == "nan":
    #         continue
    #     if not(object_ in str(df['Input.utterance'].iloc[i]).lower()):
    #         # print("The annotation of the objects: ", object_, "is not right", "in the sentence: ", str(df['Input.utterance'].iloc[i]))
    #         annoted_wrong.append([df['HITId'].iloc[i], df['Input.utterance'].iloc[i], object_])
    #         continue 
    #     best_len = len(object_.split(" ")[-1])
        
    #     '''
    #     1. Check for Exact Match. If so, we are done 
    #     2. If not, check for the longest match
    #     ''' 
        object_ = "large window"
        best_len = len(object_.split(" ")[-1])
        for obj in ["window", "shelf", "lowest shelf"]:
            isExactMatch = False
            if object_ == 'nan':
                continue
            elif object_ == obj or obj == object_:
                # df.loc[i, new_feature] = obj
                isExactMatch = True
            else:
                #Do substring matching
                annotated_object = object_.split(" ")
                scannet_object = obj.split(" ")
                if len(annotated_object) > len(scannet_object) : 
                    for left in range(len(scannet_object)):
                        import pdb; pdb.set_trace()
                        for right in range(left, len(scannet_object)): 
                            if len(" ".join(scannet_object[left:right+1])) > 0 and " ".join(scannet_object[left:right+1]) in annotated_object and len(obj) > best_len:
                                # df.loc[i, new_feature] = obj
                                best_len = len(obj)
                                import pdb; pdb.set_trace()
                else: #len(annoted_object) <= len(scannet_object)
                    for left in range(len(annotated_object)):
                        for right in range(left, len(annotated_object)): 
                            if len(" ".join(annotated_object[left:right+1])) > 0 and " ".join(annotated_object[left:right +1]) in scannet_object and len(obj) > best_len:
                                # df.loc[i, new_feature] = obj
                                best_len = len(obj)
                                import pdb; pdb.set_trace()

    