# import spacy
# from spacy.tokens import Span

# def get_subject_phrase(sentence, objects):
#     """
#     Finds the subject phrase in a sentence that is related to a given list of objects.
#     Returns the target object that the subject phrase is related to.

#     Args:
#         sentence (str): The sentence to search in.
#         objects (list of str): The list of objects to search for.

#     Returns:
#         str: The target object that the subject phrase is related to.
#     """
#     # Load the Spacy English model
#     nlp = spacy.load("en_core_web_sm")
#     doc = nlp(sentence)

#     # Create a list of Spacy spans for the objects
#     obj_spans = [Span(doc, obj.start(), obj.end(), label="OBJECT") for obj in objects]

#     # Initialize the Spacy Matcher
#     matcher = spacy.matcher.Matcher(nlp.vocab)

#     # Add the object spans to the Matcher
#     matcher.add("OBJECT", None, *obj_spans)

#     # Define a pattern to match the subject phrase
#     pattern = [{"DEP": {"IN": ["nsubj", "nsubjpass"]}}]

#     # Find the subject phrase that matches the pattern
#     matches = matcher(doc)
#     for match_id, start, end in matches:
#         for token in doc[start:end]:
#             if token.dep_ == "nsubj" or token.dep_ == "nsubjpass":
#                 # Return the target object that the subject phrase is related to
#                 for obj in objects:
#                     if obj in sentence[start:end]:
#                         return obj

#     # If no subject phrase is found, return None
#     return None
# sentence = "The wall lamp is between two posters and behind the table."
# objects = ['posters', 'table', 'wall lamp']

# target_object = get_subject_phrase(sentence, objects)
# print(target_object)
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("I ate pepperoni pizza with my best friend")

objects = ["pepperoni pizza", "best friend"]
object_indices = {}

for obj in objects:
    obj_tokens = nlp(obj)
    for i in range(len(doc) - len(obj_tokens) + 1):
        if all(doc[i + j].text.lower() == obj_tokens[j].text.lower() for j in range(len(obj_tokens))):
            object_indices[obj] = i
            break

print(object_indices)
