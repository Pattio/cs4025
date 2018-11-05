import nltk
from lemmatizer import Lemamatizer
from similarity_features import SimilarityFeatures
from negation_features import NegationFeatures
from spicy_features import SpicyFeatures

lemmatizer = Lemamatizer()
#similarity_features = SimilarityFeatures()
#negation_features = NegationFeatures()
#spicy_features = SpicyFeatures()
#############################
# Read the data
#############################


# print(lemmatizer.spacy_lemmatize("A group of kids is playing in a yard and an old man is standing in the background").data)
'''
with open("data/test.txt") as data:
    next(data) 
    for line in data:
        fields = line.rstrip('[\n\r]+').split("\t")
        meta_sentence_1 = lemmatizer.lemmatize(fields[1])
        meta_sentence_2 = lemmatizer.lemmatize(fields[2])
        print(spicy_features.synonyms(meta_sentence_1, meta_sentence_2))
'''
'''
count = 0
with open("data/test.txt") as data:
    next(data) 
    for line in data:
        fields = line.rstrip('[\n\r]+').split("\t")
        if fields[4] == 'CONTRADICTION':
            if 'not' in fields[1] or 'not' in fields[2] or 'Not' in fields[1] or 'Not' in fields[2]:
                continue
            if 'no' in fields[1] or 'no' in fields[2] or 'No' in fields[1] or 'No' in fields[2]:
                continue
            meta_sentence_1 = lemmatizer.lemmatize(fields[1])
            meta_sentence_2 = lemmatizer.lemmatize(fields[2])
            print(fields[1], fields[2])
#            print(meta_sentence_1.strip_metadata(), meta_sentence_2.strip_metadata())
            A = set(meta_sentence_1.strip_metadata())
            B = set(meta_sentence_2.strip_metadata())
            print(A-B, B-A)
            count += 1
#            print(fields[1])
 #           print(fields[2])
  #          meta_sentence_1 = lemmatizer.lemmatize(fields[1])
   #         meta_sentence_2 = lemmatizer.lemmatize(fields[2])
    #        print(meta_sentence_1.strip_metadata(), meta_sentence_2.strip_metadata())
            print("--------------------------------:)")
    print(count)

'''