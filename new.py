import nltk
import spacy
from lemmatizer import Lemamatizer
from similarity_features import SimilarityFeatures
from negation_features import NegationFeatures
from spicy_features import SpicyFeatures
from text_similarity_features import TextSimilarityFeatures

lemmatizer = Lemamatizer()
text_similarity_features = TextSimilarityFeatures()
similarity_features = SimilarityFeatures()
#negation_features = NegationFeatures()
spicy_features = SpicyFeatures()
#############################
# Read the data
#############################

nlp = spacy.load('en')
tokens = nlp(u'dog cat banana no')

# for token1 in tokens:
#     for token2 in tokens:
#         print(token1.text, token2.text, token1.similarity(token2))


# print(lemmatizer.spacy_lemmatize("A group of kids is playing in a yard and an old man is standing in the background").data)


# '''
with open("data/test.txt") as data:
    next(data) 
    for line in data:
        fields = line.rstrip('[\n\r]+').split("\t")
        meta_sentence_1 = lemmatizer.lemmatize(fields[1])
        meta_sentence_2 = lemmatizer.lemmatize(fields[2])
        print(spicy_features.get_spacy_average(meta_sentence_1, meta_sentence_2))
# '''
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
            if "isn't" in fields[1] or "isn't" in fields[2]:
                continue
            if "aren't" in fields[1] or "aren't" in fields[2]:
                continue
            meta_sentence_1 = lemmatizer.lemmatize(fields[1])
            meta_sentence_2 = lemmatizer.lemmatize(fields[2])
            print(meta_sentence_1.original_sentence)
            print(meta_sentence_2.original_sentence)
            print(text_similarity_features.sentence_originality(meta_sentence_1, meta_sentence_2))
#            print(meta_sentence_1.strip_metadata(), meta_sentence_2.strip_metadata())
            A = set(meta_sentence_1.strip_metadata())
            B = set(meta_sentence_2.strip_metadata())
#            print(A-B, B-A)
            count += 1
#            print(fields[1])
 #           print(fields[2])
  #          meta_sentence_1 = lemmatizer.lemmatize(fields[1])
   #         meta_sentence_2 = lemmatizer.lemmatize(fields[2])
    #        print(meta_sentence_1.strip_metadata(), meta_sentence_2.strip_metadata())
            print("--------------------------------:)")
    print(count)

'''