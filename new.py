import nltk
from lemmatizer import Lemamatizer
from similarity_features import SimilarityFeatures
from negation_features import NegationFeatures
from spicy_features import SpicyFeatures

lemmatizer = Lemamatizer()
similarity_features = SimilarityFeatures()
negation_features = NegationFeatures()
spicy_features = SpicyFeatures()
#############################
# Read the data
#############################


with open("data/test.txt") as data:
    next(data) 
    for line in data:
        fields = line.rstrip('[\n\r]+').split("\t")
        meta_sentence_1 = lemmatizer.lemmatize(fields[1])
        meta_sentence_2 = lemmatizer.lemmatize(fields[2])
        print(spicy_features.synonyms(meta_sentence_1, meta_sentence_2))
'''

with open("data/test.txt") as data:
    next(data) 
    for line in data:
        fields = line.rstrip('[\n\r]+').split("\t")
        if lemmatizer.lemmatize(fields[4]) == 'CONTRADICTION':
            meta_sentence_1 = lemmatizer.lemmatize(fields[1])
            meta_sentence_2 = lemmatizer.lemmatize(fields[2])
            print(meta_sentence_1, meta_sentence_2)   '''