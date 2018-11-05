import nltk
from lemmatizer import Lemamatizer
from similarity_features import SimilarityFeatures
from negation_features import NegationFeatures

lemmatizer = Lemamatizer()
similarity_features = SimilarityFeatures()
negation_features = NegationFeatures()
#############################
# Read the data
#############################
with open("data/test_new.txt") as data:
    next(data) 
    for line in data:
        fields = line.rstrip('[\n\r]+').split("\t")
        meta_sentence_1 = lemmatizer.lemmatize(fields[1])
        meta_sentence_2 = lemmatizer.lemmatize(fields[2])
        print(similarity_features.get_jcn_average(meta_sentence_1, meta_sentence_2))