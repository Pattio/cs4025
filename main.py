import nltk
import itertools
import math
# from nltk import pos_tag
# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import sent_tokenize, word_tokenize
# from nltk.corpus import wordnet
from lemmatizer import Lemamatizer
from similarity_features import SimilarityFeatures
from negation_features import NegationFeatures
from nltk.classify import SklearnClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

lemmatizer = Lemamatizer()
similarity_features = SimilarityFeatures()
negation_features = NegationFeatures()

def sentence_pair_features(sentence1, sentence2):
    return {
      'negation': negation_features.simple_negation(sentence1, sentence2),
      'similarity_path_max': similarity_features.get_path_max(sentence1, sentence2),
      'similarity_path_min': similarity_features.get_path_min(sentence1, sentence2),
      'similarity_path_avg': similarity_features.get_path_average(sentence1,sentence2),
      # 'similarity_lch_max': similarity_features.get_lch_max(sentence1,sentence2),
      # 'similarity_lch_min': similarity_features.get_lch_min(sentence1,sentence2),
      # 'similarity_lch_avg': similarity_features.get_lch_average(sentence1,sentence2),
      'similarity_wup_max': similarity_features.get_wup_max(sentence1,sentence2),
      'similarity_wup_min': similarity_features.get_wup_min(sentence1,sentence2),
      'similarity_wup_avg': similarity_features.get_wup_average(sentence1,sentence2),
      'similarity_jcn_max': similarity_features.get_jcn_max(sentence1,sentence2),
      'similarity_jcn_min': similarity_features.get_jcn_min(sentence1,sentence2),
      'similarity_jcn_avg': similarity_features.get_jcn_average(sentence1,sentence2),
      'antonyms': negation_features.antonyms(sentence1, sentence2),
    }
    

labeled_sentence_pairs = []

with open("SICK_train.txt") as data:
    next(data) 
    for line in data:
        fields = line.rstrip('[\n\r]+').split("\t")
        meta_sentence_1 = lemmatizer.lemmatize(fields[1])
        meta_sentence_2 = lemmatizer.lemmatize(fields[2])
        entailment_type = fields[4]
        labeled_sentence_pairs.append((meta_sentence_1,meta_sentence_2,entailment_type))

            
train_set = [(sentence_pair_features(sentence1,sentence2), entailment_type) for (sentence1, sentence2, entailment_type) in labeled_sentence_pairs]
# print(train_set)

# classifier = nltk.NaiveBayesClassifier.train(train_set)
# classifier = SklearnClassifier(SVC()).train(train_set)
# classifier = SklearnClassifier(RandomForestClassifier(n_estimators = 30)).train(train_set)
classifier = nltk.classify.DecisionTreeClassifier.train(train_set)


labeled_test_sentence_pairs = []

with open("SICK_test_annotated.txt") as data:
    next(data) 
    for line in data:
        fields = line.rstrip('[\n\r]+').split("\t")
        meta_sentence_1 = lemmatizer.lemmatize(fields[1])
        meta_sentence_2 = lemmatizer.lemmatize(fields[2])
        entailment_type = fields[4]
        labeled_test_sentence_pairs.append((meta_sentence_1,meta_sentence_2,entailment_type))

test_set = [(sentence_pair_features(sentence1,sentence2), entailment_type) for (sentence1, sentence2, entailment_type) in labeled_test_sentence_pairs]
# print(test_set)

print(nltk.classify.accuracy(classifier, test_set))

# print(classifier.show_most_informative_features(5))






