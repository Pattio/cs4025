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
from text_similarity_features import TextSimilarityFeatures
from nltk.classify import SklearnClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from spicy_features import SpicyFeatures

lemmatizer = Lemamatizer()
similarity_features = SimilarityFeatures()
text_similarity_features = TextSimilarityFeatures()
negation_features = NegationFeatures()
spicy_features = SpicyFeatures()

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
      'similarity_res_max': similarity_features.get_res_max(sentence1,sentence2),
      'similarity_res_min': similarity_features.get_res_min(sentence1,sentence2),
      'similarity_res_avg': similarity_features.get_res_average(sentence1,sentence2),
      'text_similarity_jaccard': text_similarity_features.jaccard(sentence1,sentence2),
      'text_similarity_dice': text_similarity_features.dice(sentence1,sentence2),
      'text_similarity_overlap1': text_similarity_features.overlap1(sentence1,sentence2),
    #   'text_similarity_overlap2': text_similarity_features.overlap2(sentence1,sentence2),
      'text_similarity_manhattan': text_similarity_features.manhattan(sentence1,sentence2),
    #   'text_similarity_euclidean': text_similarity_features.euclidean(sentence1,sentence2),
    #   'text_similarity_cosine': text_similarity_features.cosine(sentence1,sentence2),
    #   'text_similarity_stat_pearsonr': text_similarity_features.stat_pearsonr(sentence1,sentence2),
      'spicy_synonyms': spicy_features.synonyms(sentence1,sentence2),

      
      
      
      
    }
    

labeled_sentence_pairs = []

with open("SICK_train.txt") as data:
    next(data) 
    for line in data:
        fields = line.rstrip('[\n\r]+').split("\t")
        meta_sentence_1 = lemmatizer.spacy_lemmatize(fields[1])
        meta_sentence_2 = lemmatizer.spacy_lemmatize(fields[2])
        entailment_type = fields[4]
        labeled_sentence_pairs.append((meta_sentence_1,meta_sentence_2,entailment_type))

            
train_set = [(sentence_pair_features(sentence1,sentence2), entailment_type) for (sentence1, sentence2, entailment_type) in labeled_sentence_pairs]
# print(train_set)

# classifier = nltk.NaiveBayesClassifier.train(train_set)
# classifier = SklearnClassifier(SVC()).train(train_set)
classifier = SklearnClassifier(RandomForestClassifier(n_estimators = 30)).train(train_set)
# classifier = nltk.classify.DecisionTreeClassifier.train(train_set)


labeled_test_sentence_pairs = []

with open("SICK_test_annotated.txt") as data:
    next(data) 
    for line in data:
        fields = line.rstrip('[\n\r]+').split("\t")
        meta_sentence_1 = lemmatizer.spacy_lemmatize(fields[1])
        meta_sentence_2 = lemmatizer.spacy_lemmatize(fields[2])
        entailment_type = fields[4]
        labeled_test_sentence_pairs.append((meta_sentence_1,meta_sentence_2,entailment_type))

test_set = [(sentence_pair_features(sentence1,sentence2), entailment_type) for (sentence1, sentence2, entailment_type) in labeled_test_sentence_pairs]
# print(test_set)

print(nltk.classify.accuracy(classifier, test_set))

# print(classifier.show_most_informative_features(5))

# output = []
# for test_case in labeled_test_sentence_pairs:
#     # print('testcase ', test_case)
#     s1 = test_case[0]
#     s2 = test_case[1]
#     entailment_type = test_case[2] 

#     result = classifier.classify(sentence_pair_features(s1,s2))
#     # print(' '.join(s1.strip_metadata()))
#     # print(' '.join(s2.strip_metadata()))
#     # print('result ', result)

#     if result != entailment_type:
#         output.append('\t | '.join([' '.join(s1.strip_metadata()),' '.join(s2.strip_metadata()), entailment_type, result]))

# outputFile = open('missclassified.txt', 'w')
# outputFile.write('\n'.join(output))
# outputFile.close()

        








