import nltk

from nltk.classify import SklearnClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

from similarity_features import SimilarityFeatures
from negation_features import NegationFeatures
from text_similarity_features import TextSimilarityFeatures
from spicy_features import SpicyFeatures


class Classification:

    similarity_features = SimilarityFeatures()
    text_similarity_features = TextSimilarityFeatures()
    negation_features = NegationFeatures()
    spicy_features = SpicyFeatures()
    classifier = nltk.NaiveBayesClassifier

    def __init__(self, classifier_name, params={}):
        if classifier_name == 'bayes':
            self.classifier = nltk.NaiveBayesClassifier
        elif classifier_name == 'decision_tree':
            self.classifier = nltk.classify.DecisionTreeClassifier
        elif classifier_name == 'svc':
            self.classifier = SklearnClassifier(SVC(**params))
        elif classifier_name == 'gradient_boost':
            self.classifier = SklearnClassifier(GradientBoostingClassifier(**params))
        elif classifier_name == 'k-neighbors':
            self.classifier = SklearnClassifier(KNeighborsClassifier(**params))
        elif classifier_name == 'sgdc':
            self.classifier = SklearnClassifier(SGDClassifier(**params))
        elif classifier_name == 'mlp':
            self.classifier = SklearnClassifier(MLPClassifier(**params))
        elif classifier_name == 'rf':
            self.classifier = SklearnClassifier(RandomForestClassifier(**params))

    def create_features(self, sentence1, sentence2):
        return {
            'negation': self.negation_features.simple_negation(sentence1, sentence2),
            'similarity_path_max': self.similarity_features.get_path_max(sentence1, sentence2),
            'similarity_path_min': self.similarity_features.get_path_min(sentence1, sentence2),
            'similarity_path_avg': self.similarity_features.get_path_average(sentence1, sentence2),
            # 'similarity_lch_max': self.similarity_features.get_lch_max(sentence1,sentence2),
            # 'similarity_lch_min': self.similarity_features.get_lch_min(sentence1,sentence2),
            # 'similarity_lch_avg': self.similarity_features.get_lch_average(sentence1,sentence2),
            'similarity_wup_max': self.similarity_features.get_wup_max(sentence1, sentence2),
            'similarity_wup_min': self.similarity_features.get_wup_min(sentence1, sentence2),
            'similarity_wup_avg': self.similarity_features.get_wup_average(sentence1, sentence2),
            'similarity_jcn_max': self.similarity_features.get_jcn_max(sentence1, sentence2),
            'similarity_jcn_min': self.similarity_features.get_jcn_min(sentence1, sentence2),
            'similarity_jcn_avg': self.similarity_features.get_jcn_average(sentence1, sentence2),
            'antonyms': self.negation_features.antonyms(sentence1, sentence2),
            'similarity_res_max': self.similarity_features.get_res_max(sentence1, sentence2),
            'similarity_res_min': self.similarity_features.get_res_min(sentence1, sentence2),
            'similarity_res_avg': self.similarity_features.get_res_average(sentence1, sentence2),
            'text_similarity_jaccard': self.text_similarity_features.jaccard(sentence1, sentence2),
            'text_similarity_dice': self.text_similarity_features.dice(sentence1, sentence2),
            'text_similarity_overlap1': self.text_similarity_features.overlap1(sentence1, sentence2),
            # 'text_similarity_overlap2': self.text_similarity_features.overlap2(sentence1,sentence2),
            'text_similarity_manhattan': self.text_similarity_features.manhattan(sentence1, sentence2),
            # 'text_similarity_cosine_distance': self.text_similarity_features.cosine_distance_advanced(sentence1, sentence2),
            # 'text_similarity_euclidean_advanced': self.text_similarity_features.euclidean_advanced(sentence1,sentence2),
            # 'text_similarity_laplacian_advanced': self.text_similarity_features.laplacian_kernel_advanced(sentence1,sentence2),
            # 'text_similarity_chunked_advanced': self.text_similarity_features.pairwise_distances_chunked_advanced(sentence1,sentence2),
            # 'text_similarity_euclidean': self.text_similarity_features.euclidean(sentence1,sentence2),
            'text_similarity_cosine': self.text_similarity_features.cosine_advanced(sentence1, sentence2),
            # 'text_similarity_cosine_spicy': self.text_similarity_features.cosine_advanced_spicy(sentence1,sentence2),
            # 'text_similarity_stat_pearsonr': self.text_similarity_features.stat_pearsonr(sentence1,sentence2),
            # 'text_similarity_stat_spearmanr': self.text_similarity_features.stat_spearmanr(sentence1,sentence2),
            # 'text_similarity_stat_kendalltau': self.text_similarity_features.stat_kendalltau(sentence1,sentence2),
            'spicy_synonyms': self.spicy_features.synonyms(sentence1, sentence2),
            # 'text-similarity-sentence_originality': self.text_similarity_features.sentence_originality(sentence1, sentence2)
            # 'similarity_lin_max': self.similarity_features.get_lin_max(sentence1,sentence2),
            # 'similarity_lin_min': self.similarity_features.get_lin_min(sentence1,sentence2),
            # 'similarity_lin_avg': self.similarity_features.get_lin_average(sentence1,sentence2),
            'spicy_spacy_max': self.spicy_features.get_spacy_max(sentence1, sentence2),
            'spicy_spacy_min': self.spicy_features.get_spacy_min(sentence1, sentence2),
            'spicy_spacy_avg': self.spicy_features.get_spacy_average(sentence1, sentence2),
            'spicy_spacy_sentence': self.spicy_features.get_spacy_sentence(sentence1, sentence2),
        }
