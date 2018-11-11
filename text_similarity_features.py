import math, numpy
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import laplacian_kernel
from sklearn.metrics.pairwise import pairwise_distances_chunked

class TextSimilarityFeatures:
    
    def jaccard(self, sentence1, sentence2):
        normal_sentence1 = sentence1.strip_metadata()
        normal_sentence2 = sentence2.strip_metadata()
        A = set(normal_sentence1)
        B = set(normal_sentence2)
        return float(len(A.intersection(B)))/float(len(A.union(B))) 
        
    
    def dice(self, sentence1, sentence2):
        normal_sentence1 = sentence1.strip_metadata()
        normal_sentence2 = sentence2.strip_metadata()
        A = set(normal_sentence1)
        B = set(normal_sentence2)
        return 2 * float(len(A.intersection(B)))/(len(A)+len(B))
        
    
    def overlap1(self, sentence1, sentence2):
        normal_sentence1 = sentence1.strip_metadata()
        normal_sentence2 = sentence2.strip_metadata()
        A = set(normal_sentence1)
        B = set(normal_sentence2)
        return len(A.intersection(B))/len(A)
        
    
    def overlap2(self, sentence1, sentence2):
        normal_sentence1 = sentence1.strip_metadata()
        normal_sentence2 = sentence2.strip_metadata()
        A = set(normal_sentence1)
        B = set(normal_sentence2)
        return len(A.intersection(B))/len(B)
        
    
    vectorizer = TfidfVectorizer()

    def tfdif_advanced(self, sentence1, sentence2):
        lemmatized_sentence1 = sentence1.strip_metadata()
        lemmatized_sentence2 = sentence2.strip_metadata()
        separator = " "
        sentences = [
            separator.join(lemmatized_sentence1),
            separator.join(lemmatized_sentence2),
        ]
        return self.vectorizer.fit_transform(sentences)
    
    def cosine_advanced(self, sentence1, sentence2):
        tfidf = self.tfdif_advanced(sentence1, sentence2)
        cosine_similarities = linear_kernel(tfidf[0:1], tfidf).flatten()
        return cosine_similarities[0]

    def cosine_advanced_spicy(self, sentence1, sentence2):
        sentences = [
            sentence1.original_sentence,
            sentence2.original_sentence,
        ]
        tfidf = self.vectorizer.fit_transform(sentences)
        cosine_similarities = linear_kernel(tfidf[0:1], tfidf).flatten()
        return cosine_similarities[0]

    def manhattan_advanced(self, sentence1, sentence2):
        tfidf = self.tfdif_advanced(sentence1, sentence2)
        return manhattan_distances(tfidf)[0][1]

    def cosine_distance_advanced(self, sentence1, sentence2):
        tfidf = self.tfdif_advanced(sentence1, sentence2)
        return cosine_distances(tfidf)[0][1]

    def euclidean_advanced(self, sentence1, sentence2):
        tfidf = self.tfdif_advanced(sentence1, sentence2)
        return euclidean_distances(tfidf)[0][1]

    def laplacian_kernel_advanced(self, sentence1, sentence2):
        tfidf = self.tfdif_advanced(sentence1, sentence2)
        return laplacian_kernel(tfidf)[0][1]

    def pairwise_distances_chunked_advanced(self, sentence1, sentence2):
        tfidf = self.tfdif_advanced(sentence1, sentence2)
        return next(pairwise_distances_chunked(tfidf))[0][1]
    
    def tfidf(self, lemmatized_sentence1, lemmatized_sentence2):
        lemmatized_sentence1 = lemmatized_sentence1.strip_metadata()
        lemmatized_sentence2 = lemmatized_sentence2.strip_metadata()
        vector1 = []
        vector2 = []
        
        word_set = set(lemmatized_sentence1 + lemmatized_sentence2)
        for word in word_set: 
            tf1 = float(lemmatized_sentence1.count(word))/float(len(lemmatized_sentence1))
            tf2 = float(lemmatized_sentence2.count(word))/float(len(lemmatized_sentence2))

            num_of_sent_containing_the_word = [word in lemmatized_sentence1, word in lemmatized_sentence2].count(True)
            idf = float(math.log10(1+(float(2)/float(num_of_sent_containing_the_word))))
            vector1.append(float(tf1) * float(idf))
            vector2.append(float(tf2) * float(idf))
        
        return (vector1, vector2)
        
        
    def cosine(self, sentence1, sentence2):
        x, y = self.tfidf(sentence1, sentence2)
        return numpy.dot(x,y)/(math.sqrt(numpy.dot(x, x))*math.sqrt(numpy.dot(y, y)))
        
    
    def manhattan(self, sentence1, sentence2):
        x, y = self.tfidf(sentence1, sentence2)
        count = 0
        for i in range(len(x)):
            count += abs(x[i] - y[i])
        return count
        
    
    def euclidean(self, sentence1, sentence2):
        x, y = self.tfidf(sentence1, sentence2)
        count = 0
        for i in range(len(x)):
            count += (x[i] - y[i])**2
        return math.sqrt(count)

    def stat_pearsonr(self, sentence1, sentence2):
        x, y = self.tfidf(sentence1, sentence2)
        if x == y:
            return 1
        return pearsonr(x, y)[0]

    def stat_spearmanr(self, sentence1, sentence2):
        x, y = self.tfidf(sentence1, sentence2)
        if x == y:
            return 1
        return spearmanr(x, y)[0]

    def stat_kendalltau(self, sentence1, sentence2):
        x, y = self.tfidf(sentence1, sentence2)
        result = kendalltau(x, y)[0]
        if x == y:
            result = 1
        if math.isnan(result):
            print('vector1: ', x, 'vector2: ', y)
            print('pair: ', sentence1.strip_metadata(), sentence2.strip_metadata())
            print('result: ', result)
        # print(result)
        return result
        
        
    def sentence_originality(self, sentence1, sentence2):
        normal_sentence1 = sentence1.original_sentence.split(" ")
        normal_sentence2 = sentence2.original_sentence.split(" ")
        if len(normal_sentence1) == len(normal_sentence2):
            count = len(normal_sentence1)
            for i in range(len(normal_sentence1)):
                if normal_sentence1[i] == normal_sentence2[i]:
                    count -= 1
            return float(count)/len(normal_sentence1)
        else:
            return -1