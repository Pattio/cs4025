import math
import numpy

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
        
    
    def tfidf(self, sentence1, sentence2):
        normal_sentence1 = sentence1.strip_metadata()
        normal_sentence2 = sentence2.strip_metadata()
        vector1 = []
        vector2 = []
        
        word_set = set(normal_sentence1, normal_sentence2)
        for word in word_set:
            tf1 = normal_sentence1.count(word)/len(normal_sentence1)
            tf2 = normal_sentence2.count(word)/len(normal_sentence2)
            number_docs = [word in normal_sentence1, word in normal_sentence2].count(True)
            idf = (math.log10(2/number_docs))
            vector1.append(tf1 * idf)
            vector2.append(tf2 * idf)
        return vector1, vector2
        
        
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
        return float(math.sqrt(count))
        
    