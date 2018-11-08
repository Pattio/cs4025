from nltk.corpus import wordnet
from nltk.corpus import wordnet_ic
from sys import maxint

class SimilarityFeatures:
    brown_ic = wordnet_ic.ic('ic-brown.dat')
    
    def sentence_difference(self, sentence1, sentence2):
        normal_sentence1 = sentence1.strip_metadata()
        normal_sentence2 = sentence2.strip_metadata()
        sentence1_set = set(normal_sentence1)
        sentence2_set = set(normal_sentence2)
        # Find difference A - B
        sentence1_unique = sentence1_set.difference(sentence2_set)
        # Calculate B - A
        sentence2_unique = sentence2_set.difference(sentence1_set)
        return (sentence1_unique, sentence2_unique)
        
    def get_path_max(self, sentence1, sentence2):
        sentence1_unique, sentence2_unique = self.sentence_difference(sentence1, sentence2)
        max_similarity = 0
        # Measure similarity for each unique word from A to each unique word to B
        for sentence1_word in sentence1_unique:
            for sentence2_word in sentence2_unique:
                sentence1_word_tag = sentence1.get_tag(sentence1_word)
                sentence2_word_tag = sentence2.get_tag(sentence2_word)
                synsets_word1 = wordnet.synsets(sentence1_word, sentence1_word_tag)
                synsets_word2 = wordnet.synsets(sentence2_word, sentence2_word_tag)
                
                if len(synsets_word1) == 0:
                    synsets_word1 = wordnet.synsets(sentence1_word)
                if len(synsets_word2) == 0:
                    synsets_word2 = wordnet.synsets(sentence2_word)
                if len(synsets_word1) > 0 and len(synsets_word2) > 0:
                    similarity = wordnet.path_similarity(synsets_word1[0], synsets_word2[0])
                    if similarity != None:
                        max_similarity = max(similarity, max_similarity)
        return max_similarity
        
    def get_path_min(self, sentence1, sentence2):
        sentence1_unique, sentence2_unique = self.sentence_difference(sentence1, sentence2)
        min_similarity = maxint
        # Measure similarity for each unique word from A to each unique word to B
        for sentence1_word in sentence1_unique:
            for sentence2_word in sentence2_unique:
                sentence1_word_tag = sentence1.get_tag(sentence1_word)
                sentence2_word_tag = sentence2.get_tag(sentence2_word)
                synsets_word1 = wordnet.synsets(sentence1_word, sentence1_word_tag)
                synsets_word2 = wordnet.synsets(sentence2_word, sentence2_word_tag)
                
                if len(synsets_word1) == 0:
                    synsets_word1 = wordnet.synsets(sentence1_word)
                if len(synsets_word2) == 0:
                    synsets_word2 = wordnet.synsets(sentence2_word)
                if len(synsets_word1) > 0 and len(synsets_word2) > 0:
                    similarity = wordnet.path_similarity(synsets_word1[0], synsets_word2[0])
                    if similarity != None:
                        min_similarity = min(similarity, min_similarity)
        if min_similarity == maxint:
            return 0
        return min_similarity
        
    def get_path_average(self, sentence1, sentence2):
        sentence1_unique, sentence2_unique = self.sentence_difference(sentence1, sentence2)
        avg_similarity = 0
        total_count = 0
        # Measure similarity for each unique word from A to each unique word to B
        for sentence1_word in sentence1_unique:
            for sentence2_word in sentence2_unique:
                sentence1_word_tag = sentence1.get_tag(sentence1_word)
                sentence2_word_tag = sentence2.get_tag(sentence2_word)
                synsets_word1 = wordnet.synsets(sentence1_word, sentence1_word_tag)
                synsets_word2 = wordnet.synsets(sentence2_word, sentence2_word_tag)
                
                if len(synsets_word1) == 0:
                    synsets_word1 = wordnet.synsets(sentence1_word)
                if len(synsets_word2) == 0:
                    synsets_word2 = wordnet.synsets(sentence2_word)
                if len(synsets_word1) > 0 and len(synsets_word2) > 0:
                    similarity = wordnet.path_similarity(synsets_word1[0], synsets_word2[0])
                    if similarity != None:
                        avg_similarity += similarity
                        total_count += 1
        if total_count == 0:
            return 0
        return float(avg_similarity) / float(total_count)
        
    def get_lch_max(self, sentence1, sentence2):
        sentence1_unique, sentence2_unique = self.sentence_difference(sentence1, sentence2)
        max_similarity = 0
        # Measure similarity for each unique word from A to each unique word to B
        for sentence1_word in sentence1_unique:
            for sentence2_word in sentence2_unique:
                sentence1_word_tag = sentence1.get_tag(sentence1_word)
                sentence2_word_tag = sentence2.get_tag(sentence2_word)
                synsets_word1 = wordnet.synsets(sentence1_word, sentence1_word_tag)
                synsets_word2 = wordnet.synsets(sentence2_word, sentence2_word_tag)
                
                if len(synsets_word1) == 0:
                    synsets_word1 = wordnet.synsets(sentence1_word)
                if len(synsets_word2) == 0:
                    synsets_word2 = wordnet.synsets(sentence2_word)
                
                if len(synsets_word1) > 0 and len(synsets_word2) > 0:
                    # Skip words with different tags
                    if synsets_word1[0].pos() != synsets_word2[0].pos():
                        continue
                    similarity = wordnet.lch_similarity(synsets_word1[0], synsets_word2[0])
                    if similarity != None:
                        max_similarity = max(similarity, max_similarity)
        return max_similarity
        
    def get_lch_min(self, sentence1, sentence2):
        sentence1_unique, sentence2_unique = self.sentence_difference(sentence1, sentence2)
        min_similarity = maxint
        # Measure similarity for each unique word from A to each unique word to B
        for sentence1_word in sentence1_unique:
            for sentence2_word in sentence2_unique:
                sentence1_word_tag = sentence1.get_tag(sentence1_word)
                sentence2_word_tag = sentence2.get_tag(sentence2_word)
                synsets_word1 = wordnet.synsets(sentence1_word, sentence1_word_tag)
                synsets_word2 = wordnet.synsets(sentence2_word, sentence2_word_tag)
                
                if len(synsets_word1) == 0:
                    synsets_word1 = wordnet.synsets(sentence1_word)
                if len(synsets_word2) == 0:
                    synsets_word2 = wordnet.synsets(sentence2_word)
                
                if len(synsets_word1) > 0 and len(synsets_word2) > 0:
                    # Skip words with different tags
                    if synsets_word1[0].pos() != synsets_word2[0].pos():
                        continue
                    similarity = wordnet.lch_similarity(synsets_word1[0], synsets_word2[0])
                    if similarity != None:
                        min_similarity = min(similarity, min_similarity)
        if min_similarity == maxint:
            return 0
        return min_similarity
    
    def get_lch_average(self, sentence1, sentence2):
        sentence1_unique, sentence2_unique = self.sentence_difference(sentence1, sentence2)
        avg_similarity = 0
        total_count = 0
        # Measure similarity for each unique word from A to each unique word to B
        for sentence1_word in sentence1_unique:
            for sentence2_word in sentence2_unique:
                sentence1_word_tag = sentence1.get_tag(sentence1_word)
                sentence2_word_tag = sentence2.get_tag(sentence2_word)
                synsets_word1 = wordnet.synsets(sentence1_word, sentence1_word_tag)
                synsets_word2 = wordnet.synsets(sentence2_word, sentence2_word_tag)
                
                if len(synsets_word1) == 0:
                    synsets_word1 = wordnet.synsets(sentence1_word)
                if len(synsets_word2) == 0:
                    synsets_word2 = wordnet.synsets(sentence2_word)
                if len(synsets_word1) > 0 and len(synsets_word2) > 0:
                    # Skip words with different tags
                    if synsets_word1[0].pos() != synsets_word2[0].pos():
                        continue
                    similarity = wordnet.lch_similarity(synsets_word1[0], synsets_word2[0])
                    if similarity != None:
                        avg_similarity += similarity
                        total_count += 1
        if total_count == 0:
            return 0
        return float(avg_similarity) / float(total_count)
    
    def get_wup_max(self, sentence1, sentence2):
        sentence1_unique, sentence2_unique = self.sentence_difference(sentence1, sentence2)
        max_similarity = 0
        # Measure similarity for each unique word from A to each unique word to B
        for sentence1_word in sentence1_unique:
            for sentence2_word in sentence2_unique:
                sentence1_word_tag = sentence1.get_tag(sentence1_word)
                sentence2_word_tag = sentence2.get_tag(sentence2_word)
                synsets_word1 = wordnet.synsets(sentence1_word, sentence1_word_tag)
                synsets_word2 = wordnet.synsets(sentence2_word, sentence2_word_tag)
                
                if len(synsets_word1) == 0:
                    synsets_word1 = wordnet.synsets(sentence1_word)
                if len(synsets_word2) == 0:
                    synsets_word2 = wordnet.synsets(sentence2_word)
                if len(synsets_word1) > 0 and len(synsets_word2) > 0:
                    similarity = wordnet.wup_similarity(synsets_word1[0], synsets_word2[0])
                    if similarity != None:
                        max_similarity = max(similarity, max_similarity)
        return max_similarity
        
    def get_wup_min(self, sentence1, sentence2):
        sentence1_unique, sentence2_unique = self.sentence_difference(sentence1, sentence2)
        min_similarity = maxint
        # Measure similarity for each unique word from A to each unique word to B
        for sentence1_word in sentence1_unique:
            for sentence2_word in sentence2_unique:
                sentence1_word_tag = sentence1.get_tag(sentence1_word)
                sentence2_word_tag = sentence2.get_tag(sentence2_word)
                synsets_word1 = wordnet.synsets(sentence1_word, sentence1_word_tag)
                synsets_word2 = wordnet.synsets(sentence2_word, sentence2_word_tag)
                
                if len(synsets_word1) == 0:
                    synsets_word1 = wordnet.synsets(sentence1_word)
                if len(synsets_word2) == 0:
                    synsets_word2 = wordnet.synsets(sentence2_word)
                if len(synsets_word1) > 0 and len(synsets_word2) > 0:
                    similarity = wordnet.wup_similarity(synsets_word1[0], synsets_word2[0])
                    if similarity != None:
                        min_similarity = min(similarity, min_similarity)
        if min_similarity == maxint:
            return 0
        return min_similarity
        
    def get_wup_average(self, sentence1, sentence2):
        sentence1_unique, sentence2_unique = self.sentence_difference(sentence1, sentence2)
        avg_similarity = 0
        total_count = 0
        # Measure similarity for each unique word from A to each unique word to B
        for sentence1_word in sentence1_unique:
            for sentence2_word in sentence2_unique:
                sentence1_word_tag = sentence1.get_tag(sentence1_word)
                sentence2_word_tag = sentence2.get_tag(sentence2_word)
                synsets_word1 = wordnet.synsets(sentence1_word, sentence1_word_tag)
                synsets_word2 = wordnet.synsets(sentence2_word, sentence2_word_tag)
                
                if len(synsets_word1) == 0:
                    synsets_word1 = wordnet.synsets(sentence1_word)
                if len(synsets_word2) == 0:
                    synsets_word2 = wordnet.synsets(sentence2_word)
                if len(synsets_word1) > 0 and len(synsets_word2) > 0:
                    similarity = wordnet.wup_similarity(synsets_word1[0], synsets_word2[0])
                    if similarity != None:
                        avg_similarity += similarity
                        total_count += 1
        if total_count == 0:
            return 0
        return float(avg_similarity) / float(total_count)
        
    def get_jcn_max(self, sentence1, sentence2):
        sentence1_unique, sentence2_unique = self.sentence_difference(sentence1, sentence2)
        max_similarity = 0
        # Measure similarity for each unique word from A to each unique word to B
        for sentence1_word in sentence1_unique:
            for sentence2_word in sentence2_unique:
                sentence1_word_tag = sentence1.get_tag(sentence1_word)
                sentence2_word_tag = sentence2.get_tag(sentence2_word)
                synsets_word1 = wordnet.synsets(sentence1_word, sentence1_word_tag)
                synsets_word2 = wordnet.synsets(sentence2_word, sentence2_word_tag)
                
                if len(synsets_word1) == 0:
                    synsets_word1 = wordnet.synsets(sentence1_word)
                if len(synsets_word2) == 0:
                    synsets_word2 = wordnet.synsets(sentence2_word)
                
                if len(synsets_word1) > 0 and len(synsets_word2) > 0:
                    # Skip words with different tags
                    if synsets_word1[0].pos() != synsets_word2[0].pos():
                        continue
                    # Try find similarity from corpus
                    try:
                        similarity = wordnet.jcn_similarity(synsets_word1[0], synsets_word2[0], self.brown_ic)
                    except:
                        continue
                    if similarity != None:
                        max_similarity = max(similarity, max_similarity)
        if max_similarity == 1e+300:
            return 1.0
        return max_similarity
    
    def get_jcn_min(self, sentence1, sentence2):
        sentence1_unique, sentence2_unique = self.sentence_difference(sentence1, sentence2)
        min_similarity = maxint
        # Measure similarity for each unique word from A to each unique word to B
        for sentence1_word in sentence1_unique:
            for sentence2_word in sentence2_unique:
                sentence1_word_tag = sentence1.get_tag(sentence1_word)
                sentence2_word_tag = sentence2.get_tag(sentence2_word)
                synsets_word1 = wordnet.synsets(sentence1_word, sentence1_word_tag)
                synsets_word2 = wordnet.synsets(sentence2_word, sentence2_word_tag)
                
                if len(synsets_word1) == 0:
                    synsets_word1 = wordnet.synsets(sentence1_word)
                if len(synsets_word2) == 0:
                    synsets_word2 = wordnet.synsets(sentence2_word)
                
                if len(synsets_word1) > 0 and len(synsets_word2) > 0:
                    # Skip words with different tags
                    if synsets_word1[0].pos() != synsets_word2[0].pos():
                        continue
                    # Try find similarity from corpus
                    try:
                        similarity = wordnet.jcn_similarity(synsets_word1[0], synsets_word2[0], self.brown_ic)
                    except:
                        continue
                    if similarity != None:
                        min_similarity = min(similarity, min_similarity)
        if min_similarity == maxint or min_similarity == 1e-300:
            return 0
        return min_similarity
    
    def get_jcn_average(self, sentence1, sentence2):
        sentence1_unique, sentence2_unique = self.sentence_difference(sentence1, sentence2)
        avg_similarity = 0
        total_count = 0
        # Measure similarity for each unique word from A to each unique word to B
        for sentence1_word in sentence1_unique:
            for sentence2_word in sentence2_unique:
                sentence1_word_tag = sentence1.get_tag(sentence1_word)
                sentence2_word_tag = sentence2.get_tag(sentence2_word)
                synsets_word1 = wordnet.synsets(sentence1_word, sentence1_word_tag)
                synsets_word2 = wordnet.synsets(sentence2_word, sentence2_word_tag)
                
                if len(synsets_word1) == 0:
                    synsets_word1 = wordnet.synsets(sentence1_word)
                if len(synsets_word2) == 0:
                    synsets_word2 = wordnet.synsets(sentence2_word)
                
                if len(synsets_word1) > 0 and len(synsets_word2) > 0:
                    # Skip words with different tags
                    if synsets_word1[0].pos() != synsets_word2[0].pos():
                        continue
                    # Try find similarity from corpus
                    try:
                        similarity = wordnet.jcn_similarity(synsets_word1[0], synsets_word2[0], self.brown_ic)
                    except:
                        continue
                    if similarity == 1e-300:
                        similarity = 0.0
                    if similarity == 1e+300:
                        similarity = 1.0
                    if similarity != None:
                        avg_similarity += similarity
                        total_count += 1
        if total_count == 0:
            return 0
        return float(avg_similarity) / float(total_count)
        
    def get_res_max(self, sentence1, sentence2):
        sentence1_unique, sentence2_unique = self.sentence_difference(sentence1, sentence2)
        max_similarity = 0
        # Measure similarity for each unique word from A to each unique word to B
        for sentence1_word in sentence1_unique:
            for sentence2_word in sentence2_unique:
                sentence1_word_tag = sentence1.get_tag(sentence1_word)
                sentence2_word_tag = sentence2.get_tag(sentence2_word)
                synsets_word1 = wordnet.synsets(sentence1_word, sentence1_word_tag)
                synsets_word2 = wordnet.synsets(sentence2_word, sentence2_word_tag)
                
                if len(synsets_word1) == 0:
                    synsets_word1 = wordnet.synsets(sentence1_word)
                if len(synsets_word2) == 0:
                    synsets_word2 = wordnet.synsets(sentence2_word)
                
                if len(synsets_word1) > 0 and len(synsets_word2) > 0:
                    # Skip words with different tags
                    if synsets_word1[0].pos() != synsets_word2[0].pos():
                        continue
                    # Try find similarity from corpus
                    try:
                        similarity = wordnet.res_similarity(synsets_word1[0], synsets_word2[0], self.brown_ic)
                    except:
                        continue
                    if similarity != None:
                        max_similarity = max(similarity, max_similarity)
        if max_similarity == 1e+300:
            return 12.0
        return max_similarity
        
    def get_res_min(self, sentence1, sentence2):
        sentence1_unique, sentence2_unique = self.sentence_difference(sentence1, sentence2)
        min_similarity = maxint
        # Measure similarity for each unique word from A to each unique word to B
        for sentence1_word in sentence1_unique:
            for sentence2_word in sentence2_unique:
                sentence1_word_tag = sentence1.get_tag(sentence1_word)
                sentence2_word_tag = sentence2.get_tag(sentence2_word)
                synsets_word1 = wordnet.synsets(sentence1_word, sentence1_word_tag)
                synsets_word2 = wordnet.synsets(sentence2_word, sentence2_word_tag)
                
                if len(synsets_word1) == 0:
                    synsets_word1 = wordnet.synsets(sentence1_word)
                if len(synsets_word2) == 0:
                    synsets_word2 = wordnet.synsets(sentence2_word)
                
                if len(synsets_word1) > 0 and len(synsets_word2) > 0:
                    # Skip words with different tags
                    if synsets_word1[0].pos() != synsets_word2[0].pos():
                        continue
                    # Try find similarity from corpus
                    try:
                        similarity = wordnet.res_similarity(synsets_word1[0], synsets_word2[0], self.brown_ic)
                    except:
                        continue
                    if similarity != None:
                        min_similarity = min(similarity, min_similarity)
        if min_similarity == maxint:
            return 0
        return min_similarity
        

    def get_res_average(self, sentence1, sentence2):
        sentence1_unique, sentence2_unique = self.sentence_difference(sentence1, sentence2)
        avg_similarity = 0
        total_count = 0
        # Measure similarity for each unique word from A to each unique word to B
        for sentence1_word in sentence1_unique:
            for sentence2_word in sentence2_unique:
                sentence1_word_tag = sentence1.get_tag(sentence1_word)
                sentence2_word_tag = sentence2.get_tag(sentence2_word)
                synsets_word1 = wordnet.synsets(sentence1_word, sentence1_word_tag)
                synsets_word2 = wordnet.synsets(sentence2_word, sentence2_word_tag)
                
                if len(synsets_word1) == 0:
                    synsets_word1 = wordnet.synsets(sentence1_word)
                if len(synsets_word2) == 0:
                    synsets_word2 = wordnet.synsets(sentence2_word)
                
                if len(synsets_word1) > 0 and len(synsets_word2) > 0:
                    # Skip words with different tags
                    if synsets_word1[0].pos() != synsets_word2[0].pos():
                        continue
                    # Try find similarity from corpus
                    try:
                        similarity = wordnet.res_similarity(synsets_word1[0], synsets_word2[0], self.brown_ic)
                    except:
                        continue
                    if similarity == 1e+300:
                        similarity = 12.0
                    if similarity != None:
                        avg_similarity += similarity
                        total_count += 1
        if total_count == 0:
            return 0
        return float(avg_similarity) / float(total_count)
        
    def get_lin_max(self, sentence1, sentence2):
        sentence1_unique, sentence2_unique = self.sentence_difference(sentence1, sentence2)
        max_similarity = 0
        # Measure similarity for each unique word from A to each unique word to B
        for sentence1_word in sentence1_unique:
            for sentence2_word in sentence2_unique:
                sentence1_word_tag = sentence1.get_tag(sentence1_word)
                sentence2_word_tag = sentence2.get_tag(sentence2_word)
                synsets_word1 = wordnet.synsets(sentence1_word, sentence1_word_tag)
                synsets_word2 = wordnet.synsets(sentence2_word, sentence2_word_tag)
                
                if len(synsets_word1) == 0:
                    synsets_word1 = wordnet.synsets(sentence1_word)
                if len(synsets_word2) == 0:
                    synsets_word2 = wordnet.synsets(sentence2_word)
                
                if len(synsets_word1) > 0 and len(synsets_word2) > 0:
                    # Skip words with different tags
                    if synsets_word1[0].pos() != synsets_word2[0].pos():
                        continue
                    # Try find similarity from corpus
                    try:
                        similarity = wordnet.lin_similarity(synsets_word1[0], synsets_word2[0], self.brown_ic)
                    except:
                        continue
                    if similarity != None:
                        max_similarity = max(similarity, max_similarity)
        return max_similarity
        
    
    def get_lin_min(self, sentence1, sentence2):
        sentence1_unique, sentence2_unique = self.sentence_difference(sentence1, sentence2)
        min_similarity = maxint
        # Measure similarity for each unique word from A to each unique word to B
        for sentence1_word in sentence1_unique:
            for sentence2_word in sentence2_unique:
                sentence1_word_tag = sentence1.get_tag(sentence1_word)
                sentence2_word_tag = sentence2.get_tag(sentence2_word)
                synsets_word1 = wordnet.synsets(sentence1_word, sentence1_word_tag)
                synsets_word2 = wordnet.synsets(sentence2_word, sentence2_word_tag)
                
                if len(synsets_word1) == 0:
                    synsets_word1 = wordnet.synsets(sentence1_word)
                if len(synsets_word2) == 0:
                    synsets_word2 = wordnet.synsets(sentence2_word)
                
                if len(synsets_word1) > 0 and len(synsets_word2) > 0:
                    # Skip words with different tags
                    if synsets_word1[0].pos() != synsets_word2[0].pos():
                        continue
                    # Try find similarity from corpus
                    try:
                        similarity = wordnet.lin_similarity(synsets_word1[0], synsets_word2[0], self.brown_ic)
                    except:
                        continue
                    if similarity != None:
                        min_similarity = min(similarity, min_similarity)
        if min_similarity == maxint:
            return 0
        return min_similarity
        
    def get_lin_average(self, sentence1, sentence2):
        sentence1_unique, sentence2_unique = self.sentence_difference(sentence1, sentence2)
        avg_similarity = 0
        total_count = 0
        # Measure similarity for each unique word from A to each unique word to B
        for sentence1_word in sentence1_unique:
            for sentence2_word in sentence2_unique:
                sentence1_word_tag = sentence1.get_tag(sentence1_word)
                sentence2_word_tag = sentence2.get_tag(sentence2_word)
                synsets_word1 = wordnet.synsets(sentence1_word, sentence1_word_tag)
                synsets_word2 = wordnet.synsets(sentence2_word, sentence2_word_tag)
                
                if len(synsets_word1) == 0:
                    synsets_word1 = wordnet.synsets(sentence1_word)
                if len(synsets_word2) == 0:
                    synsets_word2 = wordnet.synsets(sentence2_word)
                
                if len(synsets_word1) > 0 and len(synsets_word2) > 0:
                    # Skip words with different tags
                    if synsets_word1[0].pos() != synsets_word2[0].pos():
                        continue
                    # Try find similarity from corpus
                    try:
                        similarity = wordnet.lin_similarity(synsets_word1[0], synsets_word2[0], self.brown_ic)
                    except:
                        continue
                    if similarity != None:
                        avg_similarity += similarity
                        total_count += 1
        if total_count == 0:
            return 0
        return float(avg_similarity) / float(total_count)