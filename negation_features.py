class NegationFeatures:
    negation_list = ["no", "not", "nor"]
    
    def simple_negation(self, sentence1, sentence2):
        normal_sentence1 = sentence1.strip_metadata()
        normal_sentence2 = sentence2.strip_metadata()
        count = 0
        for negation in self.negation_list:
            count += normal_sentence1.count(negation) 
            count -= normal_sentence2.count(negation)
        return count != 0
    
    
    def antonyms(self, sentence1, sentenc2):
        normal_sentence1 = sentence1.strip_metadata()
        normal_sentence2 = sentence2.strip_metadata()
        
        
    
    
    def get_antonyms(word):
        antonyms = []
        for synonym in wordnet.synsets(word): 
            for synonym_lemma in synonym.lemmas(): 
                if synonym_lemma.antonyms(): 
                    antonyms.append(synonym_lemma.antonyms()[0].name())
        return set(antonyms)
        
    