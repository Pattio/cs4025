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
    