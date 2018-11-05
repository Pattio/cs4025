from nltk.corpus import wordnet

class SpicyFeatures:
    
    def synonyms(self, sentence1, sentence2):
        normal_sentence1 = sentence1.strip_metadata()
        normal_sentence2 = sentence2.strip_metadata()
        sentence1_set = set(normal_sentence1)
        sentence2_set = set(normal_sentence2)
        
        # Find difference A - B
        sentence1_unique = sentence1_set.difference(sentence2_set)
        # Calculate B - A
        sentence2_unique = sentence2_set.difference(sentence1_set)
        
        synonym_count = 0
        for sentence1_word in sentence1_unique:
            synonyms = self.get_synonyms(sentence1.get_data(sentence1_word))
            for synonym in synonyms:
                synonym_count += normal_sentence2.count(synonym)
                
        for sentence2_word in sentence2_unique:
            synonyms = self.get_synonyms(sentence2.get_data(sentence2_word))
            for synonym in synonyms:
                synonym_count += normal_sentence1.count(synonym)
        return synonym_count
        
        
    def get_synonyms(self, word_tuple):
        synonyms = []
        for syn in wordnet.synsets(word_tuple[0]): 
            for l in syn.lemmas():
                if (word_tuple[1] == 'v' and l._synset.pos() == 'v') or (word_tuple[1] != 'v' and l._synset.pos() != 'v'):
                    synonyms.append(l.name())
        return set(synonyms)