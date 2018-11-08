import spacy
from nltk.corpus import wordnet
from sys import maxsize as maxint

class SpicyFeatures:
    nlp = spacy.load('en')

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

    def get_spacy_max(self, sentence1, sentence2):
        normal_sentence1 = sentence1.strip_metadata()
        normal_sentence2 = sentence2.strip_metadata()
        sentence1_set = set(normal_sentence1)
        sentence2_set = set(normal_sentence2)

        sentence1_unique = sentence1_set.difference(sentence2_set)
        sentence2_unique = sentence2_set.difference(sentence1_set)
        seperator = ' '
        # seperator.join(sentence1_unique)
        spacy_sentence1 = self.nlp(seperator.join(sentence1_unique))
        spacy_sentence2 = self.nlp(seperator.join(sentence2_unique))
        max_similarity = 0

        for sentence1_token in spacy_sentence1:
            for sentence2_token in spacy_sentence2:
                max_similarity = max(sentence1_token.similarity(sentence2_token), max_similarity)

        return max_similarity

    def get_spacy_min(self, sentence1, sentence2):
        normal_sentence1 = sentence1.strip_metadata()
        normal_sentence2 = sentence2.strip_metadata()
        sentence1_set = set(normal_sentence1)
        sentence2_set = set(normal_sentence2)

        sentence1_unique = sentence1_set.difference(sentence2_set)
        sentence2_unique = sentence2_set.difference(sentence1_set)
        seperator = ' '
        # seperator.join(sentence1_unique)
        spacy_sentence1 = self.nlp(seperator.join(sentence1_unique))
        spacy_sentence2 = self.nlp(seperator.join(sentence2_unique))

        min_similarity = maxint

        for sentence1_token in spacy_sentence1:
            for sentence2_token in spacy_sentence2:
                min_similarity = min(sentence1_token.similarity(sentence2_token), min_similarity)

        if min_similarity == maxint:
            return 0
        return min_similarity

    def get_spacy_average(self, sentence1, sentence2):
        normal_sentence1 = sentence1.strip_metadata()
        normal_sentence2 = sentence2.strip_metadata()
        sentence1_set = set(normal_sentence1)
        sentence2_set = set(normal_sentence2)

        sentence1_unique = sentence1_set.difference(sentence2_set)
        sentence2_unique = sentence2_set.difference(sentence1_set)
        seperator = ' '
        # seperator.join(sentence1_unique)
        spacy_sentence1 = self.nlp(seperator.join(sentence1_unique))
        spacy_sentence2 = self.nlp(seperator.join(sentence2_unique))


        avg_similarity = 0
        total_count = 0

        for sentence1_token in spacy_sentence1:
            for sentence2_token in spacy_sentence2:
                avg_similarity += sentence1_token.similarity(sentence2_token)
                total_count += 1

        if total_count == 0:
            return 0
        return float(avg_similarity) / float(total_count)

    def get_spacy_sentence(self, sentence1, sentence2):
        normal_sentence1 = sentence1.strip_metadata()
        normal_sentence2 = sentence2.strip_metadata()
        sentence1_set = set(normal_sentence1)
        sentence2_set = set(normal_sentence2)

        sentence1_unique = sentence1_set.difference(sentence2_set)
        sentence2_unique = sentence2_set.difference(sentence1_set)
        seperator = ' '
        # seperator.join(sentence1_unique)
        spacy_sentence1 = self.nlp(seperator.join(sentence1_unique))
        spacy_sentence2 = self.nlp(seperator.join(sentence2_unique))
        return spacy_sentence1.similarity(spacy_sentence2)

