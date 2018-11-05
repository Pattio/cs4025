import nltk
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from meta_sentence import MetaSentence
import spacy

class Lemamatizer:
    remove_stop_words = True
    wnl = WordNetLemmatizer()
    allowed_stop_words = ["not", "no", "nor"]
    stop_words = []
    nlp = spacy.load('en')

    def __init__(self):
        all_stop_words = [word for word in set(stopwords.words('english')) if word not in self.allowed_stop_words]
        for word, tag in pos_tag(all_stop_words):
            self.stop_words.append(self.wnl.lemmatize(word.lower(), pos = self.replace_tag(tag)))
    
    #["the", "an", "a"]
    # Lematize the sentence and create tags for each word
    def lemmatize(self, sentence): 
        lematized_sentence_with_metadata = []
        for word, tag in pos_tag(word_tokenize(sentence)):
            lematized_word = self.wnl.lemmatize(word.lower(), pos = self.replace_tag(tag))
            # Remove stop words
            if self.remove_stop_words and lematized_word in self.stop_words:
                continue
            lematized_sentence_with_metadata.append((lematized_word, self.replace_tag(tag)))
        return MetaSentence(lematized_sentence_with_metadata)
    
    def spacy_lemmatize(self, sentence):
        lematized_sentence_with_metadata = []
        word_data = self.nlp(unicode(sentence, encoding="utf-8"))
        for token in word_data:
            # Remove stop words
            if self.remove_stop_words and token.lemma_ in self.stop_words:
                continue
            lematized_sentence_with_metadata.append((token.lemma_, self.replace_spacy_tag(token.pos_)))
        return MetaSentence(lematized_sentence_with_metadata)

    # Simplify WordNet tags
    def replace_tag(self, tag):
        available_tags = {'NN':'n', 'JJ':'a', 'VB':'v', 'RB':'r'}
        try:
            return available_tags[tag[:2]]
        except:
            return 'n' 
    
    # Simplify Spacy tags
    def replace_spacy_tag(self, tag):
        available_tags = {
            'NOUN':'n', 
            'PROPN': 'n',
            'PRON': 'n',
            'ADJ':'a', 
            'VERB':'v', 
            'ADV':'r',
            'AUX':'v'
        }
        try:
            return available_tags[tag]
        except:
            return 'n' 
    
    