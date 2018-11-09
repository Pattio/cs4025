import nltk
from lemmatizer import Lemamatizer
from classification import Classification
from multiprocessing import Pool

classification = Classification()
lemmatizer = Lemamatizer()
classifier = classification.create_classifier()
pool_size = 4

def proccess_line(line):
    fields = line.rstrip('[\n\r]+').split("\t")
    meta_sentence_1 = lemmatizer.spacy_lemmatize(fields[1])
    meta_sentence_2 = lemmatizer.spacy_lemmatize(fields[2])
    entailment_type = fields[4]
    return (meta_sentence_1, meta_sentence_2, entailment_type)

def create_labeled_sentence(line):
  (sentence1, sentence2, entailment_type) = line
  return (classification.create_features(sentence1, sentence2), entailment_type)

def create_labeled_data(filepath):
    lines = []
    proccessed_lines = []
    labeled_data = []
    with open(filepath) as data:
        next(data) 
        for line in data:
            lines.append(line)
    with Pool(pool_size) as p:
        proccessed_lines = p.map(proccess_line, lines)
    with Pool(pool_size) as p:
        labeled_data = p.map(create_labeled_sentence, proccessed_lines) 
    return labeled_data

print("======== LABELING TRAINING DATA ========")
train_data = create_labeled_data("data/SICK_train.txt")
print("======== LABELING TEST DATA ========")
test_data = create_labeled_data("data/SICK_test_annotated.txt")
print("======== TRAINING CLASSIFIER ========")
classifier.train(train_data)
print("======== TESTING CLASSIFIER ========")
print("Accuracy: " + str(nltk.classify.accuracy(classifier, test_data) * 100.0))
