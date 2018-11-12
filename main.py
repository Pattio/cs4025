import nltk
import pickle
from sys import argv
from classification import Classification
from lemmatizer import Lemamatizer
from multiprocessing import Pool

classification = Classification("rf",
    {
        'n_estimators': 3000,
    }
)
lemmatizer = Lemamatizer()
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


def confusion_matrix(test_data, classifier):
    correct_entailment_type = []
    predicted_entailment_type = []

    for test_case in test_data:
        correct_entailment_type.append(test_case[1])
        result = classifier.classify(test_case[0])
        predicted_entailment_type.append(result)

    cm = nltk.ConfusionMatrix(correct_entailment_type, predicted_entailment_type)
    return cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9)


if __name__ == '__main__':

    if len(argv) == 2:
        if argv[1] == '--save':
            print("======== LABELING TRAINING DATA ========")
            train_data = create_labeled_data("data/SICK_train.txt")
            print("======== TRAINING CLASSIFIER ========")
            classification.classifier.train(train_data)
            f = open('saved_classifier.pickle', 'wb')
            pickle.dump(classification.classifier, f)
            f.close()
            print("Your classifier is now trained and saved into the following file: 'saved_classifier.pickle' ")

        elif argv[1] == "--load":
            try:
                print("======== LOADING SAVED CLASSIFIER ========")
                f = open('saved_classifier.pickle', 'rb')
                classifier = pickle.load(f)
                f.close()
                print("======== LABELING TEST DATA ========")
                test_data = create_labeled_data("data/SICK_test_annotated.txt")
                print("Accuracy: " + str(nltk.classify.accuracy(classifier, test_data) * 100.0))
            except:
                print("\n'saved_classifier.pickle' is not found")
                print("Run: 'python3 main.py' instead ")
                exit()

        elif argv[1] == "--cm":
            try:
                print("======== LOADING PREPROCESSED TRAIN DATA ========")
                f = open('preprocessed-train-data.pickle', 'rb')
                train_data = pickle.load(f)
                f.close()
                print("======== LOADING PREPROCESSED TEST DATA ========")
                f = open('preprocessed-test-data.pickle', 'rb')
                test_data = pickle.load(f)
                f.close()
                print("======== TRAINING CLASSIFIER ========")
                classification.classifier.train(train_data)
                print("======== PRINTING CONFUSION MATRIX ========")
                print(confusion_matrix(test_data, classification.classifier))
            except:
                print("Saved preprocessed data sets are not found")
                print("Run 'python3 main.py --preprocess' to save the preprocessed data sets!")
                exit()

        elif argv[1] == "--preprocess":
            print("======== LABELING TRAINING DATA ========")
            train_data = create_labeled_data("data/SICK_train.txt")
            f = open('preprocessed-train-data.pickle', 'wb')
            pickle.dump(train_data, f)
            f.close()
            print("Preprocessed training data is saved into 'preprocessed-train-data.pickle'")
            print("======== LABELING TEST DATA ========")
            test_data = create_labeled_data("data/SICK_test_annotated.txt")
            f = open('preprocessed-test-data.pickle', 'wb')
            pickle.dump(test_data, f)
            f.close()
            print("Preprocessed test data is saved into 'preprocessed-test-data.pickle'")

        elif argv[1] == '--fast':
            try:
                print("======== LOADING PREPROCESSED TRAIN DATA ========")
                f = open('preprocessed-train-data.pickle', 'rb')
                train_data = pickle.load(f)
                f.close()
                print("======== LOADING PREPROCESSED TEST DATA ========")
                f = open('preprocessed-test-data.pickle', 'rb')
                test_data = pickle.load(f)
                f.close()
                print("======== TRAINING CLASSIFIER ========")
                classification.classifier.train(train_data)
                print("======== TESTING CLASSIFIER ========")
                print("Accuracy: " + str(nltk.classify.accuracy(classification.classifier, test_data) * 100.0))
            except:
                print("Saved preprocessed data sets are not found")
                print("Run 'python3 main.py --preprocess' to save the preprocessed data sets!")
                exit()

        else:
            print('Invalid command line argument entered')

    elif len(argv) > 2:
        print('\nToo many command line arguments!')
        print('Please enter zero or one argument according to README.md')

    else:
        print("======== LABELING TRAINING DATA ========")
        train_data = create_labeled_data("data/SICK_train.txt")
        print("======== LABELING TEST DATA ========")
        test_data = create_labeled_data("data/SICK_test_annotated.txt")
        print("======== TRAINING CLASSIFIER ========")
        classification.classifier.train(train_data)
        print("======== TESTING CLASSIFIER ========")
        print("Accuracy: " + str(nltk.classify.accuracy(classification.classifier, test_data) * 100.0))
