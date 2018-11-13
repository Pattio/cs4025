import pickle
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV


class ParameterSearch:
    def load_data(self, filename):
        f = open(filename, 'rb')
        data = pickle.load(f)
        f.close()
        return data

    def transform_data(self, data):
        features = []
        labels = []
        for data_item in data:
            labels.append(data_item[1])
            train_features = []
            for key, value in data_item[0].items():
                train_features.append(value)
            features.append(train_features)

        data_X = np.array(features)
        data_y = np.matrix([labels]).A1
        return (data_X, data_y)

    def search_parameters(self, classifier, parameter_space):
        train_data = self.load_data("preprocessed-train-data.pickle")
        test_data = self.load_data("preprocessed-test-data.pickle")
        train_X, train_y = self.transform_data(train_data)
        test_X, test_y = self.transform_data(test_data)

        best_score = 0
        for g in ParameterGrid(parameter_space):
            classifier.set_params(**g)
            classifier.fit(train_X, train_y)

            score = classifier.score(test_X, test_y)
            if score > best_score:
                print("Current best" + str(score))
                print("Grid:", g)
                print("----------------------")
                best_score = score
                best_grid = g
        print("--------------------------")
        print("Best Score: %0.6f" % best_score)
        print("Grid:", best_grid)

    def search_parameters_cv(self, classifier, parameter_space):
        train_data = self.load_data("preprocessed-train-data.pickle")
        test_data = self.load_data("preprocessed-test-data.pickle")
        train_X, train_y = self.transform_data(train_data)
        test_X, test_y = self.transform_data(test_data)

        grid_search = GridSearchCV(classifier, parameter_space, n_jobs=-1, cv=3)
        grid_search.fit(train_X, train_y)
        means = grid_search.cv_results_['mean_test_score']
        stds = grid_search.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
            print("%0.4f (+/-%0.04f) for %r" % (mean, std * 2, params))

        print("---------------------------------")
        print('Best parameters found:\n', grid_search.best_params_)
        print(grid_search.best_score_)


pm = ParameterSearch()
pm.search_parameters(RandomForestClassifier(n_jobs=-1), {
    'n_estimators': [30],
    'criterion': ['gini', 'entropy'],
    'max_depth': [25, 50, 75, 100, None],
    'min_samples_split': [2, 5, 10, 25],
    'min_samples_leaf': [1, 10, 25],
    'max_features': ['auto', 'sqrt', 'log2', None],
    'class_weight': ['balanced', 'balanced_subsample', None]
})
