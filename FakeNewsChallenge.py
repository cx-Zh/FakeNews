import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from scorer import score_submission, print_confusion_matrix, score_defaults, SCORE_REPORT
from scipy.spatial.distance import cosine
from tqdm import tqdm

'''
    File name: FakeNewsChallenge.py
    Author: WhyK
    Date created: 4/30/2018
    Date last modified: 7/15/2018
    Python Version: 3.5
'''


# combine headlines and bodies together
# return:
#       full_data = [headline, id, stance, body]
def combine_hl_n_b(data1, data2):
    full_data = pd.merge(data1, data2,  how='inner', left_on=['Body ID'], right_on=['Body ID'])
    return full_data    # full_data = [headline, id, stance, body]


# sort the matrix to an appropriate order
# return:
#       train_data = [headline, body, stance]
def sort_matrix(train_matrix):
    train_data = np.delete(train_matrix, 1, axis=1)
    sort_order = np.argsort([0, 2, 1])
    train_data = train_data[:, sort_order]
    return train_data


# calculate the cosine_similarity of headline and body
def cosine_similarity(doc):
    tf_idf_vec = TfidfVectorizer(stop_words='english')
    tf_idf = tf_idf_vec.fit_transform([doc[0], doc[1]])
    vec_headline = np.squeeze(np.array(tf_idf[0].todense()))
    vec_body = np.squeeze(np.array(tf_idf[1].todense()))
    vec = np.array([0.0])
    vec[0] = 1.0 - cosine(vec_headline, vec_body)
    return vec


################################################### train part ##########################################################

# load training data
print('loading train dataset...')
train_hl = pd.read_csv('data/train_stances.csv')
train_b = pd.read_csv('data/train_bodies.csv')
print('Successfully loading train dataset!\n')
print('Initialise x and y for train dataset...')

# preprocessing data
data = combine_hl_n_b(train_hl, train_b)
train_data = sort_matrix(np.array(data))

# Initialise x and y for train dataset
x_train = np.array([cosine_similarity(doc) for doc in tqdm(train_data)])
print('Successfully Initialising x and y for train dataset!\n')
le = LabelEncoder()
y_train = le.fit_transform(list(train_data[:, 2]))

# linear regression grid search
# param_grid = {'fit_intercept': [True, False], 'normalize': [True, False], 'copy_X': [True, False]}
# model = LinearRegression()
# Best parameter: {'copy_X': True, 'fit_intercept': True, 'n_jobs': 1, 'normalize': False}


# Logistic Regression grid search

# param_grid = {'penalty': ['l1', 'l2'],
#               'C': [0.1, 0.5, 0.8, 1, 2, 3], 'solver': ['saga'],
#               'multi_class': ['ovr', 'multinomial']}
# model = LogisticRegression()
# Best parameter{solver='saga', multi_class='multinomial',}

# LinearSVC
# param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]}
# model = LinearSVC()
# Best parameter {'C': 1000, 'class_weight': None, 'dual': True, 'fit_intercept': True, 'intercept_scaling': 1, 'loss':
# 'squared_hinge','max_iter': 1000, 'multi_class': 'ovr', 'penalty': 'l2', 'random_state': None, 'tol': 0.0001,
# 'verbose': 0}

# DecisionTreeClassifier
# model = DecisionTreeClassifier()
#
print('Using grid searching to find best parameters...')
param_grid = {'penalty': ['l1', 'l2'],
              'C': [0.1, 0.5, 0.8, 1, 2, 3]}
model = LogisticRegression(solver='saga', multi_class='multinomial', max_iter=300)

# use gridseach to find the best parameters

grid_search = GridSearchCV(model, param_grid)
grid_search.fit(x_train, y_train)
best_parameters = grid_search.best_estimator_.get_params()
print('Best parameters have been found as follows')
print(best_parameters)
print('')

# put the train vectors into the model
clf = LogisticRegression(solver='saga', multi_class='multinomial', max_iter=300, C=best_parameters['C'],
                             penalty=best_parameters['penalty'])
clf.fit(x_train, y_train)


################################################### test part ##########################################################
print('Initialise x for test dataset...')
# load training data
test_hl = pd.read_csv('data/competition_test_stances.csv')
test_b = pd.read_csv('data/competition_test_bodies.csv')



# preprocessing training data
data = combine_hl_n_b(test_hl, test_b)
test_data = sort_matrix(np.array(data))


x_test = np.array([cosine_similarity(doc) for doc in tqdm(test_data)])
print('Successfully Initialising x for test dataset!\n')

# predict y for test set
y = clf.predict(x_test)
y_test = []
for i in range(len(y)):
    y_test.append(int(round(y[i])))

stance = ['agree', 'disagree', 'discuss', 'unrelated']
pred_test = np.array([stance[i] for i in y_test])

# Prepare test dataset for testing accuracy
print('Prepare test dataset for testing accuracy...')
test_labels = np.array([{'Headline': test_data[i][0], 'Body ID': test_data[i][1], 'Stance': pred_test[i]}
                       for i in tqdm(range(len(test_data)))])
gold_labels = np.array([{'Headline': test_data[i][0], 'Body ID': test_data[i][1], 'Stance': test_data[i][2]}
                       for i in range(len(test_data))])
print('Successfully generate the required dataset!\n')

################################################### output part ########################################################

# output the result
test_score, cm = score_submission(gold_labels, test_labels)
null_score, max_score = score_defaults(gold_labels)
print_confusion_matrix(cm)
print(SCORE_REPORT.format(max_score, null_score, test_score))


