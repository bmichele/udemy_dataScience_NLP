#################
# SPAM DETECTOR
#################

import os
import os.path as path

from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np

DATADIR = './data/spambase'

data = pd.read_csv(path.join(DATADIR, 'spambase.data')).values
np.random.shuffle(data) # shuffle the data before splitting

X = data[:, :48]
Y = data[:, -1]

# split train and test
X_train = X[:-100,]
Y_train = Y[:-100,]
X_test = X[-100:,]
Y_test = Y[-100:,]

model = MultinomialNB()
model.fit(X_train, Y_train)
print("Classification rate for Naive Baise: ", model.score(X_test, Y_test))

# same usage for any other sklearn model!
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier()
model.fit(X_train, Y_train)
print("Classification rate for Adam Boost Classifier: ", model.score(X_test, Y_test))