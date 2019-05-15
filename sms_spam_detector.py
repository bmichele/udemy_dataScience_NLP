#####################
# SMS Spam Detector #
#####################

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
import matplotlib.pyplot as plt
##
data = pd.read_csv('./data/spam.csv', encoding='latin-1')
data = data.iloc[:, [0, 1]]
data.columns = ['label', 'text']
data['label'].map({'ham': 0, 'spam': 1})

# Define different set of features
vectorizers = {
    'tfidf': TfidfVectorizer(),
    'count': CountVectorizer()
}

models = {
    'adaBoost': AdaBoostClassifier(),
    'naiveBayes': MultinomialNB()
}

y = data['label'].values # labels does not depend on features definition

# Compute score using all combinations of vectorizers and models
runs = 1
for name, obj in vectorizers.items():
    print()
    print('vectorizer: ', name)
    obj.fit(data['text'])
    X = obj.transform(data['text'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

    for model_name, model in models.items():
        print('model: ', model_name)
        eval_train = 0
        eval_test = 0
        for run in range(runs):
            model.fit(X_train, y_train)
            eval_test += model.score(X_test, y_test)
            eval_train += model.score(X_train, y_train)
        print('train score: ', eval_train/runs)
        print('test score:  ', eval_test/runs)

# Best combination: vectorizer:  count, model:  naiveBayes, score:  0.9809160305343508
## Look into the predictions with wordclouds
data['prediction'] = model.predict(X)
false_neg = data.loc[(data['prediction'] == 'ham') & (data['label'] == 'spam')]['text'].values
false_pos = data.loc[(data['prediction'] == 'spam') & (data['label'] == 'ham')]['text'].values
true_neg = data.loc[(data['prediction'] == 'ham') & (data['label'] == 'ham')]['text'].values
true_pos = data.loc[(data['prediction'] == 'spam') & (data['label'] == 'spam')]['text'].values

wordcloudFN = WordCloud().generate(' '.join(false_neg))
plt.imshow(wordcloudFN, interpolation='bilinear')
##
wordcloudFP = WordCloud().generate(' '.join(false_pos))
plt.imshow(wordcloudFP, interpolation='bilinear')
##
wordcloudTN = WordCloud().generate(' '.join(true_neg))
plt.imshow(wordcloudTN, interpolation='bilinear')
##
wordcloudTP = WordCloud().generate(' '.join(true_pos))
plt.imshow(wordcloudTP, interpolation='bilinear')