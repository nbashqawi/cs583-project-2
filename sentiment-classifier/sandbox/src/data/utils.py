'''
Created on Nov 7, 2017

@author: nbash
'''
import numpy as np
import pandas as pd
import re
import time

import functools

from nltk import TweetTokenizer

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.classification import f1_score, precision_score,\
    recall_score, accuracy_score
from sklearn.metrics.scorer import make_scorer

init_time = time.process_time()

data_file_location = r"..\..\resources\training-Obama-Romney-tweets.xlsx"
tweet_tokenizer = TweetTokenizer()

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

#def bin_data(data, num_bins):

def import_and_filter(dropnan =  None):
    raw_obama_dataframe = pd.read_excel(data_file_location, sheetname='Obama', header=0, parse_cols="D,E")
    raw_obama_dataframe['Class'] = pd.to_numeric(raw_obama_dataframe['Class'], errors='coerce', downcast='float')
    filtered_obama_df = raw_obama_dataframe[raw_obama_dataframe.Class != 2]
    
    raw_romney_dataframe = pd.read_excel(data_file_location, sheetname='Romney', header=0, parse_cols="D,E")
    raw_romney_dataframe['Class'] = pd.to_numeric(raw_romney_dataframe['Class'], errors='coerce', downcast='float')
    filtered_romney_df = raw_romney_dataframe[raw_romney_dataframe.Class != 2]
    
    if dropnan:
        filtered_obama_df = filtered_obama_df.dropna()
        filtered_romney_df = filtered_romney_df.dropna()
    
    return filtered_obama_df, filtered_romney_df
        
def tweet_token_validator(token):
    valid = (not re.search(r"//t\.co.*", token))
    return valid

obama_dataframe, romney_dataframe = import_and_filter(dropnan=True)
print(obama_dataframe['Class'].unique())

obama_train, obama_test = split_train_test(obama_dataframe, 0.10)
obama_train_X = obama_train['Annotated Tweet']
obama_train_y = obama_train['Class']
obama_test_X = obama_test['Annotated Tweet']
obama_test_y = obama_test['Class']

#count_vect = CountVectorizer()
#X_train_counts = count_vect.fit_transform(obama_train_X)
#tfidf_transformer = TfidfTransformer()
#X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#clf = MultinomialNB().fit(X_train_tfidf, obama_train_y)
#X_new_counts = count_vect.transform(obama_test_X)
#X_new_tfidf = tfidf_transformer.transform(X_new_counts)
#predicted = clf.predict(X_new_tfidf)

sentiment_clf = Pipeline([('vect', CountVectorizer(tokenizer=tweet_tokenizer.tokenize, ngram_range=(1,3))),
                          ('tfidf', TfidfTransformer()),
                          ('clf', MultinomialNB()),])

#sentiment_clf.fit(obama_train_X, obama_train_y)
#predicted = sentiment_clf.predict(obama_test_X)

def score_reducer(new_score, accum, divisor):
    return {"precision": new_score["precision"]/divisor + accum["precision"], "recall": new_score["recall"]/divisor + accum["recall"], "fscore": new_score["fscore"]/divisor + accum["fscore"]}

def kFoldValidation(model, X, y, folds):
     shuffled_indices = np.random.permutation(len(X))
     remainder = len(X) % folds
     bin_size = int(len(X) / folds)
     remainder_indices = shuffled_indices[len(X)-remainder:]
     
     pos_scores = []
     neutral_scores = []
     neg_scores = []
     
     for x in range(1, folds+1):
         test_indices = shuffled_indices[(x-1)*bin_size:x*bin_size]
         train_indices = np.concatenate([shuffled_indices[:(x-1)*bin_size], shuffled_indices[x*bin_size:]])
         model.fit(X.iloc[train_indices], y.iloc[train_indices])
         predicted = model.predict(X.iloc[test_indices])
         precision, recall, fscore, support = precision_recall_fscore_support(y.iloc[test_indices], predicted, labels=[1, 0, -1])
         pos_scores.append({"precision": precision[0], "recall": recall[0], "fscore": fscore[0]})
         neutral_scores.append({"precision": precision[1], "recall": recall[1], "fscore": fscore[1]})
         neg_scores.append({"precision": precision[2], "recall": recall[2], "fscore": fscore[2]})
         
     pos = functools.reduce(functools.partial(score_reducer, divisor=folds), pos_scores, {"precision": 0, "recall": 0, "fscore": 0})
     neutral = functools.reduce(functools.partial(score_reducer, divisor=folds), neutral_scores, {"precision": 0, "recall": 0, "fscore": 0})
     neg = functools.reduce(functools.partial(score_reducer, divisor=folds), neg_scores, {"precision": 0, "recall": 0, "fscore": 0})
     
     print(pos)
     print(neutral)
     print(neg)
         
         
kFoldValidation(sentiment_clf, obama_dataframe['Annotated Tweet'], obama_dataframe['Class'], 10)

#for actual, estimate in zip(predicted, obama_test_y):
#    print("Actual: %f, Estimate: %f" % (actual, estimate))

#print("\n")
#print(metrics.classification_report(obama_test_y, predicted))
#print("Accuracy: %f" % np.mean(predicted == obama_test_y))
#print("\n")

total_time = time.process_time() - init_time
print(total_time)
