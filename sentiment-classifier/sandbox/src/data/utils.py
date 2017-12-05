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
from nltk.corpus import stopwords

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support

data_file_location = r"..\..\resources\training-Obama-Romney-tweets.xlsx"
tweet_tokenizer = TweetTokenizer()
special_stopwords = ['omg', 'lol', 'umm', 'hmm', 'ah', 'oh', 'yea']
special_stopwords.extend(stopwords.words('english'))
eng_stopwords = set(special_stopwords)

def tweet_editor(tweet): 
    result = tweet
    
    patterns = [{'pattern': re.compile(r'^l+o+l+$'), 'fix': 'lol'}, 
                {'pattern': re.compile(r'^y+e+a+h*$'), 'fix': 'yea'}, 
                {'pattern': re.compile(r'^r+i+g+h+t+$'), 'fix': 'right'}, 
                {'pattern': re.compile(r'^o+m+f*g+$'), 'fix': 'omg'},
                {'pattern': re.compile(r'^h+m+$'), 'fix': 'hmm'},
                {'pattern': re.compile(r'^u+m+$'), 'fix': 'umm'},
                {'pattern': re.compile(r'^a+h+$'), 'fix': 'ah'},
                {'pattern': re.compile(r'^o+h+$'), 'fix': 'oh'},
                {'pattern': re.compile(r'^n+o+$'), 'fix': 'no'},
                {'pattern': re.compile(r'^y+e+s+$'), 'fix': 'yes'},
                {'pattern': re.compile(r'^(yr|yrs|year|years)$'), 'fix': 'year'},
                {'pattern': re.compile(r'^zzz+s*$'), 'fix': 'zzz'}]
    
    for pattern in patterns:
        result = re.sub(pattern['pattern'], pattern['fix'], tweet)
    
    return result

def repeat_letter_reducer(word):
    result = word
    
    patterns = [{'pattern': re.compile(r'^l+o+l+$'), 'fix': 'lol'}, 
                {'pattern': re.compile(r'^y+e+a+h*$'), 'fix': 'yea'}, 
                {'pattern': re.compile(r'^r+i+g+h+t+$'), 'fix': 'right'}, 
                {'pattern': re.compile(r'^o+m+f*g+$'), 'fix': 'omg'},
                {'pattern': re.compile(r'^h+m+$'), 'fix': 'hmm'},
                {'pattern': re.compile(r'^u+m+$'), 'fix': 'umm'},
                {'pattern': re.compile(r'^a+h+$'), 'fix': 'ah'},
                {'pattern': re.compile(r'^o+h+$'), 'fix': 'oh'},
                {'pattern': re.compile(r'^n+o+$'), 'fix': 'no'},
                {'pattern': re.compile(r'^y+e+s+$'), 'fix': 'yes'},
                {'pattern': re.compile(r'^(yr|yrs|year|years)$'), 'fix': 'year'},
                {'pattern': re.compile(r'^zzz+s*$'), 'fix': 'zzz'},
                {'pattern': re.compile(r'^.*(m|h|n)$'), 'fix': word}]
    
    for pattern in patterns:
        if pattern['pattern'].search(word):
            result = pattern['fix']
            break
    
    return result

def import_test_data(data_file):
    raw_obama_dataframe = pd.read_excel(data_file, sheetname='Obama', header=None, parse_cols="A,B")
    raw_obama_dataframe.rename(columns={0: 'id', 1: 'tweet'}, inplace=True)
    #raw_obama_dataframe['Class'] = pd.to_numeric(raw_obama_dataframe['Class'], errors='coerce', downcast='float')
        
    raw_romney_dataframe = pd.read_excel(data_file, sheetname='Romney', header=None, parse_cols="A,B")
    raw_romney_dataframe.rename(columns={0: 'id', 1: 'tweet'}, inplace=True)
    #raw_romney_dataframe['Class'] = pd.to_numeric(raw_romney_dataframe['Class'], errors='coerce', downcast='float')
    
    print(raw_obama_dataframe)
    print(raw_romney_dataframe)
    
    return raw_obama_dataframe, raw_romney_dataframe
    
def export_classifier_results(test_data, predictions, out_file):
    export_data = {'id': test_data['id'], 'prediction': predictions}
    export_df = pd.DataFrame(export_data)
    export_df.to_csv(out_file, sep='\t', index=False)

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

def alteredTweetTokenize(text):
    tweet_tokens = list(map(repeat_letter_reducer, tweet_tokenizer.tokenize(text.encode('ascii', 'ignore').decode('ascii'))))
    link_regex = re.compile(r'^(!|\.|,|<|>|:|;|{|}|\||~)$')
    return [word for word in tweet_tokens if (tweet_token_validator(word) and not (link_regex.search(word) or word in eng_stopwords))]

def score_reducer(accum, new_score, divisor):
    return {"precision": new_score["precision"]/divisor + accum["precision"], "recall": new_score["recall"]/divisor + accum["recall"], "fscore": new_score["fscore"]/divisor + accum["fscore"]}

def kFoldValidation(model, X, y, folds, stratified=None):
     shuffled_indices = np.random.permutation(len(X))
     remainder = len(X) % folds
     bin_size = int(len(X) / folds)
     remainder_indices = shuffled_indices[len(X)-remainder:]
     
     pos_scores = []
     neutral_scores = []
     neg_scores = []
     accuracies = []
     
     if not stratified:
         for x in range(1, folds+1):
             test_indices = shuffled_indices[(x-1)*bin_size:x*bin_size]
             train_indices = np.concatenate([shuffled_indices[:(x-1)*bin_size], shuffled_indices[x*bin_size:]])
             model.fit(X.iloc[train_indices], y.iloc[train_indices])
             predicted = model.predict(X.iloc[test_indices])
             precision, recall, fscore, support = precision_recall_fscore_support(y.iloc[test_indices], predicted, labels=[1, 0, -1])
             pos_scores.append({"precision": precision[0], "recall": recall[0], "fscore": fscore[0]})
             neutral_scores.append({"precision": precision[1], "recall": recall[1], "fscore": fscore[1]})
             neg_scores.append({"precision": precision[2], "recall": recall[2], "fscore": fscore[2]})
             accuracies.extend(predicted == y.iloc[test_indices])
     else:
        strat = StratifiedKFold(n_splits=folds, shuffle=True)
        for train_indices, test_indices in strat.split(X, y):
            model.fit(X.iloc[train_indices], y.iloc[train_indices])
            predicted = model.predict(X.iloc[test_indices])
            precision, recall, fscore, support = precision_recall_fscore_support(y.iloc[test_indices], predicted, labels=[1, 0, -1])
            pos_scores.append({"precision": precision[0], "recall": recall[0], "fscore": fscore[0]})
            neutral_scores.append({"precision": precision[1], "recall": recall[1], "fscore": fscore[1]})
            neg_scores.append({"precision": precision[2], "recall": recall[2], "fscore": fscore[2]})
            accuracies.extend(predicted == y.iloc[test_indices])
         
         
     pos = functools.reduce(functools.partial(score_reducer, divisor=folds), pos_scores, {"precision": 0, "recall": 0, "fscore": 0})
     neutral = functools.reduce(functools.partial(score_reducer, divisor=folds), neutral_scores, {"precision": 0, "recall": 0, "fscore": 0})
     neg = functools.reduce(functools.partial(score_reducer, divisor=folds), neg_scores, {"precision": 0, "recall": 0, "fscore": 0})
     accuracy = np.mean(accuracies)
     
     print(pos)
     print(neutral)
     print(neg)
     print(accuracy)
         
if __name__ == "__main__":
    init_time = time.process_time()
    
    obama_dataframe, romney_dataframe = import_and_filter(dropnan=True)
    print(obama_dataframe['Class'].unique())
    print(romney_dataframe['Class'].unique())
    
    rom_sentiment_clf = Pipeline([('vect', CountVectorizer(tokenizer=alteredTweetTokenize, ngram_range=(1,3))),
                              #('tfidf', TfidfTransformer()),
                              #('clf', MultinomialNB(alpha=0.84, class_prior=[0.40,0.40,0.20]))])
                               ('clf', LinearSVC(class_weight="balanced"))])
    obo_sentiment_clf = Pipeline([('vect', CountVectorizer(tokenizer=alteredTweetTokenize, ngram_range=(1,3))),
                              #('tfidf', TfidfTransformer()),
                              #('clf', MultinomialNB(alpha=0.84))])
                               ('clf', LinearSVC(class_weight="balanced"))])
    
    def clean_imported_tweets(tweets):
        for tweet in tweets:
            ascii_tweet = tweet.encode('ascii', 'ignore').decode('ascii')
            tweet_tokens = list(map(repeat_letter_reducer, tweet_tokenizer.tokenize(tweet)))
            clean_tweet = ' '.join(tweet_tokens)
            print(clean_tweet)
    
    kFoldValidation(obo_sentiment_clf, obama_dataframe['Annotated Tweet'], obama_dataframe['Class'], 10, stratified=True)
    kFoldValidation(rom_sentiment_clf, romney_dataframe['Annotated Tweet'], romney_dataframe['Class'], 10, stratified=True)
    
    total_time = time.process_time() - init_time
    print("Total run time: %f" % total_time)



# obama_dataframe, romney_dataframe = import_and_filter(dropnan=True)
# rom_sentiment_clf = Pipeline([('vect', CountVectorizer(tokenizer=alteredTweetTokenize, ngram_range=(1,3))),
#                               #('tfidf', TfidfTransformer()),
#                               #('clf', MultinomialNB(alpha=0.84, class_prior=[0.40,0.40,0.20]))])
#                                ('clf', LinearSVC(class_weight="balanced"))])
# obo_sentiment_clf = Pipeline([('vect', CountVectorizer(tokenizer=alteredTweetTokenize, ngram_range=(1,3))),
#                               #('tfidf', TfidfTransformer()),
#                               #('clf', MultinomialNB(alpha=0.84))])
#                                ('clf', LinearSVC(class_weight="balanced"))])
# 
# init_time = time.process_time()
# rom_X_train, rom_X_test, rom_y_train, rom_y_test = train_test_split(romney_dataframe['Annotated Tweet'], romney_dataframe['Class'], test_size=0.20, random_state=0)
# rom_sentiment_clf.fit(rom_X_train, rom_y_train)
# rom_predicted = rom_sentiment_clf.predict(rom_X_test)
# print(metrics.classification_report(rom_y_test, rom_predicted))
# 
# obo_X_train, obo_X_test, obo_y_train, obo_y_test = train_test_split(obama_dataframe['Annotated Tweet'], obama_dataframe['Class'], test_size=0.20, random_state=0)
# obo_sentiment_clf.fit(obo_X_train, obo_y_train)
# obo_predicted = obo_sentiment_clf.predict(obo_X_test)
# print(metrics.classification_report(obo_y_test, obo_predicted))
# total_time = time.process_time() - init_time
# print("Total run time: %f" % total_time)




# print(obama_dataframe['Annotated Tweet'][200:300].values)
# obama_dataframe['Annotated Tweet'] = obama_dataframe['Annotated Tweet'].apply(tweet_editor)
# print(obama_dataframe['Annotated Tweet'][1:10])



#SVM Text 
# obama_dataframe, romney_dataframe = import_and_filter(dropnan=True)
# print(obama_dataframe['Class'].unique())
# print(romney_dataframe['Class'].unique())
# 
# obo_svm = Pipeline([('vect', CountVectorizer(tokenizer=alteredTweetTokenize, ngram_range=(1,3))),
#                               #('tfidf', TfidfTransformer()),
#                               ('clf', LinearSVC(random_state=0))])
# 
# 
# obo_svm.fit(obama_train_X, obama_train_y)
# predicted = obo_svm.predict(obama_test_X)
# print(metrics.classification_report(obama_test_y, predicted))
# 
# 
# rom_svm = Pipeline([('vect', CountVectorizer(tokenizer=alteredTweetTokenize, ngram_range=(1,3))),
#                               #('tfidf', TfidfTransformer()),
#                               ('clf', LinearSVC(random_state=0, class_weight={1.: 3.0, -1.: 0.5, 0.: 1.0}))])
# 
# 
# rom_svm.fit(romney_train_X, romney_train_y)
# predicted = rom_svm.predict(romney_test_X)
# print(metrics.classification_report(romney_test_y, predicted))

# obo_params = [{'vect__tokenizer': [alteredTweetTokenize], 'vect__ngram_range':[(1,3), (2,3), (2,4)], 'clf__kernel': ['linear', 'rbf', 'sigmoid'], 'clf__C': [1.0, 10.0], 'clf__class_weight': [None, 'balanced']},
#               {'vect__tokenizer': [alteredTweetTokenize], 'vect__ngram_range':[(1,3), (2,3), (2,4)], 'clf__kernel': ['poly'], 'clf__C': [1.0, 10.0], 'clf__class_weight': [None, 'balanced'], 'clf__degree': [1, 2, 3, 4]}]
# 
# obo_est = Pipeline([('vect', CountVectorizer(tokenizer=alteredTweetTokenize, ngram_range=(1,3))),
#                               #('tfidf', TfidfTransformer()),
#                               ('clf', SVC())])
# clf = GridSearchCV(obo_est, obo_params)
# clf.fit(obama_dataframe['Annotated Tweet'], obama_dataframe['Class'])
# print(clf.best_params_)

#rom_params = [{'clf__kernel': ['linear', 'rbf', 'sigmoid'], 'clf__C': [1.0, 10.0], 'clf__class_weight': [None, 'balanced']},
#              {'clf__kernel': ['poly'], 'clf__C': [1.0, 10.0], 'clf__class_weight': [None, 'balanced'], 'clf__degree': [1, 2, 3, 4]}]
#
#rom_est = Pipeline([('vect', CountVectorizer(tokenizer=alteredTweetTokenize, ngram_range=(1,3))),
#                              #('tfidf', TfidfTransformer()),
#                              ('clf', SVC())])
#clf = GridSearchCV(rom_est, rom_params)
#clf.fit(romney_dataframe['Annotated Tweet'], romney_dataframe['Class'])
#print(clf.best_params_)

