import pandas as pd
from nltk import word_tokenize, sent_tokenize, TweetTokenizer
from nltk.corpus import stopwords
import re
import sys, os
import time

init_time = time.process_time()
eng_stopwords = set(stopwords.words('english'))
tweet_tokenizer = TweetTokenizer()

data_file_location = r"..\resources\training-Obama-Romney-tweets.xlsx"

raw_obama_dataframe = pd.read_excel(data_file_location, sheetname='Obama', header=0, parse_cols="D,E");
raw_romney_dataframe = pd.read_excel(data_file_location, sheetname='Romney', header=0, parse_cols="D,E");

#print("Before Removal")
#print("-----------------------------------------------")
#print(raw_obama_dataframe.info())
#print("\n")
#print(raw_romney_dataframe.info())
#print("-----------------------------------------------\n")

#print("Poor Data Info")
#print("-----------------------------------------------")
#print(raw_obama_dataframe[raw_obama_dataframe['Annotated Tweet'].isnull()])
#print(raw_obama_dataframe[pd.to_numeric(raw_obama_dataframe['Class'], errors='coerce').isnull()].shape)
#print(raw_romney_dataframe[pd.to_numeric(raw_romney_dataframe['Class'], errors='coerce').isnull()]['Class'].shape)
#print(raw_obama_dataframe[pd.to_numeric(raw_obama_dataframe['Class'], errors='coerce').isnull()]['Class'].unique())
#print(raw_romney_dataframe[pd.to_numeric(raw_romney_dataframe['Class'], errors='coerce').isnull()]['Class'].unique())
#print("-----------------------------------------------\n")

#print("After Removal")
#print("-----------------------------------------------")
obama_dataframe = raw_obama_dataframe.copy(deep=True)
obama_dataframe['Class'] = pd.to_numeric(obama_dataframe['Class'], errors='coerce', downcast='float')
obama_dataframe = obama_dataframe.dropna()
#print(obama_dataframe.info())

romney_dataframe = raw_romney_dataframe.copy(deep=True)
romney_dataframe['Class'] = pd.to_numeric(romney_dataframe['Class'], errors='coerce', downcast='float')
romney_dataframe = romney_dataframe.dropna()
#print(romney_dataframe.info())
vocabulary = dict()

def is_valid(token):
    valid = (not re.search(r"//t\.co.*", token))
    return valid

def process_text(sentence):
    tokens = tweet_tokenizer.tokenize(sentence)
    for token in tokens:
        if is_valid(token):
            vocabulary[token] = vocabulary.get(token, 0) + 1
    
    
obama_dataframe['Annotated Tweet'].apply(process_text)
romney_dataframe['Annotated Tweet'].apply(process_text)

print(vocabulary)

total_time = time.process_time() - init_time
print(total_time)
