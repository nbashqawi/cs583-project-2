# cs583-project-2
Nicholas Bashqawi  
CS 583 Project 2  
Tweet Sentiment Classification  



## Description:
A Tweet sentiment classifier for CS 583 project 2 at University of Ilinois at Chicago. The objective is to train a classifier, using a list of labelled tweets about Obama and Romney from the 2012 election, to label tweets as positive, negative, or neutral. In the end, two classifiers are constructed, one for Obama and one for Romeny. These two classifiers will then be used to classify the opinion orientaion of unlabelled tweets.

## Requirements:
* Input spreadhseet
  * Labelled Obama data
  * Labelled Romney data
* Test data spreadsheet

## Known Limitations/Errors:

## How to Run:
### Command-line:

## Methods Tried:
### First attempt
First attempt followed example described here: http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html#evaluation-of-the-performance-on-the-test-set.  
Data was imported and any columns with NaN class values or a class of 2 were removed. Scikit-Learn was used to tokenize the data into unigrams and standardize with tf-idf scheme. A multinomial Naive Baysian classifier was then trained using a training set and used to predict the class values of data in a test set. The Scikit-Learn metrics module was used to calculate precision, recall, F1-score of all three classes. Accuracy was also calculated.  
The primary reason for this first attempt was to get a handle on a possible data import, cleaning, and classification workflow. This will aid in maturing the workflow, choosing a classifier, and deciding how best to clean and prepare the data.

### Further Iterations
#### Preprocessing Steps
* All data with missing or incorrect labels were removed
* All data instances with class 2 were removed, as per instructions
* Data were tokenized using the NLTK TweetTokenizer
* Common instances or word shortening or letter repition are handled using regular expressions
  * Currently, these are manually determined
* Stopword removal using NLTK english stopword dictionary
  * Added additional words: 'omg', 'lol', 'umm', 'hmm', 'ah', 'oh', 'yea' (likely to change)
* Removed all non-ASCII characters
* Removed urls (w/ regex pattern r"//t\.co.*")
* Removed certain punctuation (w/ regex r'^(!|\.|,|<|>|:|;|{|}|\||~)$') 
* Tried TF-IDF weighting scheme, though currently not using
* Data were vectorized using Scikit-learn's CountVectorizer function
  * Multiple n-gram ranges were tried; 1-3 is the current range


#### Features
* Current features are simply n-grams, where n ranges from 1 to 3

#### Model Evaluation
* Obama
  * Current: 10-fold cross-validation
* Romney
  * First attempt: regular 10-fold cross-validation
  * Current: Stratified 10-fold cross-validation
* Both
  * n-fold cross-validation is done using a custom function
    * Indicies are shuffled
    * Stratification is done using scikit-learn's StratifiedKFold class
  * F-score, precision, and recall calculated at each fold using scikit-learn's precision_recall_fscore_support
  * Accuracy calculated at each fold
  * All scores averaged after gathering scores from all folds

#### Models Tried
* Multinomial Naive Bayes
* Support Vector Machine
  * Linear Kernel

## Packages Used:
* Anaconda Distribution
  * Version: 5.0.1
  * Website: https://www.anaconda.com/distribution/
  * Packages:
    * Scikit-learn
      * Version: 0.19.1
      * Website: http://scikit-learn.org/stable/
    * Pandas
      * Version: 0.20.3
      * Website: https://pandas.pydata.org/
    * SciPy
      * Version: 0.19.1
      * Website: https://www.scipy.org/
* NLTK
  * Version: 3.2.5
  * Website: http://www.nltk.org/
    
## Bibliography:
* Bing Liu. 2006. Web Data Mining: Exploring Hyperlinks, Contents, and Usage Data (Data-Centric Systems and Applications). Springer-Verlag New York, Inc., Secaucus, NJ, USA.
* Géron, Aurélien. 2017. Hands-on machine learning with Scikit-Learn and TensorFlow: concepts, tools, and techniques to build intelligent systems.
* http://scikit-learn.org/stable/index.html
* https://pandas.pydata.org/pandas-docs/stable/index.html
* http://www.nltk.org/book/
* https://docs.scipy.org/doc/numpy-dev/index.html
