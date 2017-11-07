# cs583-project-2
Nicholas Bashqawi  
CS 583 Project 2  
Tweet Sentiment Classification  



## Description:


## Requirements:

## Known Limitations/Errors:

## How to Run:

## Methods Tried:
### First attempt
First attempt followed example described here: http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html#evaluation-of-the-performance-on-the-test-set.  
Data was imported and any columns with NaN class values or a class of 2 were removed. Scikit-Learn was used to tokenize the data into unigrams and standardize with tf-idf scheme. A multinomial Naive Baysian classifier was then trained using a training set and used to predict the class values of data in a test set. The Scikit-Learn metrics module was used to calculate precision, recall, F1-score of all three classes. Accuracy was also calculated.  
The primary reason for this first attempt was to get a handle on a possible data import, cleaning, and classification workflow. This will aid in maturing the workflow, choosing a classifier, and deciding how best to clean and prepare the data.

## Packages Used:
  * Anaconda Distribution
    * Version: 5.0.1
    * Website: https://www.anaconda.com/distribution/
  * NLTK
    * Version: 3.2.5
    * Website: http://www.nltk.org/
    
## Bibliography:
* Bing Liu. 2006. Web Data Mining: Exploring Hyperlinks, Contents, and Usage Data (Data-Centric Systems and Applications). Springer-Verlag New York, Inc., Secaucus, NJ, USA.
