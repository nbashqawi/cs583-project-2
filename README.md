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

### Further Iterations
#### Preprocessing Steps

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
