# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 22:16:43 2019

@author: Sridhar
"""

"""LOAD THE DATA"""
import pandas as pd
dataset = pd.read_csv('Climate_Change_Data.csv',encoding='latin-1') #Set Work Directory

#########################################################################################################
"""PREPROCESSING"""
##missing values
dataset.isnull().sum()

##normalising your data
import re
contractions_dict = {
    'didn\'t': 'did not',
    'don\'t': 'do not',
    "aren't": "are not",
    "can't": "cannot",
    "cant": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "didnt": "did not",
    "doesn't": "does not",
    "doesnt": "does not",
    "don't": "do not",
    "dont" : "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i had",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'm": "i am",
    "im": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had",
    "she'd've": "she would have",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they had",
    "they'd've": "they would have",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who's": "who is",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have"
    }

contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))

def expand_contractions(s, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, s)
dataset['Tweets']=dataset['Tweets'].apply(expand_contractions)

import nltk
import re

stop_words = nltk.corpus.stopwords.words('english')

def normalize_document(doc):
    doc = doc.lower()
    doc=re.sub(r'^https?:\/\/.*[\r\n]*', "", doc)
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
    doc= re.sub("rt","",doc)
    doc = doc.strip()
    # tokenize document
    tokens = nltk.word_tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc
dataset['Tweets']=dataset['Tweets'].apply(normalize_document)

##remove html and xml(links)
import re
from bs4 import BeautifulSoup
def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    [s.extract() for s in soup(['iframe', 'script'])]
    stripped_text = soup.get_text()
    stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
    return stripped_text

dataset['Tweets']=dataset['Tweets'].apply(strip_html_tags)
dataset['Tweets']

##############################################################################################################

##creating positive and negative statement
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def analyze_sentiment_vader_lexicon(review, 
                                    threshold=0.1,
                                    verbose=False):    
    # analyze the sentiment for review
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(review)
    # get aggregate scores and final sentiment
    agg_score = scores['compound']
    final_sentiment = 'positive' if agg_score >= threshold                                   else 'negative'
    if verbose:
        # display detailed sentiment statistics
        positive = str(round(scores['pos'], 2)*100)+'%'
        final = round(agg_score, 2)
        negative = str(round(scores['neg'], 2)*100)+'%'
        neutral = str(round(scores['neu'], 2)*100)+'%'
        
        sentiment_frame = pd.DataFrame([[final_sentiment, final, positive,
                                        negative, neutral]]
                                                            #  codes=[[0,0,0,0,0],[0,1,2,3,4]]
                                                             )
        print(sentiment_frame)
    
    return final_sentiment

nltk.download('vader_lexicon')

review = """tspooky guardian cli amatechange play words weather cli amate always changing amp cyclical httpstcoopgncwrdy"""
analyze_sentiment_vader_lexicon(review, 
                                    threshold=0.1,
                                    verbose=False)

dataset['sentiment_new'] = dataset['Tweets'].apply(analyze_sentiment_vader_lexicon)
dataset['sentiment_new']

dataset['sentiment_new'].value_counts()

#############################################################################################################

##split the data

reviews = dataset['Tweets'].values
sentiments = dataset['sentiment_new'].values

train_reviews = reviews[:25000]
train_sentiments = sentiments[:25000]

test_reviews = reviews[25000:]
test_sentiments = sentiments[25000:]

#############################################################################################################

##create bag of words
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# build BOW features on train reviews
cv = CountVectorizer(binary=False, min_df=5, max_df=1.0, ngram_range=(1,2))
cv_train_features = cv.fit_transform(train_reviews)

# transform test reviews into features
cv_test_features = cv.transform(test_reviews)

print('BOW model:> Train features shape:', cv_train_features.shape, ' Test features shape:', cv_test_features.shape)

# Logistic Regression model on BOW features
from sklearn.linear_model import LogisticRegression

# instantiate model
lr = LogisticRegression(penalty='l2', max_iter=500, C=1, solver='lbfgs', random_state=42)

# train model
lr.fit(cv_train_features, train_sentiments)

lr_bow_predictions_tr = lr.predict(cv_train_features)

##accuracy for train data
from sklearn.metrics import accuracy_score
accuracy_score(train_sentiments, lr_bow_predictions_tr)

# predict on test data
lr_bow_predictions = lr.predict(cv_test_features)

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
np.set_printoptions(precision=2, linewidth=80)

labels = ['negative', 'positive']
print(classification_report(test_sentiments, lr_bow_predictions))
pd.DataFrame(confusion_matrix(test_sentiments, lr_bow_predictions), index=labels, columns=labels)

accuracy_score(test_sentiments, lr_bow_predictions)

################################################################################################################

##create TFIDF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# build TFIDF features on train reviews
tv = TfidfVectorizer(use_idf=True, min_df=5, max_df=1.0, ngram_range=(1,2),
                     sublinear_tf=True)
tv_train_features = tv.fit_transform(train_reviews)

# transform test reviews into features
tv_test_features = tv.transform(test_reviews)
print('TFIDF model:> Train features shape:', tv_train_features.shape, ' Test features shape:', tv_test_features.shape)

# Logistic Regression model on TF-IDF features

# train model
lr.fit(tv_train_features, train_sentiments)
lr_tfidf_predictions_tr = lr.predict(tv_train_features)
# predict on test data
lr_tfidf_predictions = lr.predict(tv_test_features)

from sklearn.metrics import accuracy_score

accuracy_score(train_sentiments, lr_tfidf_predictions_tr)

labels = ['negative', 'positive']
print(classification_report(test_sentiments, lr_tfidf_predictions))
pd.DataFrame(confusion_matrix(test_sentiments, lr_tfidf_predictions), index=labels, columns=labels)

accuracy_score(test_sentiments, lr_tfidf_predictions)

##############################################################################################################














































