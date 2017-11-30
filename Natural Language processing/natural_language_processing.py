# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 14:20:08 2017

@author: NirajS
"""

#importing the NLP libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
#ignore the double quotes "" to not use as a delimiter
dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter = '\t',quoting = 3)

#clean the data 
import re
import nltk 
nltk.download('stopwords')   #get words like 'this the and' not imported yet
from nltk.corpus import stopwords #importing here after d/w
from nltk.stem.porter import PorterStemmer #for stemming loved -> love
corpus = []

for i in range(0,1000):
    #substitute the non alpahbets by space
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    #to lowercase
    review = review.lower()
    #split the words as tokens
    review = review.split()
    
    ps = PorterStemmer()
    #set is used to exe faster than the list
    #stem is used on the filtered words
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] 
    #join back the split words
    review = " ".join(review)
    corpus.append(review)

#create the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()