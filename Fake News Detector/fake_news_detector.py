import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#Reading the csv file. 
#Here, the path should be replaced with the path that takes you to the location of the csv file.
dataframe = pd.read_csv('C:/Users/pedro/OneDrive/Software Development/Portfólio/Python Projects/Fake News Detector/news.csv')

dataframe.shape
dataframe.head()

#Getting Labels
labels = dataframe.label
labels.head()

#Split the dataset
first_train, first_test, second_train, second_test = train_test_split(dataframe['text'], labels, test_size = 0.2, random_state = 7)

tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', max_df = 0.7)

tfidf_train = tfidf_vectorizer.fit_transform(first_train)
tfidf_test = tfidf_vectorizer.transform(first_test)

passive_classifier = PassiveAggressiveClassifier(max_iter = 50)
passive_classifier.fit(tfidf_train, second_train)

second_pred = passive_classifier.predict(tfidf_test)
score = accuracy_score(second_test, second_pred)

print(f'Accuracy: {round(score * 100, 2)}%')

#Build and print confusion matrix
print(confusion_matrix(second_test, second_pred, labels=['FAKE','REAL']))