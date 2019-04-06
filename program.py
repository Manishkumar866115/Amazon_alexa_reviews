
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer

file_path="C:/Users/Manish/Downloads/smartphone-review/amazon_alexa.tsv"
dataset=pd.read_csv(file_path,sep='\t')

#getting porter stemmer for stemming and stopwords for eliminating them from reviews
corpus1=[]
ps=PorterStemmer()
stopwords=set(stopwords.words("english"))
for i in range(0,3):
    review=dataset["verified_reviews"][i]
    review = re.sub('[^a-zA-Z]', ' ', dataset['verified_reviews'][i] )
    review=review.lower()
    words=word_tokenize(review)
    d=list()
    for w in words:
        if w not in stopwords:
            d.append(ps.stem(w))
    review=" ".join(d)
    corpus1.append(review)
  

#creating a bag of words using countVectorizer from sklearn   
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X1=cv.fit_transform(corpus1).toarray()
y1=dataset.iloc[:,4].values


#splitting the dataset
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.20,random_state=0)

#Using the kaggle's library xgboost for classification
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

#result on the test dataset
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
