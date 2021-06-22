
from sklearn.model_selection import train_test_split
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from sklearn.model_selection import train_test_split
import pandas as pd 
import io

# Read the train data

train = pd.read_csv('train_ex2_dl2021b.csv')
train.head()

# Read the test data
uploaded = files.upload()
test = pd.read_csv('test_ex2_dl2021b.csv')
test.head()

# Pre-processing

# Invert capital letters to small letter
train['sentence']=train['sentence'].str.lower()
test['sentence']=test['sentence'].str.lower()

# Removing punctuation marks and special characters
def Clean(word):
    return re.sub('[^A-Za-z0-9]+', '', word)
train['sentence'] = train['sentence'].apply(lambda y: " ".join(Clean(x) for x in y.split()))
test['sentence'] = test['sentence'].apply(lambda y: " ".join(Clean(x) for x in y.split()))

# Split the train data for train and test 
sentences = train['sentence'].values
y=train['label'].values
sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.2, random_state=1000)

from sklearn.feature_extraction.text import CountVectorizer
# Build a vector representation for the train & test text

vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)

X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)

from keras.models import Sequential
from keras import layers
from keras.backend import clear_session

# clear model history & session
clear_session()


# Build the model
input_dim = X_train.shape[1]  # Number of features

model = Sequential()
model.add(layers.Dense(16, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(2, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
#fiting the model
model.fit(X_train, y_train,epochs=200,verbose=False,validation_data=(X_test, y_test),batch_size=10)

# Evaluate the train and the test data
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

# Build a vector representation for the test text
test_sent=test['sentence'].values
X_test = vectorizer.transform(test_sent)
preds=model.predict(X_test)

import numpy as np
# Rounding the output of the model to 0 or 1
ans=[]
for i in preds:
  if i==0.5:
    i+=0.001
  ans.append(int(np.round(i)[0]))

# Saving the results in a csv file
Test_pred = pd.DataFrame({"sid":test['sid'],"label":ans})
Test_pred.to_csv('Results.csv') 


