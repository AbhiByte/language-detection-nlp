import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
import seaborn


df = pd.read_csv('data/Language Detection.csv')
print(df.head())

cv = CountVectorizer()
x = np.array(df["Text"]) 
X = cv.fit_transform(x)
y = np.array(df["Language"])
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=45)

#Model 1
mNB_model = MultinomialNB()
mNB_model.fit(X_train, y_train)
y_pred_mnb = mNB_model.predict(X_test)
acc_score_mnb = accuracy_score(y_pred_mnb, y_test)
cm_mnb = confusion_matrix(y_test, y_pred_mnb)
print('accuracy %s' % acc_score_mnb)

#Model 2
SGD_model = SGDClassifier()
SGD_model.fit(X_train, y_train)
y_pred_sgd = SGD_model.predict(X_test)
acc_score_sgd = accuracy_score(y_pred_sgd, y_test)
cm_sgd = confusion_matrix(y_test, y_pred_sgd)
print('accuracy %s' % acc_score_sgd)

#Plot mnb results
plt.figure(figsize=(15, 10))
seaborn.heatmap(cm_mnb, annot=True)
plt.show()














