import matplotlib as plt
import numpy as np
import pandas as pd
import pickle 
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes()
print(diabetes.DESCR)
db = pd.DataFrame(diabetes.data)
db['Prediction'] = diabetes.target
X = db.drop('Prediction',axis=1)
Y = db['Prediction']
print(X.shape)
print(Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=101)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

lm = LinearRegression()
lm.fit(X_train,Y_train)
print(lm.score)

Y_pred = lm.predict(X_test)
print(lm.score(X_train,Y_train))

pickle.dump(lm, open('lrmodel.pkl','wb'))
model = pickle.load(open('lrmodel.pkl','rb'))
 

