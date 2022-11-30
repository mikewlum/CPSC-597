'''
Created on Oct 11, 2022
@author: Michael Lum
CPSC 597
'''

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.metrics._classification import accuracy_score
from sklearn.metrics import mean_absolute_error, r2_score


#download the data
file = pd.read_csv("D:\CPSC 597\ph-data.csv")

#normalize data using per Test Case 6 of KNNmodel.py Test Plan 
#b = file['blue'].div(file['green']+file['red']+file['blue'])
#g = file['green'].div(file['green']+file['red']+file['blue'])
#r = file['red'].div(file['green']+file['red']+file['blue'])
#file['blue'] = b
#file['green'] = g
#file['red'] = r

    
#split data into x variables
xdata=file.drop("label", axis='columns')
#split data into y variables
ydata=file["label"]

#splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(xdata, ydata, test_size=.10, random_state = None)

#normalize data per Test Case 5 of of KNNmodel.py Test Plan
#X_train = preprocessing.normalize(X_train)
#X_test = preprocessing.normalize(X_test)

#creation and fitting of mode
model = KNeighborsClassifier(n_neighbors= 5, weights = None, algorithm = 'ball_tree')
model.fit(X_train, y_train)
    
#predict reading from ph input
predict = model.predict(X_test)

result = accuracy_score(y_test,predict)
mae = mean_absolute_error(y_test,predict)
r2 = r2_score(y_test, predict)

print("Accuracy:",result)
print("Mean Absolute Error:",mae)
print("R squared value:", r2)
