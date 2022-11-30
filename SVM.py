'''
Created on Oct 11, 2022
@author: Michael Lum
CPSC 597
'''

# import SVC classifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
from sklearn.metrics import accuracy_score


#download the data
file = pd.read_csv("D:\CPSC 597\ph-data.csv")

#split data into x variables
xdata=file.drop("label", axis='columns')
#split data into y variables
ydata=file["label"]

#splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(xdata, ydata, test_size=.10, random_state = None)

#creation and fitting of mode
modelSCV= SVC(C=1000, kernel = 'sigmoid')
modelSCV.fit(X_train, y_train)

#predict reading from ph input
pred = modelSCV.predict(X_test)

#model assessment
mae = mean_absolute_error(y_test,pred)
r2 = r2_score(y_test, pred)
result = accuracy_score(y_test,pred)


print("Accuracy:",result)
print("Mean Absolute Error:",mae)
print("R squared value:", r2)
