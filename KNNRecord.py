'''
Created on Oct 11, 2022
@author: Michael Lum
CPSC 597
'''

from datetime import date
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from PIL import Image
from PIL import UnidentifiedImageError
import numpy as np

#download the data
file = pd.read_csv("D:\CPSC 597\standards\RGBdata.csv")

#dictionary to convert pH values into int. KNN model will only take int classifiers
updatepH = {5.5 :0, 5.8 :1, 6.0 :2, 6.2 :3, 6.4 :4, 6.6 :5, 6.8 :6, 7.0 :7, 7.2 :8, 7.4 :9, 7.6 :10, 8.0 :11}
#update pH column in data frame as int in dictionary
file["pH"] = file["pH"].map(updatepH)

#split data into x variables
xdata=file.drop("pH", axis='columns')
#split data into y variables
ydata=file["pH"]

#creation and fitting of model, k=5
model=KNeighborsClassifier()
model.fit(xdata, ydata)

#get image to process and extract RGB values
while True:
    try:
        pHinputfile = input("Type 'Exit' to quit or enter location of file to read: \n")
        if pHinputfile == "Exit":
            exit()
        #attempt opening file
        arr = np.array(Image.open(pHinputfile))
        break
    except FileNotFoundError as err:
        print("File not found or bad directory.")
    except PermissionError as err:
        print("File not found or bad directory.")
    except UnidentifiedImageError as err:
        print("File must be .png image")

#empty variables to hold RGB values
Red = None
Green = None 
Blue = None
#creating arrays for image
#calculating the dimensions (RBG or greyscale) for each array
arr_dim = arr.ndim 
# extacting RGB
arr_mean = np.mean(arr, axis=(0,1))

#the print command below is for diagnostic purposes
#print(f'[R={arr_mean[0]:.1f},  G={arr_mean[1]:.1f}, B={arr_mean[2]:.1f}, ALPHA={arr_mean[3]:.1f} ]')

#capture the RGB elements of photo to one decimal place
Red = round(arr_mean[0], 1)
Green = round(arr_mean[1], 1)
Blue = round(arr_mean[2], 1)

#create a new diction of RGB values to creates image data frame
RGBd = {"Red": [Red], "Green": [Green], "Blue": [Blue]}
#Create test set data frame
testset = pd.DataFrame(data=RGBd)
    
#predict reading from ph input
predict = model.predict(testset)

#extract prediction value from matrix. Will use value as an index of pHarray
index = predict[0]
pHarray= [5.5, 5.8, 6.0, 6.2, 6.4, 6.6, 6.8, 7.0, 7.2, 7.4, 7.6, 8.0]

#ouputs the pH reading
print("Ph value:", pHarray[index])

#creates .txt file with pH reading and date
while True:
    recordtitle = input("Type 'Exit' to quit or enter record directory and title: \n")

    if recordtitle == "Exit":
        exit()
    else:
        try:
            phrecord = open(recordtitle, "w")
            phrecord.write("Ph read is:")
            phrecord.write(str(pHarray[index]))
            phrecord.write("\nToday's date:")
            phrecord.write(str(date.today()))
            phrecord.close()
            exit()
        except PermissionError as err:
            print("Bad directory or title.")