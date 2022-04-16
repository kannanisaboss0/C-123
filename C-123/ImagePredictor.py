#----------------------------------------------------------C-123-------------------------------------------------------------------#
#-----------------------------------------------------ImagePredictor.py-----------------------------------------------------#

import numpy as np 
import pandas as pd 
import cv2
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.model_selection import train_test_split as TTs
from sklearn.metrics import accuracy_score as a_s
import sys
import webcolors as wb
import random as rd
import time as tm
from PIL import Image
import PIL.ImageOps

'''
Imporitng Modules:
-numpy (np)
-pandas (pd)
-cv2
-sklearn.linear_model:- LogisticRegression (LogReg)
-sklearn.model_selection:- train_test_split (TTs)
-sklearn.metrics:- accuracy_score (a_s)
-sys
-webcolors (wb)
-random (rd)
-time as tm
-PIL:- Image 
-PIL.ImageOps
'''

#Printing introductory messages and showing an image of Raymond Kurzeil
#(i)
print("Welcome to ImagePredictor.py")
tm.sleep(2.3)

#(ii)
print("We provide alphabet image detection services")
tm.sleep(1.3)

#(iii)
print("Alphabet image detection finds its way in many facets of our lives- from school teaching to criminal investigation")
tm.sleep(3)

#(iv)
print("Image detection is a subordinate of OCR or Optical Character Recognition")
tm.sleep(1.2)

#(v)
print("The methodology was first developed and put to effective use by Raymond Kurzweil in 1974.")
tm.sleep(3.4)

#(vi) Opening the image of Raymond Kurzeil using PIL
with Image.open("OCR-1.png") as file:
    file.show(title="The Man Behind OCR - Raymond Kurzweil")

#(vii)
print("Kurzweil creeated a machine that can recognize alphabets and numbers of many fonts and process them into computer genrated bit text. This wonder machine was supported by two hallmark technologies at the time:-CCD Scanners and Text-to-Speech Synthesizers.")
tm.sleep(2.3)

#(viii)
print("Now it's possible to conduct such actions in the comfort of our home, and effortless availability of python libarries~ Another way to gauge the exponential improvement of technology and subsequently, our lives")

#Printing the loading files prompt
print("Loading files...")
tm.sleep(1.2)

#Reading data and assigning t variable of Y and X respectively
Y_val=pd.read_csv("data.csv")
X_val=np.load("image.npz")["arr_0"]

#Printing the sgregating data prompt
print("Segregating data...")
tm.sleep(0.2)

#Using train_test_split to accoridngly split the data for trainng and testing
X_train,X_test,Y_train,Y_test=TTs(X_val,Y_val,train_size=7500,test_size=2500,random_state=9)

#Scaling the X values to the ratios of 255
X_train=X_train/255
X_test=X_test/255

#Printing the training data prompt
print("Training data (this might take a while)...")

#Initiating a Logistic Regression classifier and fitting the data using train X and Y values
LR=LogReg(solver="saga",multi_class="multinomial")
LR.fit(X_train,Y_train)

#Predicting the data using the test X values
Y_prediction=LR.predict(X_test)

#Calculating the accuracy score sing test Y and predicted Y values
accuracy_sc=a_s(Y_test,Y_prediction)

#Printing the accuracy score in the form of a percentage
print("Overall accuracy is {}%".format(accuracy_sc*100))

#Printing the prediciton done prompt
print("Prediciton done")
tm.sleep(0.2)

#Printing the enter parameters prompt
print("Please enter the following parameters:")

#Adding an input for box size
input_float=input("Please provide a box size (Recommended:{}-{})-> ".format(rd.randint(50,70),rd.randint(100,110)))

#Adding a try block to the code to prevent drastic errors caused by user's input
#Try block
try:

    #Converting the value to a float
    input_float=float(input_float)

#Except block
except:

    #Printing the error message and using sys to terminate the program
    print("Error: Please provide a valid input") 
    sys.exit()   

#Adding an input for color size
input_color=input("Please provide the color for the box->")

#Adding a try block to the code to prevent drastic errors caused by user's input
#Try block
try:

    #Using webcolors to convert the string into a rgb tuple
    input_color_bgr=wb.name_to_rgb(input_color)
    
#Except block    
except:

    #Printing the error message and using sys to terminate the program
    print("Error: Please provide a valid color name")
    sys.exit()

#Printing the preparing camera prompt
print("Preparing camera, permission might have to be granted to the program...")
tm.sleep(2.4)

#Capturing the frames
video=cv2.VideoCapture(0)

#Using a while block to indefenitely extend the prediciton alogorithm 
while(True):

    #Reading from the capture
    ret_r,frame_r=video.read()

    #Converting the image to grayscale and extracting its shape 
    gray_im=cv2.cvtColor(frame_r,cv2.COLOR_BGR2GRAY)
    height_fr,width_fr=gray_im.shape

    #Verifying whether the height or width of the cmaera frame are lesser than the inputted values

    #Case-1 -Either of the dimensions are lesser than the width or height 
    if(height_fr<input_float or width_fr<input_float):

        #Releasing the video, closing all active ouput windows and terminating the program
        video.release()
        cv2.destroyAllWindows()
        sys.exit()

    #Creating two tuples for the upper left and bottom right reaches of the rectangle
    upper_left=(int(width_fr/2-input_float),int(height_fr/2-input_float))
    bottom_right=(int(width_fr/2+input_float),int(height_fr/2+input_float))

    #Creating a rectangle for both the colored and grayscaled frames
    cv2.rectangle(gray_im,upper_left,bottom_right,(255,255,255),2)
    cv2.rectangle(frame_r,upper_left,bottom_right,(input_color_bgr[2],input_color_bgr[1],input_color_bgr[0]),2)

    #Extracting a region of interest where the algorithm will function
    region_of_inter=gray_im[upper_left[1]:bottom_right[1],upper_left[0]:bottom_right[0]]

    #Covnerting the region of interest into an Image object from an array and modifying it to better suit the algorithm
    region_of_inter_im=Image.fromarray(region_of_inter)
    region_of_inter_im=region_of_inter_im.convert('L')
    region_of_inter_im_reszd=region_of_inter_im.resize((33,20),Image.ANTIALIAS)
    region_of_inter_im_reszd_invrtd=PIL.ImageOps.invert(region_of_inter_im_reszd)
    pxl_flter=20
    min_pxl=np.percentile(region_of_inter_im_reszd_invrtd,pxl_flter)
    region_of_inter_im_reszd_invrtd_scld=np.clip(region_of_inter_im_reszd_invrtd-min_pxl,0,255)
    max_pxl=np.max(region_of_inter_im_reszd_invrtd)
    region_of_inter_im_reszd_invrtd_scld=np.asarray(region_of_inter_im_reszd_invrtd_scld)/max_pxl
    im_pred=LR.predict(region_of_inter_im_reszd_invrtd_scld.reshape(1,660))

    #Printing the predicted alphabet
    print(im_pred[0])

    #Showing the frame title "Alphabet Detector" and calling a 'waitKey' method
    cv2.imshow("Alphabet Detector",frame_r)
    cv2.waitKey(1)

#Releasing the video and closing all active ouput windows
video.release()
cv2.destroyAllWindows()    

#-----------------------------------------------------ImagePredictor.py-----------------------------------------------------# 
#----------------------------------------------------------C-123-------------------------------------------------------------------#
