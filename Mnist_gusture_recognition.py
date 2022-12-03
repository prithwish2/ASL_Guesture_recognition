import cv2
import numpy as np
import tensorflow as tf
import os
#import pandas as pd

import pyttsx3
from keras.models import load_model
from keras.models import model_from_json

# Model reconstruction from JSON file
with open(r'C:\Users\Prithwish\Desktop\ASL_Guesture_recognition\Model_Architecture_MNIST_Guesture_Recognition.json', 'r') as f:
    model = model_from_json(f.read())

# Load weights into the new model
model.load_weights(r'C:\Users\Prithwish\Desktop\ASL_Guesture_recognition\MNIST_Guesture_Recognition.h5')

model.summary()
#data_dir = (r'C:\Users\Prithwish\Desktop\ASL_Guesture_recognition\Dataset\sign_mnist_train.csv')
#df=pd.read_csv(r'C:\Users\Prithwish\Desktop\ASL_Guesture_recognition\Dataset\sign_mnist_train.csv')
#df.head()
#getting the labels form data directory
#labels = sorted(os.listdir(r'C:\Users\Prithwish\Desktop\ASL_Guesture_recognition\Dataset\sign_mnist_train.csv'))
#labels[-1] = 'Nothing'
#print(labels)

#initiating the video source, 0 for internal camera
cap = cv2.VideoCapture(0)

while(True):
    
    _ , frame = cap.read()
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 0, 255), 5)
    #cv2.imshow() 
    #region of intrest
    roi = frame[100:300, 100:300]
    img = cv2.resize(roi, (50, 50))
    #cv2.imshow('roi', roi)
    
    

    img = img/255

    #make predication about the current frame
    prediction = model.predict(img.reshape(28,28,1))
    char_index = np.argmax(prediction)
    

    #confidence = round(prediction[0,char_index]*100, 1)
    #predicted_char = labels[char_index]

  

    font = cv2.FONT_HERSHEY_TRIPLEX
    fontScale = 1
    color = (0,255,255)
    thickness = 2

    #writing the predicted char and its confidence percentage to the frame
    #msg = predicted_char +', Conf: ' +str(confidence)+' %'
    #cv2.putText(frame, msg, (80, 80), font, fontScale, color, thickness)
    
    cv2.imshow('frame',frame)
    
    #close the camera when press 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
        
#release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
