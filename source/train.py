import os
import numpy as np
import cv2
import _pickle as pickle

baseDir = os.path.dirname(os.path.abspath(__file__))
imageDir = os.path.join(baseDir, "images")
faceCascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
faceRecognizer = cv2.face.LBPHFaceRecognizer_create()

labelArray = [] # y_labels
labelDictionary = {} #label_ids
trainArray = [] # x_train 
currentId = 0

# sprehajamo se znotraj zbirke images
# in preverjamo vsako datoteko vanjo
for root, dirs, files in os.walk(imageDir):
    for file in files:
        if file.endswith("jpg") or file.endswith("jpg"):
            path = os.path.join(root, file) # dobimo pot do slike, ki ima končnico .jpg ali .png
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower() #dobimo direktorij od slike 
            
            # label za vsako osebo, ki jo poznamo 
            if(not label in labelDictionary):         
                labelDictionary[label] = currentId
                currentId += 1
                  
            id_ = labelDictionary[label]
        
            imageArray = cv2.imread(path, 0)
            
            faces = faceCascade.detectMultiScale(imageArray, scaleFactor=1.5, minNeighbors=5)
            
            for(x,y,w,h) in faces:
                regionOfInterest = imageArray[y:y+h, x:x+w]
                trainArray.append(regionOfInterest)
                labelArray.append(id_)

with open("labels.pickle", 'wb') as file:
    pickle.dump(labelDictionary, file)


faceRecognizer.train(trainArray, np.array(labelArray))
faceRecognizer.save("trainner.yml")