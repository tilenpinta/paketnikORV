import os
import numpy as np
import cv2
import _pickle as pickle
import json
import algorithm
import matplotlib.pyplot as plt

baseDir = os.path.dirname(os.path.abspath(__file__)) #pot do direktorija "source"
imageDir = os.path.join(baseDir, "images") # tej poti dodamo še "images", dobimo pot do direktorija "images"
faceCascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt.xml') # za detekcijo sprednjega dela obraza

def trainData(histogram, label):
    if not isinstance(histogram, int): 
        histogram = histogram.tolist()
    else: 
        return
    with open('data.json', 'a') as file:
        data = {}
        data['id'] = label
        data['hist'] = histogram
        json.dump(data, file)
        print("Im here")
        file.write('\n')

def train():
    with open('data.json', 'w') as file:
        file.write("")
    for root, dirs, files in os.walk(imageDir):
        for file in files:
            if file.endswith("jpg") or file.endswith("png") or file.endswith("jpeg"): # sprejeti formati za branje slike so jpeg, jpg, png  
                path = os.path.join(root, file) # dobimo pot do slike 
                label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower() #dobimo direktorij od slike 
                
                imageArray = cv2.imread(path, 0) # preberemo sliko, do katere smo pridobili pot, preberemo kot sivinsko sliko
                
                faces = faceCascade.detectMultiScale(imageArray, scaleFactor=1.5, minNeighbors=5) # iz prebrane slike (imageArray) vzamemo roi(regionOfInterest) oz. v našem primeru obraz osebe

                regionOfInterest = 0
                histogram = 0
                for(x,y,w,h) in faces:
                    regionOfInterest = imageArray[y:y+h, x:x+w]
                    histogram = algorithm.LBPH(regionOfInterest)
                    #plt.plot(histogram)
                trainData(histogram, label)


train()
