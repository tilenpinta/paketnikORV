import os
import numpy as np
import cv2
import _pickle as pickle

baseDir = os.path.dirname(os.path.abspath(__file__)) #pot do direktorija "source"
imageDir = os.path.join(baseDir, "images") # tej poti dodamo še "images", dobimo pot do direktorija "images"

faceCascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml') # za detekcijo sprednjega dela obraza
faceRecognizer = cv2.face.LBPHFaceRecognizer_create() # ne smemo uporabit, drugače pa funkcija train() izlušči značilnice, katere s pomočjo save() shranimo v .yml datoteko 
                                                      # na podlagi teh značilnic opravimo razpoznavo 

labelArray = [] # y_labels
labelDictionary = {} #label_ids
trainArray = [] # x_train 
currentId = 0

# sprehajamo se znotraj direktorija images
# in preverjamo vsako datoteko vanj
for root, dirs, files in os.walk(imageDir):
    for file in files:
        if file.endswith("jpg") or file.endswith("png") or file.endswith("jpeg"): # sprejeti formati za branje slike so jpeg, jpg, png  
            path = os.path.join(root, file) # dobimo pot do slike 
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower() #dobimo direktorij od slike 
            
            # če osebo(ime direktorija) še ne poznamo, ga dodamo v slovar 
            if(not label in labelDictionary):         
                labelDictionary[label] = currentId
                currentId += 1
                  
            id_ = labelDictionary[label] # v id_ shranimo ime direktorija(osebe)
        
            imageArray = cv2.imread(path, 0) # preberemo sliko, do katere smo pridobili pot
            
            faces = faceCascade.detectMultiScale(imageArray, scaleFactor=1.5, minNeighbors=5) # iz prebrane slike (imageArray) vzamemo roi(regionOfInterest) oz. v našem primeru obraz osebe
            
            for(x,y,w,h) in faces:
                regionOfInterest = imageArray[y:y+h, x:x+w]
                trainArray.append(regionOfInterest) # polje regionOfInteres (obraz), dodamo v polje za učenje modela
                labelArray.append(id_) # dodamo še id direktorija(osebe)

with open("labels.pickle", 'wb') as file:
    pickle.dump(labelDictionary, file) #shranimo slovar, ki vsebuje id in ime direktorija(osebe)


faceRecognizer.train(trainArray, np.array(labelArray)) # učimo naš model
faceRecognizer.save("trainner.yml") #shranimo, da lahko potem v main.py preberemo