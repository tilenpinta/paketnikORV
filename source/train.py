import os
import numpy as np
import cv2
import _pickle as pickle
import inspect
import json

baseDir = os.path.dirname(os.path.abspath(__file__)) #pot do direktorija "source"
imageDir = os.path.join(baseDir, "images") # tej poti dodamo še "images", dobimo pot do direktorija "images"
faceCascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt.xml') # za detekcijo sprednjega dela obraza
#faceRecognizer = cv2.face.LBPHFaceRecognizer_create() # ne smemo uporabit, drugače pa funkcija train() izlušči značilnice, katere s pomočjo save() shranimo v .yml datoteko 
                                                      # na podlagi teh značilnic opravimo razpoznavo 

class svmData:
    def __init__(self, id, hist):
        self.id = id
        self.hist = hist

def localBinaryPattern(image):
    height = image.shape[0]
    width = image.shape[1]
    imgLBP = np.zeros(shape=(height, width), dtype="uint8")

    maskSize = 3
    for i in range(0, height - maskSize):
        for j in range(0, width - maskSize):
            img = image[i:i+maskSize, j:j+maskSize]

            # če je vrednost večja ali enaka centralnemu pikslu, zapiši ena, sicer nič
            # nato metoda flatten() pretvori polje iz 2D v 1D
            result = (img >= img[1, 1])*1
            result2 = np.zeros(8)

            # v smeri urinega kazalca
            result2[0] = result[0, 0]
            result2[1] = result[0, 1]
            result2[2] = result[0, 2]
            result2[3] = result[1, 2]
            result2[4] = result[2, 2]
            result2[5] = result[2, 1]
            result2[6] = result[2, 0]
            result2[7] = result[1, 0]

            # shranimo samo indekse bitov, kjer je vrednost 1
            vector = np.where(result2)[0]

            # pretvorimo v decimalno število
            num = np.sum(2**vector)

            # popravimo vrednost centralnega piksla
            imgLBP[i+1, j+1] = num
    return imgLBP

def LBPH(img):
    imgLbp = localBinaryPattern(img)
    histogram = cv2.calcHist([imgLbp], [0], None, [256], [0, 256])
    return histogram  

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
                    histogram = LBPH(regionOfInterest)
                trainData(histogram, label)

train()
