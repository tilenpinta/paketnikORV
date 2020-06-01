import numpy as np
import cv2
import _pickle as pickle
import json


class svmData:
    def __init__(self, id, hist):
        self.id = id
        self.hist = hist

def loadModel(): 
    svm = []
    name = 'data.json'
    for line in open(name, mode='r'):
        tmp = json.loads(line)
        svm.append(svmData(tmp["id"], tmp["hist"]))
    return svm


faceCascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml') # za detekcijo sprednjega dela obraza
faceRecognizer = cv2.face.LBPHFaceRecognizer_create() # tega ne smemo uporabiti




labelDictionary = {}  # direktorije labeliramo, ta labela je slovar (HashMap, Dictionary), ki je sestavljen iz id(ključ) na katerem se nahaja ime direktorija
                      #(v našem primeru so imena direktorij pač imena oseb v source/images)

with open("labels.pickle", 'rb') as file:
    tmpLabels = pickle.load(file)
    labelDictionary = {v:k for k,v  in tmpLabels.items()} # napolnimo slovar

capture = cv2.VideoCapture(0, cv2.CAP_DSHOW) # za zajemanje videa 

while(True):
    ret, frame = capture.read()
    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    facesArray = faceCascade.detectMultiScale(grayImage, scaleFactor=1.5, minNeighbors=5) # za detekcijo obraza (6 vrstica v tej skripti)

    for (x, y, w, h) in facesArray:
        """
        lbp = lbp(grayImage)
        cv2.imshow("LBP", lbp)
        """

        regionOfInterestGray = grayImage[y:y+h, x:x+w]  # y je začetna točka, y+h pa končna točka vertikalne črte. x je začetna točka, x+w pa končna točka horizontalne črte
        
        id_, conf = faceRecognizer.predict(regionOfInterestGray) # ne smemo uporabiti, moramo implementirati
        
        if (conf > 15 and conf <50):     # če je ocena v redu (med 15 in 50), potem sprejmi id kot veljaven (id je ključ, s katerim najdemo ime osebe v labelDictionary)
            print(labelDictionary[id_])

        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 255, 255), 3) # zacetne in končne koordinate x, y ter x+w in y+h

    cv2.imshow('Slika', frame)
    if (cv2.waitKey(20) & 0xFF == ord('q')):
        break

capture.release()
cv2.destroyAllWindows()