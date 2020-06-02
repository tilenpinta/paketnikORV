import numpy as np
import cv2
import _pickle as pickle
import json
import algorithm

DBL_EPSILON = np.finfo(float).eps

faceCascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt.xml') # za detekcijo sprednjega dela obraza, alt.xml se je pokazal kot najboljši (vsaj pri naših testih)
model = algorithm.loadModel()
diff = np.zeros(len(model))
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW) # za zajemanje videa


while(True):
    ret, frame = capture.read()
    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    facesArray = faceCascade.detectMultiScale(grayImage, scaleFactor=1.5, minNeighbors=5) # za detekcijo obraza (6 vrstica v tej skripti)
    x = y = w = h = regionOfInterest = histogram = 0
    for (x, y, w, h) in facesArray:
        regionOfInterest = grayImage[y:y+h, x:x+w]  # y je začetna točka, y+h pa končna točka vertikalne črte. x je začetna točka, x+w pa končna točka horizontalne črte
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 0), 3) # zacetne in končne koordinate x, y ter x+w in y+h
        histogram = algorithm.LBPH(regionOfInterest)
        for i in range(len(model)):
            fml = np.float32(model[i].hist) 
            tmp = cv2.compareHist(fml, histogram, 1)
            le = 0
            lel = 0
            le = len(fml)
            for x in range(0, le):
                lel += fml[x]
            print(lel)
            print("nibba")
            print(tmp)
            #print(tmp, model[i].id)           
 #if(tmp > 0.9):
            diff[i] = tmp
        index = np.argmin(diff)
        if(diff[index] < 15000):
            cv2.putText(frame, model[index].id, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX,  1, (255, 0, 0), 3, cv2.LINE_AA) 
    cv2.imshow('Slika', frame)
    if (cv2.waitKey(20) & 0xFF == ord('q')):
        break

capture.release()
cv2.destroyAllWindows()