import numpy as np
import cv2
import _pickle as pickle
import requests

def lbp(image):
    height = image.shape[0]
    width = image.shape[1]
    imgLBP = np.zeros(shape=(height, width), dtype="uint8")

    maskSize = 3
    for i in range(0, height - maskSize):
        for j in range(0, width - maskSize):
            img = image[i:i+maskSize, j:j+maskSize]

            result = (img >= img[1, 1])*1
            result2 = np.zeros(8)

            result2[0] = result[0, 0]
            result2[1] = result[0, 1]
            result2[2] = result[0, 2]
            result2[3] = result[1, 2]
            result2[4] = result[2, 2]
            result2[5] = result[2, 1]
            result2[6] = result[2, 0]
            result2[7] = result[1, 0]

            vector = np.where(result2)[0]

            num = np.sum(2**vector)

            imgLBP[i+1, j+1] = num
    return imgLBP

def LBPd(d):
    LPBd1

    for y in range(0, height - 1):
        for x in range(0, width - 1):
            byte = []
            upL = grayImg[y-1, x-1]
            upR = grayImg[y - 1, x + 1]
            left = grayImg[y, x - 1]
            right = grayImg[y, x + 1]
            downL = grayImg[y + 1, x - 1]
            down = grayImg[y + 1, x]
            downR = grayImg[y + 1, x + 1]
            arr = [upL, up, upR, left, right, downL, down, downR]

            num = 0

            for i in arr:
                primerjaj = num+d

                if primerjaj > 7:
                    primerjaj = (primerjaj - 7) - 1

                if i >= arr[primerjaj]:
                    byte.append(1)
                else:
                    byte.append(0)
                num =+ 1

            rezultat = 0

            for i in range(0, 8):
                rezultat += utezi[i] * byte[i]

            LBPd1[y, x] = rezultat
    return LBPd1


faceCascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml') # za detekcijo sprednjega dela obraza
faceRecognizer = cv2.face.LBPHFaceRecognizer_create() # tega ne smemo uporabiti
faceRecognizer.read("trainner.yml") # ne smemo uporabiti, drugače pa read() prebere naš model, katerega smo naučili na podlagi slik (znotraj train.py skripte)


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
        lbp = lbp(faceRecognizer.read()) #read da dobiš sliko
        cv2.imshow("LBP", lbp)

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