import numpy as np
import cv2
import _pickle as pickle
import json
from scipy.spatial import procrustes

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

def compare(modelHistogram, currentHistogram):
    _, _, disparity = procrustes(modelHistogram, currentHistogram)
    return disparity

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

faceCascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt.xml') # za detekcijo sprednjega dela obraza, alt.xml se je pokazal kot najboljši (vsaj pri naših testih)

model = loadModel()
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW) # za zajemanje videa 
print(model[1].hist)
   
counterHis = 0
counter = 0
while(True):
    ret, frame = capture.read()
    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    facesArray = faceCascade.detectMultiScale(grayImage, scaleFactor=1.5, minNeighbors=5) # za detekcijo obraza (6 vrstica v tej skripti)
    x = y = w = h = regionOfInterest = histogram = 0
    for (x, y, w, h) in facesArray:
        regionOfInterest = grayImage[y:y+h, x:x+w]  # y je začetna točka, y+h pa končna točka vertikalne črte. x je začetna točka, x+w pa končna točka horizontalne črte
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 0), 3) # zacetne in končne koordinate x, y ter x+w in y+h
        histogram = LBPH(regionOfInterest)
        for i in range(len(model)):
            #print("His", histogram)
            fml = np.float32(model[i].hist) 
            diff = cv2.compareHist(fml, histogram, 1) #moralo bi bit 
            print("Ime", model[i].id)
            print(i, "Diff: ", diff)
    cv2.imshow('Slika', frame)
    if (cv2.waitKey(20) & 0xFF == ord('q')):
        break

capture.release()
cv2.destroyAllWindows()