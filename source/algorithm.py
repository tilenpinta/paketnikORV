import numpy as np 
import cv2
import json
import math
from decimal import Decimal


DBL_EPSILON = np.finfo(float).eps

class svmData:
    def __init__(self, id, hist):
        self.id = id
        self.hist = hist

def resizeImage(img):
    scale_percent = 60 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
 
def localBinaryPattern(image):
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

            # shranimo samo indekse bitov, kjer je vrednost 1
            vector = np.where(result2)[0]

            # pretvorimo v decimalno število
            num = np.sum(2**vector)

            # popravimo vrednost centralnega piksla
            imgLBP[i+1, j+1] = num
    return imgLBP

def calcHist(img):
    height = img.shape[0]
    width = img.shape[1]
    hist = np.zeros((256, 1), dtype=np.float32)
    for i in range(0, height):
        for j in range(0, width):
            hist[img[i, j]] = hist[img[i, j]] + 1
    return hist

def LBPH(img):
    imgLbp = localBinaryPattern(img)
    lel = calcHist(imgLbp)
    #histogram = cv2.calcHist([imgLbp], [0], None, [256], [0, 256])
    return lel

def loadModel(): 
    svm = []
    name = 'data.json'
    for line in open(name, mode='r'):
        tmp = json.loads(line)
        svm.append(svmData(tmp["id"], tmp["hist"]))
    return svm

def compareHist(hist1, hist2):
    a = 0.0
    b = 0.0
    result = 0.0
    his1 = 0.0
    his2 = 0.0
    for i in range(0, len(hist1)):
        his1 = hist1[i].astype(float)
        his2 = hist2[i].astype(float)
        a = his1 - his2
        b = his1
        if math.fabs(b) > DBL_EPSILON:
            result += a*a/b
    
    return result[0]

    