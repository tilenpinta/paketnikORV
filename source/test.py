from skimage import feature
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import numpy as np
import cv2


def resizeImage(image, defaultWidth=360): 
    height = image.shape[0]
    width = image.shape[1]
    ratio = defaultWidth / width
    newHeight = (int(height*ratio))
    dimensions = (defaultWidth, newHeight)
    return cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA), (newHeight//10)


def lbp(image, neighbours, radius):
    normalize=0.0000001
    lbp = feature.local_binary_pattern(image, neighbours, radius, method= "uniform" )
    (hist, _) = np.histogram(lbp.reshape(-1),
            bins=np.arange(0, neighbours + 3),
            range=(0, neighbours + 2))

    hist = hist.astype("float")
    hist /=(hist.sum() + normalize)
    return hist 

def getHistogramAndLabel(image, label):
    divider = 10
    histograms = []
    labels = []
    height = image.shape[0]
    width = image.shape[1]
    cellHeight = height // divider
    cellWidth = width // divider

    

    for i in range(divider):
        y = cellHeight * i
        for j in range(divider):
            x = cellWidth * j
            
            regionOfInterest = image[y: cellHeight*(i+1), x:cellWidth * (j+1)]

            regionOfInterest = cv2.cvtColor(regionOfInterest, cv2.COLOR_BGR2GRAY)
            histogram = lbp(regionOfInterest, 10, 5)

            histograms.append(histogram)
            labels.append(label)
 
    return histograms, labels

def compute(image):
        winSize = (128, 128)
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (8, 8)
        nbins = 9
        derivAperture = 1
        winSigma = 4.
        histogramNormType = 0
        L2HysThreshold = 0.4
        gammaCorrection = 0
        nlevels = 64

        hog_desc = cv2.HOGDescriptor(winSize,
                                    blockSize,
                                    blockStride,
                                    cellSize,
                                    nbins,
                                    derivAperture,
                                    winSigma,
                                    histogramNormType,
                                    L2HysThreshold,
                                    gammaCorrection,
                                    nlevels)
        return hog_desc.compute(image)
       # feat = [item for sublist in desc for item in sublist]
        #return feat 
    
image = cv2.imread('image.png')


his, _ = getHistogramAndLabel(image, 'ivan')
print(len(his[0]))
#lbpHist = lbp(image, 10, 5)
        # Normalize the histogram
#hist = hist.astype("float")
#hist /= (hist.sum() + eps)
#print(lbp.shape[0])
#eps=1e-7
