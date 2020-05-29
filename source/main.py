import numpy as np
import cv2

faceCascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

capture = cv2.VideoCapture(0) # za zajemanje videa 

while(True):
    retval, image = capture.read() #zajami vsako sliko (frame)
    grayImage = cv2.cvtColor(image, 0)
    facesArray = faceCascade.detectMultiScale(grayImage, scaleFactor=1.5, minNeighbors=5)

    for (x, y, w, h) in facesArray:
        print(x,y,w,h)                   
        regionOfInterestGray = grayImage[y:y+h, x:x+w]
        regionOfInterestBGR = image[y:y+h, x:x+w]
        
        
        
        imgItem = "ivan.png"
        cv2.imwrite(imgItem, regionOfInterestGray)

        # s tem izrišemo pravokotnik znotraj katerega je ROI
        cv2.rectangle(image, (x,y), (x+w, y+h), (255, 255, 255), 3) # zacetne in koncne koordinate x, y ter x+w in y+h

    cv2.imshow('Slika', image)
    if (cv2.waitKey(27) & 0xFF == ord('q')):
        break

capture.release()
cv2.destroyAllWindows()


#print(cv2.__file__)