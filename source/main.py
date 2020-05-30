import numpy as np
import cv2
import _pickle as pickle
import requests

#loll
faceCascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
faceRecognizer.read("trainner.yml")


labelDictionary = {}

with open("labels.pickle", 'rb') as file:
    tmpLabels = pickle.load(file)
    labelDictionary = {v:k for k,v  in tmpLabels.items()} # obrnemo

capture = cv2.VideoCapture(0) # za zajemanje videa 

while(True):
    #images = requests.get("http://192.168.0.11:8080/shot.jpg")
    #video = np.array(bytearray(images.content), dtype=np.uint8)
    #render = cv2.imdecode(video, -1)
    ret, frame = capture.read()
    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    facesArray = faceCascade.detectMultiScale(grayImage, scaleFactor=1.5, minNeighbors=5)

    for (x, y, w, h) in facesArray:
        #print(x,y,w,h)                   
        regionOfInterestGray = grayImage[y:y+h, x:x+w]
        regionOfInterestBGR = frame[y:y+h, x:x+w] 
        
        id_, conf = faceRecognizer.predict(regionOfInterestGray)
        
        if (conf > 100):
            print(labelDictionary[id_])

       # imgItem = "ivan.png"
        #cv2.imwrite(imgItem, regionOfInterestGray)

        # s tem izrišemo pravokotnik znotraj katerega je ROI
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 255, 255), 3) # zacetne in koncne koordinate x, y ter x+w in y+h

    cv2.imshow('Slika', frame)
    if (cv2.waitKey(27) & 0xFF == ord('q')):
        break

capture.release()
cv2.destroyAllWindows()


#print(cv2.__file__)