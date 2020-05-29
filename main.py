import numpy as np
import cv2
'''
capture = cv2.VideoCapture(0) # za zajemanje videa 

while(True):
    retval, image = capture.read() #zajami vsako sliko (frame)

    cv2.imshow('Slika', image)
    if (cv2.waitKey(27) & 0xFF == ord('q')):
        break

capture.release()
cv2.destroyAllWindows()
'''

print(cv2.__file__)