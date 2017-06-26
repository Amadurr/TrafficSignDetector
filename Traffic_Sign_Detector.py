import cv2
import numpy as np

roundSigns_cascade = cv2.CascadeClassifier('RoundTraffic_HaarCascade_BU.xml')
#specificSign_cascade = cv2.CascadeClassifier('SpecificSign_HaarCascade.xml')
#stopSign_cascade = cv2.CascadeClassifier('StopTraffic_HaarCascade.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    roundSigns = roundSigns_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in roundSigns:

        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)


#        roi_gray = gray[y:y+h, x:x+w]
#        roi_color = img[y:y+h, x:x+w]
#        specificSign = specificSign_cascade.detectMultiScale(roi_gray)
#        for (ex,ey,ew,eh) in specificSign:
#            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)

    cv2.imshow('img' ,img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()