import numpy
import cv2
# open camare
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
mouth_cascade = cv2.CascadeClassifier('mouth.xml')
while(True):
    r,f = cap.read()
      
    gray = cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
    # face detection
    faces = face_cascade.detectMultiScale(gray,1.3,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(f,(x,y),(x+w,y+h),(0,255,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = f[y:y+h, x:x+w]
        #smile detection
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        for (sx, sy, sw, sh) in smiles: 
            cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (255,0,0), 1)
        # eye detection
        eyes= eye_cascade.detectMultiScale(roi_gray)
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),1)
    # show 
    cv2.imshow('Detection Program',f)
    # when user click "q" exit 
    if cv2.waitKey(1) & 0xFF == ord('q'):
          break
cap.release()
cv2.destroyAllWindows()
