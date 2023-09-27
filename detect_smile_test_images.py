from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
from imutils import paths

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-m", "--model", required=True,
	help="path to pre-trained smile detector CNN")
ap.add_argument("-i", "--images",required=True,
	help="path to the image file")
args = vars(ap.parse_args())

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model(args["model"])


for imagePath in paths.list_images(args["images"]):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects= faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,      
        minSize=(30, 30)
    )
    for(fX,fY,fW,fH) in rects:
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (28, 28))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        (notSmiling, smiling) = model.predict(roi)[0]
        label = "Smiling" if smiling > notSmiling else "Not Smiling"
        cv2.putText(image, label, (fX, fY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(image, (fX, fY), (fX + fW, fY + fH),(0, 0, 255), 2)
    cv2.imshow("Face", image)
    k = cv2.waitKey(0) & 0xff
    

       
        
        
        
    