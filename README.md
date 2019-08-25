## Project Name:  


![](https://forthebadge.com/images/badges/made-with-python.svg)

## Short Project Description:

## Team ID: HX012

## Team Name: Hack_Creators

## Team Members
- Smiti Khurana ([@smiti-123](https://github.com/smiti-123))
- Rohit Swami ([@rowhitswami](https://github.com/rowhitswami))
- Mohd Shoaib Rayeen([@shoaibrayeen](https://github.com/shoaibrayeen))

### [Demo Link]()

### [Repository Link](https://github.com/rowhitswami/HX012)

### Labels: 


### Required Header Files
```py
import os
import imutils
import dlib
import cv2
from imutils import face_utils
from notify_run import Notify
from scipy.spatial import distance
from utilities import eye_aspect_ratio, yawning
```

### Utilities to Calculate Euclidean Distance For Face Aspect Ratio and Mouth Aspect Ratio

```py
def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

def yawning(mouth):
	A = distance.euclidean(mouth[3], mouth[9])
	B = distance.euclidean(mouth[2], mouth[10])
	C = distance.euclidean(mouth[4], mouth[8])
	L = (A+B+C)/3
	D = distance.euclidean(mouth[0], mouth[6])
	mar=L/D
	return mar
```

### To Run the Project
```
  Clone the Repository
<$  git clone https://github.com/rowhitswami/HX012/

  Install Required Libraries
<$  pip install -r requirements.txt
  
  Go to src folder
<$  cd src

  Execute app file
<$  python app.py
```
