## Project Name:  Accident Prevention using OpenCV and Dlib


![](https://forthebadge.com/images/badges/made-with-python.svg)

## Short Project Description: Accident Prevention by notifying on the smartphone about any mishappenings. 

## Team ID: HX012

## Team Name: Hack_Creators

## Team Members
- Smiti Khurana ([@smiti-123](https://github.com/smiti-123))
- Rohit Swami ([@rowhitswami](https://github.com/rowhitswami))
- Mohd Shoaib Rayeen([@shoaibrayeen](https://github.com/shoaibrayeen))

### [Repository Link](https://github.com/rowhitswami/HX012)

### Labels: Influence the Mass, Revolutionize the Smart-World and Enhance the Social-Norms

### Scope
-	
-
-
- 


## Use Case for The Project
<details>
<summary>Use Case 1</summary>

```
	When It detects the face and eyes of the person, it does not notify to anyone.
```

##
<img src="/Image/use-case-1.png">
</details>	

<details>
<summary>Use Case 2</summary>

```	
	When It detects the person yawning while driving, it notifies to everyone who gets access for the same.
```

##
<img src="/Image/use-case-2.png">)
##
<img src="/Image/Notify-2.png" width="400" height="650">)
</details>

<details>
<summary>Use Case 3</summary>
	
	When It detects the person sleeping while driving, it notifies to everyone who gets access for the same.

##
<img src="/Image/use-case-3.png">)
##
<img src="/Image/Notify-1.png" width="400" height="650">)
</details>

<details>
<summary>Use Case 4</summary>
	
	When It detects the person both sleeping and yawning while driving, it notifies to everyone who gets access for the same.

##
<img src="/Image/use-case-4.png">)
##
<img src="/Image/Notify-2.png" width="400" height="650">)
</details>



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

### To Get Started with the Project

Clone the Repository
``` python
>>>  git clone https://github.com/rowhitswami/HX012/
```
Install Required Libraries
``` python
>>>  pip install -r requirements.txt
```
  
Go to src folder
``` python
>>>  cd src
```
Execute app file
``` python
>>>  python app.py
```
