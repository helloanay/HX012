from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2

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


eyeThresh = 0.25
mouthThresh = 0.60
frame_check_eye = 20
frame_check_mouth = 10
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("/media/rowhit/01D4B52C49FF3EC0/Data Science/Drowsiness_Detection/shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]
cap=cv2.VideoCapture(0)
flag_eye=0
flag_mouth=0
while True:
	ret, frame=cap.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	subjects = detect(gray, 0)
	for subject in subjects:
		shape = predict(gray, subject)
		shape = face_utils.shape_to_np(shape)#converting to NumPy Array
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		mouth = shape[mStart:mEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		ear = (leftEAR + rightEAR) / 2.0
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		mar = yawning(mouth)
		mouthHull = cv2.convexHull(mouth)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [mouth], -1, (0, 255, 0), 1)
		#cv2.putText(frame, "EAR: {}".format(ear), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
		#cv2.putText(frame, "MAR: {}".format(mar), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

		if mar > mouthThresh:
			flag_mouth += 1
			if flag_mouth >= frame_check_mouth:
				cv2.putText(frame, "**************** YAWNING ****************", (10, 300),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		else:
			flag_mouth = 0

		if ear < eyeThresh:
			flag_eye += 1
			if flag_eye >= frame_check_eye:
				cv2.putText(frame, "**************** SLEEPING ***************", (10,325),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		else:
			flag_eye = 0


	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
cv2.destroyAllWindows()
cap.stop()