# Import necessary libraries
import os
import imutils
import dlib
import cv2
from imutils import face_utils
from notify_run import Notify
from scipy.spatial import distance
from utilities import eye_aspect_ratio, yawning

def helper():

	# To notify the user
	os.system('notify-run register')
	notify = Notify()

	# Eyes and mouth threshold value
	eyeThresh = 0.25
	mouthThresh = 0.60

	# frame to check
	frame_check_eye = 10
	frame_check_mouth = 7

	# Initializing the Face Detector object
	detect = dlib.get_frontal_face_detector()

	# Loading the trained model
	predict = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")

	# Getting the eyes and mouth index
	(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
	(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
	(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

	# Initializing the Video capturing object
	cap=cv2.VideoCapture(0)

	# Initializing the flags for eyes and mouth
	flag_eye=0
	flag_mouth=0

	# Calculating the Euclidean distance between facial landmark points in the Eye Aspect Ratio (EAR)
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

			# Drawing the overlay on the face
			cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
			cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
			cv2.drawContours(frame, [mouth], -1, (255, 0, 0), 1)

			# Comparing threshold value of Mouth Aspect Ratio (MAR)
			if mar > mouthThresh:
				flag_mouth += 1
				if flag_mouth >= frame_check_mouth:
					cv2.putText(frame, "**************** YAWNING ****************", (10, 300),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
					notify.send('Yawning While Driving')
			else:
				flag_mouth = 0


			# Comparing threshold value of Eye Aspect Ratio (EAR)
			if ear < eyeThresh:
				flag_eye += 1
				if flag_eye >= frame_check_eye:
					cv2.putText(frame, "**************** SLEEPING ***************", (10,325),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
					notify.send('Sleeping While Driving')
			else:
				flag_eye = 0
		
		# Plotting the frame
		cv2.imshow("Frame", frame)

		# Waiting for exit key
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break
	
	# Destroying all windows
	cv2.destroyAllWindows()
	cap.stop()

def main():
	helper()

if __name__ == '__main__':
	main()