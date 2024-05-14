from tkinter import *
from threading import Thread, Event
from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

mixer.init()
mixer.music.load("music.wav")

def start_detection():
    global flag
    while not stop_event.is_set():
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)
        for subject in subjects:
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            if ear < thresh:
                flag += 1
                print(flag)
                if flag >= frame_check:
                    cv2.putText(frame, "****************ALERT!****************", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "****************ALERT!****************", (10, 325),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    mixer.music.play()
            else:
                flag = 0
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    cap.release() 

def start_detection_thread():
    global detection_thread
    global stop_event
    stop_event = Event()
    detection_thread = Thread(target=start_detection)
    detection_thread.start()

def stop_detection_thread():
    global stop_event
    if stop_event.is_set():
        return
    stop_event.set()
    detection_thread.join()

def start_detection_gui():
    start_button.config(state=DISABLED)
    stop_button.config(state=NORMAL)
    start_detection_thread()

def stop_detection_gui():
    stop_button.config(state=DISABLED)
    start_button.config(state=NORMAL)
    stop_detection_thread()

root = Tk()
root.title("Eye Drowsiness Detection")
root.geometry("300x100")

start_button = Button(root, text="Start Detection", command=start_detection_gui)
start_button.pack(pady=5)

stop_button = Button(root, text="Stop Detection", command=stop_detection_gui, state=DISABLED)
stop_button.pack(pady=5)

cap = cv2.VideoCapture(0)
flag = 0

root.mainloop()
