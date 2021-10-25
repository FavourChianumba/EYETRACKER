import cv2
import numpy as np
import dlib

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
detector = dlib.get_frontal_face_detector()


def eye(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)
    return


def eye_region(eye_points, facial_landmarks):
    region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                       (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                       (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                       (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                       (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                       (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
    cv2.polylines(frame, [region], True, (0, 0, 255), 2)
    return


predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def midpoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)


while True:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(frame)
    for face in faces:

        landmarks = predictor(frame, face)
        left_eye = eye([36, 37, 38, 39, 40, 41], landmarks)
        right_eye = eye([42, 43, 44, 45, 46, 47], landmarks)

        #gaze detection
        left_eye_region = eye_region([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_region = eye_region([42, 43, 44, 45, 46, 47], landmarks)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(30)
    if key == 27:
        break

cv2.destroyAllWindows()
