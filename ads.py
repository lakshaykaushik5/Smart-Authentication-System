import cv2
import mediapipe as mp
import time
import face_recognition
import numpy as np
import os


# "videos/test1.mp4"
cap = cv2.VideoCapture(0)
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)


path = "test_images"

images = []
classNames = []
my_list = os.listdir(path)
print(my_list)

for c1 in my_list:
    curImg = cv2.imread(f'{path}/{c1}')
    images.append(curImg)
    classNames.append(os.path.splitext(c1)[0])

print(classNames)


def find_encodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist


encode_list_known = find_encodings(images)
print("encodings_complete")


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = faceDetection.process(imgRGB)
    # print(result)
    encoding_in_curr_frame =face_recognition.face_encodings()

    if result.detections:
        for id, detection in enumerate(result.detections):
            # mpDraw.draw_detection(img, detection)
            # print(id, detection)

            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih),\
                    int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img, bbox, (0, 255, 0), 2)
            cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

    # cTime = time.time()
    # fps = 1/(cTime-pTime)
    # pTime = cTime
    # cv2.putText(img,f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 2)
    # cv2.imshow("record", img)
    cv2.waitKey(1)