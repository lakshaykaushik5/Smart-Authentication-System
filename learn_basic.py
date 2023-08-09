import os

import cv2
import mediapipe as mp
import time
import face_recognition
import numpy as np


# # "videos/test1.mp4"
# cap = cv2.VideoCapture(0)
# pTime = 0
#
# mpFaceDetection = mp.solutions.face_detection
# mpDraw = mp.solutions.drawing_utils
# faceDetection = mpFaceDetection.FaceDetection(0.75)
#
# while True:
#     success, img = cap.read()
#     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     result = faceDetection.process(imgRGB)
#     # print(result)
#
#     if result.detections:
#         for id, detection in enumerate(result.detections):
#             # mpDraw.draw_detection(img, detection)
#             # print(id, detection)
#
#             bboxC = detection.location_data.relative_bounding_box
#             ih, iw, ic = img.shape
#             bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih),\
#                     int(bboxC.width * iw), int(bboxC.height * ih)
#             cv2.rectangle(img, bbox, (0, 255, 0), 2)
#             cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
#
#     cTime = time.time()
#     fps = 1/(cTime-pTime)
#     pTime = cTime
#     cv2.putText(img,f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 2)
#     cv2.imshow("record", img)
#     cv2.waitKey(1)

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


class Face_Detector():
    def __init__(self, minDetect = 0.5):
        self.minDetect = minDetect
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(0.75)

    def findFaces(self,img,draw=False):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.faceDetection.process(imgRGB)
        # print(result)
        bboxss = []
        if self.result.detections:
            for id, detection in enumerate(self.result.detections):
                # mpDraw.draw_detection(img, detection)
                # print(id, detection)

                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                # bboxss.append([id, bbox, detection.score])
                bboxss.append(bbox)
                cv2.rectangle(img, bbox, (0, 255, 0), 2)
                cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                            3, (255, 0, 0), 2)
        return img, bboxss


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = Face_Detector()
    while True:
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS, bboxss = detector.findFaces(imgS)
        print(bboxss)
        face_in_curr_frame = face_recognition.face_encodings(img, bboxss)
        for encode in face_in_curr_frame:
            matches = face_recognition.compare_faces(encode_list_known, encode)
            facedistance = face_recognition.face_distance(encode_list_known, encode)


            matchIndex = np.argmin(facedistance)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                y1, x2, y2, x1 = bboxss[2], bboxss[1], bboxss[3], bboxss[0]
                y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
                # cv2.rectangle(img,bboxss,(0,255,0),2)
                cv2.rectangle(img,(x1, y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                cv2.putText(img,name,(x1+6, y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                # cv2.putText(img, f'{name}', (bboxss[0], bboxss[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                #             3, (255, 0, 0), 2)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv2.imshow("record", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()