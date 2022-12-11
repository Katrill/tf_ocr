import numpy as np
import cv2
import pytesseract
from imutils import contours
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def viewImage(image, name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_window, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image = cv2.imread("passport/0.jpeg")
# print('image.shape', image.shape)
height, width, _ = image.shape
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("gray", gray)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# cv2.imshow("thresh", thresh)
boxes = pytesseract.image_to_boxes(image, lang="rus", config='--psm 4 ')
# d = pytesseract.image_to_data(thresh, lang="eng+rus", config='--psm 6')
#print(boxes)

# box_coord = []
# for box in boxes.splitlines():
#     box = box.split(" ")
#     x, y, w, h = int(box[1]), int(box[2]), int(box[3]), int(box[4])
#     image_box = cv2.rectangle(thresh, (x, height - y), (w, height - h), (50, 50, 255), 1)
#     cv2.putText(thresh, box[0], (x, height - y + 13), cv2.FONT_HERSHEY_COMPLEX, 0.4, (50, 205, 50), 1)
#     if (0.3*height < y < 0.45*height):
#             # and (x > 0.2*width):
#         box_coord.append([w - x, h - y, box[0]])

# for i in box_coord:
#     print(i)
# cv2.imshow('thresh', thresh)

# Face recognition
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# image = cv2.imread(image_path)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor= 1.1,
    minNeighbors= 5,
    minSize=(10, 10)
)


faces_detected = format(len(faces)) + " faces detected!"
# print(faces_detected)
# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)
    xc, yc, width, height = x, y, x+w, y+h

# print(xc, yc, width, height)
# viewImage(image, faces_detected)
cv2.imshow('thresh', image)

# len(lines[i].split()) == 12
# int(lines[i].split()[6]) > x + 2*width
# y + 0.5*height< int(lines[i].split()[7]) < y + 2.25*height
#
#

# Tesseract reading
d = pytesseract.image_to_data(gray, lang="rus", config='--psm 3')
lines = d.splitlines()
# for i in range(1, len(lines)):
#     if len(lines[i].split()) == 12 and (10 < int(lines[i].split()[9])):
#         print(lines[i].split())
#
# print()
for i in range(1, len(lines)):
    if len(lines[i].split()) == 12:
        if int(lines[i].split()[6]) > (xc + 1.0*width):
            if (yc + 0.05*height) > int(lines[i].split()[7]) > (yc - 0.5*height):
                print(lines[i].split()[-1])


cv2.waitKey(0)


