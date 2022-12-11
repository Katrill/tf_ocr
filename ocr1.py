import numpy
import cv2
import pytesseract
from imutils import contours
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


image = cv2.imread("passport/2.jpeg")
print('image.shape', image.shape)
height, width, _ = image.shape
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# print("gray", gray)
# cv2.imshow("gray", gray)
#значение пикселя, > 0, меняется на 255, а  которое < 215, устанавливается равным нулю.
# последний параметр передаёт  пороговое значение: значения пикселей, которые > 0 устанавливаются в максимальное значение, которое передается 3им параметром.
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]
# thresh = cv2.threshold(gray, 100, 255, 0)[1]  # 5.jpeg
# thresh = cv2.threshold(gray, 170, 255, 0)[1] # 4.png
# print("thresh", thresh)
# cv2.imshow("thresh", thresh)
# cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
boxes = pytesseract.image_to_boxes(thresh, lang="eng+rus", config='--psm 3')
d = pytesseract.image_to_data(thresh, lang="eng+rus", config='--psm 3')
print(d)
lines = d.splitlines()
n = 0
for i in range(1, len(lines)):
    while n != 3:
        if len(lines[i].split()) == 12 and int(lines[i].split()[7]) > 0.5*height:
            n += 1
            print(lines[i].split()[-1])
        break

# for i, d in enumerate(d.splitlines()):
#     if i != 0:
#         r = d.split()
#         j = i
#         print(r)
#         if len(r) == 12 and 0.5*height < int(r[7]) < 0.75*height:
#             x = int(r[6])
#             y = int(r[7])
#             w = int(r[8])
#             h = int(r[9])
#             # cv2.rectangle(thresh, (x,y), (x+w,y+h), (0,0,255),1)
#             # изображение,верхний левый угол (x1, y1), нижний правый угол (x2, y2), Цвет прямоуг.(GBR/RGB), Толщина линии
#             # if 0.5*height < y < 0.75*height:
#             print(r[11])

for box in boxes.splitlines():
   box = box.split(" ")
   thresh = cv2.rectangle(thresh, (int(box[1]), height - int(box[2])), (int(box[3]), height - int(box[4])), (0, 0, 255), 1)
cv2.imshow('thresh', thresh)
cv2.waitKey()
# cnts, _ = contours.sort_contours(cnts[0])
# print(cnts)

# for c in cnts:
#     area = cv2.contourArea(c)
#     x, y, w, h = cv2.boundingRect(c)
#     if y < height / 2:
#         # print('c:',c)
#         img = image[y:y + h, x:x + w]
        # result = pytesseract.image_to_string (img, lang="eng+rus", config='--psm 3')
        # print(result)
   # if area > 1000:
    #    img = image[y:y+h, x:x+w]
     #   result = pytesseract.image_to_string(img, lang = "eng+rus", config='--psm 3')
      #  print(result)
#result = pytesseract.image_to_string(thresh, lang = "rus+eng", config='--psm 1')
#print(result)
# cv2.imshow("Test", thresh)
cv2.waitKey()
