from imutils.perspective import four_point_transform
import pytesseract
import argparse
import imutils
import cv2
import re

# convert the image to grayscale, blur it, and apply edge detection
# to reveal the outline of the business card
image = cv2.imread("passport/2.jpeg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 30, 150)


# detect contours in the edge map, sort them by size (in descending order)
# grab the largest contours
cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)#[:5]
print(cnts)
# initialize a contour that corresponds to the business card outline
cardCnt = None

# loop over the contours
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    # if this is the first contour we've encountered that has four
    # vertices, then we can assume we've found the business card
    if len(approx) == 4:
        cardCnt = approx
        break
# if the business card contour is empty then our script could not
# find the  outline of the card, so raise an error
if cardCnt is None:
    raise Exception("Could not find receipt outline." "Try debugging your edge detection and contour steps.")

output = image.copy()
# cv2.drawContours(output, [cardCnt], -1, (0, 255, 0), 2)
cv2.drawContours(output, cnts, -1, (0, 255, 0), 2)
cv2.imshow("Business Card Outline", output)



# cv2.imshow('test', edged)
cv2.waitKey(0)
