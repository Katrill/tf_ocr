import tensorflow as tf
import cv2
import numpy as np


def viewImage(name_of_window, img):
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_window, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def face(image_face):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image_face, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10))
    # faces_detected = format(len(faces)) + " faces detected!"
    # Draw a rectangle around the faces
    for (xc, yc, wc, hc) in faces:  # top left coordinates
        cv2.rectangle(image_face, (xc, yc), (xc + wc, yc + hc), (255, 255, 0), 2)
        return xc, yc, wc, hc


def sorted_list(list_of_tuples):
    return list_of_tuples.sort(key=lambda z: z[0], reverse=False)


def input_image(img_arr):
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = img_arr / 255.0
    img_arr = img_arr.reshape((1, 28, 28, 1))
    return img_arr


def decode_res(ind):
    letters_set = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
    dict_letters = {}
    for j, letter in enumerate(letters_set):
        dict_letters[j] = letter
    return dict_letters[ind]


# Recreate the exact same model, including its weights and the optimizer
model = tf.keras.models.load_model('my_h5_model.h5')

# Show the model architecture
# new_model.summary()

image = cv2.imread("passport/0.jpeg")
height, width, _ = image.shape
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# viewImage("gray", gray)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
thresh_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# viewImage("thresh_inv", thresh_inv)
# img_erode = cv2.erode(thresh_inv, np.ones((2, 2), np.uint8), iterations=1)
img_dilate = cv2.dilate(thresh_inv, np.ones((1, 1), np.uint8), iterations=1)
# viewImage("erode", img_dilate)
contours, hierarchy = cv2.findContours(img_dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# get face coordinates
xf, yf, wf, hf = face(gray)

output = image.copy()
im2 = image.copy()
out_size = 28
letters = []

for idx, contour in enumerate(contours):
    (x, y, w, h) = cv2.boundingRect(contour)
    if (2.5*(xf+w) < x < 4.7*(xf+w)) and (0.85*yf < y < 0.95*(yf + hf)) and h > 0.085*hf:
        cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
        letter_crop = gray[y:y + h, x:x + w]
        # Resize letter canvas to square
        size_max = max(w, h)
        letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
        if w > h:
            # Enlarge image top-bottom
            y_pos = size_max // 2 - h // 2
            letter_square[y_pos:y_pos + h, 0:w] = letter_crop
        elif w < h:
            # Enlarge image left-right
            x_pos = size_max // 2 - w // 2
            letter_square[0:h, x_pos:x_pos + w] = letter_crop
        else:
            letter_square = letter_crop
        # Resize letter to 28x28 and add letter and its X-coordinate
        if len(letters) == 0:
            letters.append((x, w, cv2.resize(letter_square, (out_size, out_size), interpolation=cv2.INTER_AREA), y, h))
        else:
            if abs(y - letters[-1][3]) > 2*letters[-1][4]:
                letters.append('N')
                letters.append((x, w, cv2.resize(letter_square, (out_size, out_size), interpolation=cv2.INTER_AREA),
                                y, h))
            else:
                letters.append((x, w, cv2.resize(letter_square, (out_size, out_size), interpolation=cv2.INTER_AREA),
                                y, h))
# viewImage("output", output)

# sort by coordinate x
letters_x = []
num = []
for i in range(len(letters)):
    if letters[i] != "N":
        num.append((letters[i][0], letters[i][2]))
    else:
        sorted_list(num)
        letters_x.append(num)
        num = []
sorted_list(num)
letters_x.append(num)

# predict, decode and print
result = []
for i in range(len(letters_x)):
    for j in range(len(letters_x[i])):
        image_to_predict = input_image(letters_x[i][j][1])
        predict = model.predict([image_to_predict])
        result.append(decode_res(np.argmax(predict, axis=1)[0]))
    result.append('Next')
[print(elem, end='') if elem != "Next" else print() for i, elem in enumerate(result)]
