"""
Домашнє завдання 1, група результатів 1
"""

import cv2
from matplotlib import pyplot as plt

image_1 = cv2.imread("Sunf5.jpg")
image_2 = cv2.imread("Sunf4.jpg")


def image_processing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # корекція кольору (відтінки сірого)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)  # Гаусова фільтрація
    edged = cv2.Canny(gray, 30, 150)  # фільтр Кенні - векторизація

    return edged


edged_1 = image_processing(image_1)
edged_2 = image_processing(image_2)


def show_(val):
    plt.imshow(val)
    plt.show()

show_(edged_1)
show_(edged_2)

contours1 = max(cv2.findContours(edged_1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
contours2 = max(cv2.findContours(edged_2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)

compare = cv2.matchShapes(contours1, contours2, cv2.CONTOURS_MATCH_I1, 0)
print("Схожість об’єктів:", compare)
