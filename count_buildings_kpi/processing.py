'''
Спробувала виділити контури будівель кампусу КПІ
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter


def image_read(file_name):
    image = cv2.imread(file_name)
    return image


def filter_image(img_cv):
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    img_contrast = ImageEnhance.Contrast(img_pil).enhance(1.5)

    img_sharpness = ImageEnhance.Sharpness(img_contrast).enhance(2.0)

    img_edges = img_sharpness.filter(ImageFilter.EDGE_ENHANCE)

    img_cv_filtered = cv2.cvtColor(np.array(img_edges), cv2.COLOR_RGB2BGR)
    return img_cv_filtered


def gabor_filter(gray):
    kernel = cv2.getGaborKernel(
        (21, 21),
        8.0,
        np.pi / 4,
        10.0,
        0.5,
        0,
        ktype=cv2.CV_32F
    )
    filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
    return filtered


def image_processing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    # gabor = gabor_filter(gray) # Gabor фільтр
    edged = cv2.Canny(gray, 100, 200)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel_close)
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

    plt.imshow(cleaned)
    plt.show()

    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 500
    building_count = 0

    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            building_count += 1
            cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)

    print(f"Знайдено будівель: {building_count}")
    cv2.imshow("Result", image)
    cv2.waitKey(0)


if __name__ == "__main__":
    image = image_read("campus.png")
    fltr = filter_image(image)
    plt.imshow(fltr, cmap='gray')
    plt.show()
    image_processing(fltr)
