
import cv2
import numpy as np
from matplotlib import pyplot as plt


def sobel_edge_detection(image):

    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image_blur = cv2.GaussianBlur(image_gray, (3, 3), 0)

    sobel_x = cv2.Sobel(src=image_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=1)
    sobel_y = cv2.Sobel(src=image_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=1)
    sobel_xy = cv2.Sobel(src=image_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=1)

    cv2.imshow('sobel x', sobel_x)
    cv2.imwrite('solutions/sobel_x.png', sobel_x)
    cv2.waitKey(0)
    cv2.imshow('sobel y', sobel_y)
    cv2.imwrite('solutions/sobel_y.png', sobel_y)
    cv2.waitKey(0)
    cv2.imshow('sobel xy', sobel_xy)
    cv2.imwrite('solutions/sobel_xy.png', sobel_xy)
    cv2.waitKey(0)



def canny_edge_detection(image, threshold_1, threshold_2):
    edges = cv2.Canny(image, threshold_1, threshold_2)
    plt.subplot(121), plt.imshow(edges, cmap='gray')
    plt.title('Edge Detection'), plt.xticks([]), plt.yticks([])
    plt.show()


def template_match(image, template):
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    w, h = template.shape[::-1]

    res = cv2.matchTemplate(image_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    cv2.imwrite('solutions/template_match.png', image)



def resize(image, scale_factor, up_or_down):
    if up_or_down == 'up':
        image = cv2.pyrUp(image, scale_factor)
        cv2.imwrite('solutions/resize_up.png', image)
    elif up_or_down == 'down':
        image = cv2.pyrDown(image, scale_factor)
        cv2.imwrite('solutions/resize_down.png', image)


def main():
    image = cv2.imread('lambo.png')

    #sobel edge
    sobel_edge_detection(image)

    #canny
    threshold_1 = 50
    threshold_2 = 50
    canny_edge_detection(image, threshold_1, threshold_2)

    #image template matching
    image = cv2.imread('shapes-1.png')
    template = cv2.imread('shapes_template.jpg', 0)
    template_match(image, template)

    #resize
    image = cv2.imread('lambo.png')
    scale_factor = 2
    up_or_down = 'down'
    resize(image, scale_factor, up_or_down)


if __name__ == "__main__":
    main()
