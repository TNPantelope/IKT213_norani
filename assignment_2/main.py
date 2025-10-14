
import cv2
import numpy as np
from matplotlib import pyplot as plt


def padding(image, border_width):
    reflect = cv2.copyMakeBorder(image, border_width, border_width, border_width, border_width, cv2.BORDER_REFLECT)
    plt.imshow(reflect, 'gray'), plt.title('reflect')
    cv2.imwrite('solutions/padded_image.png', reflect)
    plt.show()

def crop(image, x_0, x_1, y_0, y_1):
    print(image.shape)
    cropped_image = image[y_0:y_1, x_0:x_1]
    cv2.imshow("cropped", cropped_image)
    cv2.imwrite('solutions/cropped_image.png', cropped_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def resize(image, width, height):
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("resized", resized_image)
    cv2.imwrite('solutions/resized_image.png', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def copy(image, emptyPictureArray):
    emptyPictureArray[:] = image[:]

    cv2.imshow('image', image)
    cv2.imshow('emptyPictureArray', emptyPictureArray)
    cv2.imwrite('solutions/copy.png', emptyPictureArray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def grayscale(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('grayscale', grayscale_image)
    cv2.imwrite('solutions/grayscale.png', grayscale_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imshow('hsv', hsv_image)
    cv2.imwrite('solutions/hsv.png', hsv_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def hue_shifted(image, emptyPictureArray, hue):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    emptyPictureArray[:] = hsv_image[:]
    emptyPictureArray[:, :, 0] = (emptyPictureArray[:, :, 0].astype(np.int16) + hue) % 180
    hue_shifted_image = cv2.cvtColor(emptyPictureArray, cv2.COLOR_HSV2BGR)
    cv2.imshow('hue_shifted', hue_shifted_image)
    cv2.imwrite('solutions/hue_shiftedfixed.png', hue_shifted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def smoothing(image):
    blurred_image = cv2.GaussianBlur(image, (15, 15), cv2.BORDER_DEFAULT)
    cv2.imshow('blurred', blurred_image)
    cv2.imwrite('solutions/blurred.png', blurred_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def rotation(image, rotation_angle):
    if rotation_angle == 90:
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        cv2.imshow('rotated', rotated_image)
        cv2.imwrite('solutions/rotated.png', rotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif rotation_angle == 180:
        rotated_image = cv2.rotate(image, cv2.ROTATE_180)
        cv2.imshow('rotated', rotated_image)
        cv2.imwrite('solutions/rotated180.png', rotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("wrong rotation angle")


def main():
    image = cv2.imread('lena-2.png')
    height, width, channels = image.shape

    #padding part
    border_width = 100
    padding(image, border_width)

    #crop part
    x_0 = 80
    x_1 = image.shape[1] - 130
    y_0 = 80
    y_1 = image.shape[0] - 130
    crop(image, x_0, x_1, y_0, y_1)

    #resize part
    resize(image, 200, 200)

    #copy part
    height, width, channels = image.shape
    emptyPictureArray = np.zeros((height, width, 3), dtype=np.uint8)
    copy(image, emptyPictureArray)

    #grayscale part
    grayscale(image)

    #hsv part
    hsv(image)

    #color shifting
    hue_shifted(image, emptyPictureArray, 50)

    #smoothing
    smoothing(image)

    #rotation
    rotation(image, 90)
    rotation(image, 180)

if __name__ == "__main__":
    main()