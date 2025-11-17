import cv2
import numpy as np
from matplotlib import pyplot as plt

def harris(refrence_image):
    gray = cv2.cvtColor(refrence_image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    refrence_image[dst > 0.01 * dst.max()] = [0, 0, 255]
    cv2.imwrite("solutions/harris.png", refrence_image)

def SIFT(image_to_align, refrence_image, max_features, good_match_precent):
    img1 = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(refrence_image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < good_match_precent * n.distance:
            good.append(m)

    if len(good) >= max_features:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        height, width = img2.shape
        aligned_img = cv2.warpPerspective(image_to_align, M, (width, height))
        cv2.imwrite("solutions/aligned.png", aligned_img)
    else:
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask, flags=2)
    matches_img = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    cv2.imwrite("solutions/matches.png", matches_img)

def main():
    refrence_image = cv2.imread('reference_img.png')
    image_to_align = cv2.imread("align_this.jpg")

    max_features = 10
    good_match_percent = 0.7

    harris(refrence_image)
    SIFT(image_to_align, refrence_image, max_features, good_match_percent)

if __name__ == "__main__":
    main()