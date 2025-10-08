
import cv2
import numpy as np
import os
import time
from sklearn.metrics import confusion_matrix, accuracy_score



def preprocess_image(img_path):
    img = cv2.imread(img_path, 0)
    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return img_bin


def match_orb(img1, img2):
    start_time = time.time()

    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return 0, time.time() - start_time, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    time_spent = time.time() - start_time
    return len(good_matches), time_spent, match_img

def match_sift(img1, img2):
    start_time = time.time()

    sift = cv2.SIFT_create(nfeatures=1000)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return 0, time.time() - start_time, None

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    time_spent = time.time() - start_time
    return len(good_matches), time_spent, match_img


def test_data_check(data_path, match_func, method_name, results_folder):
    y_true = []
    y_pred = []
    times = []
    threshold = 20

    os.makedirs(results_folder, exist_ok=True)

    print(f"\ntesting {method_name} data_check")

    for folder in sorted(os.listdir(data_path)):
        folder_path = os.path.join(data_path, folder)
        if not os.path.isdir(folder_path):
            continue

        images = [f for f in os.listdir(folder_path) if f.endswith(('.tif', '.png', '.jpg', '.bmp'))]
        if len(images) != 2:
            continue

        img1_path = os.path.join(folder_path, images[0])
        img2_path = os.path.join(folder_path, images[1])

        img1 = preprocess_image(img1_path)
        img2 = preprocess_image(img2_path)

        matches, time_spent, match_img = match_func(img1, img2)
        times.append(time_spent)

        actual = 1 if "same" in folder.lower() else 0
        predicted = 1 if matches > threshold else 0

        y_true.append(actual)
        y_pred.append(predicted)

        if match_img is not None:
            img_filename = f"{folder}_{method_name}.png"
            cv2.imwrite(os.path.join(results_folder, img_filename), match_img)

        print(f"{folder}: {matches} matches, {time_spent:.3f}s")

    return y_true, y_pred, times

def test_uia_images(data_path, results_folder):
    images = sorted([f for f in os.listdir(data_path) if f.endswith(('.tif', '.png', '.jpg', '.bmp'))])

    os.makedirs(results_folder, exist_ok=True)
    print(f"testing uia images")


    results = []
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            img1_path = os.path.join(data_path, images[i])
            img2_path = os.path.join(data_path, images[j])

            img1 = preprocess_image(img1_path)
            img2 = preprocess_image(img2_path)

            orb_matches, orb_time, orb_img = match_orb(img1, img2)
            sift_matches, sift_time, sift_img = match_sift(img1, img2)

            print(f"{images[i]} vs {images[j]}:")
            print(f"  orbbf:     {orb_matches} matches, {orb_time:.3f}s")
            print(f"  siftflann: {sift_matches} matches, {sift_time:.3f}s\n")

            results.append({'orb': (orb_matches, orb_time), 'sift': (sift_matches, sift_time)})

    return results

def print_results(method_name, y_true, y_pred, times):
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    print(f"\n{method_name} results:")
    print(f"acc: {acc * 100:.1f}%")
    print(f"avg_time: {np.mean(times):.4f}s")
    print(f"TN={cm[0, 0]} FP={cm[0, 1]} FN={cm[1, 0]} TP={cm[1, 1]}")




def main():
    data_check = "data_check"
    data_uia = "data_uia"

    results_orb = "results_orb"
    results_sift = "results_sift"
    results_uia = "results_uia"

    y_true_orb, y_pred_orb, times_orb = test_data_check(data_check, match_orb, "orbbf", results_orb)
    print_results("orbbf", y_true_orb, y_pred_orb, times_orb)

    y_true_sift, y_pred_sift, times_sift = test_data_check(data_check, match_sift, "siftflann", results_sift)
    print_results("siftflann", y_true_sift, y_pred_sift, times_sift)

    uia_results = test_uia_images(data_uia, results_uia)

    print(f"\nuia:")
    orb_total = sum([r['orb'][1] for r in uia_results])
    sift_total = sum([r['sift'][1] for r in uia_results])
    print(f"orb total: {orb_total:.3f}s")
    print(f"sift total: {sift_total:.3f}s")


if __name__ == "__main__":
    main()