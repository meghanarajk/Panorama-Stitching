# -*- coding: utf-8 -*-
u

import cv2
import numpy as np
# np.random.seed(<int>) # you can use this line to set the fixed random seed if you are using np.random
import random


# random.seed(<int>) # you can use this line to set the fixed random seed if you are using random

def warpTwoImages(left_image, right_image, H):
    '''warp img2 to img1 with homograph H'''
    m, n = left_image.shape[:2]
    k, l = right_image.shape[:2]
    pts1 = np.float32([[0, 0], [0, m], [n, m], [n, 0]]).reshape(-1, 1, 2)
    pts2 = np.float32([[0, 0], [0, k], [l, k], [l, 0]]).reshape(-1, 1, 2)
    tranformed_p = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, tranformed_p), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() - 0.5)
    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])  # translate

    result = cv2.warpPerspective(right_image, Ht.dot(H), (xmax - xmin, ymax - ymin))
    result[t[1]:m+t[1],t[0]:n+t[0]] = left_image
    return result


def h_func(r_tuple):
    A = np.zeros((len(r_tuple) * 2, 9))
    i = 0
    for tuple in r_tuple:
        A[i] = [tuple[2], tuple[3], 1, 0, 0, 0, -tuple[0] * tuple[2], -tuple[0] * tuple[3], - tuple[0]]
        A[i + 1] = [0, 0, 0, tuple[2], tuple[3], 1, -tuple[1] * tuple[2], -tuple[1] * tuple[3], - tuple[1]]
        i = i + 2
    w, s, vh = np.linalg.svd(A, full_matrices=True)
    x = vh[8]
    result_x = np.reshape(x, (3, 3))
    return result_x


def solution(left_img, right_img):
    """
    :param left_img:
    :param right_img:
    :return: you need to return the result panorama image which is stitched by left_img and right_img
    """


    return result_img


if __name__ == "__main__":
    left_img = cv2.imread('left.jpg')
    right_img = cv2.imread('right.jpg')
    result_img = solution(left_img, right_img)
    cv2.imwrite('results/task1_result.jpg', result_img)
