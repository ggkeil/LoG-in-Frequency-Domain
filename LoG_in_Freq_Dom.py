# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 20:02:42 2019

@author: eagle
"""

import numpy as np
import cv2
import cmath

def padImage(image):
    (iH, iW) = image.shape[:2]
    
    pad1 = iH
    pad2 = iW
    
    image = cv2.copyMakeBorder(image, 0, pad1, 0, pad2,
		cv2.BORDER_CONSTANT, 0) # you get a P x Q image
                                # P = 2M and Q = 2N
    return image
    
def centerImage(image):
    result = np.zeros((image.shape[0], image.shape[1]))
    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[1]):
            result[x, y] = image[x, y] * ((-1)**(x + y))
    
    return result

def forwardTransform(image):
    (M, N) = image.shape[:2]
    fourier = np.zeros((M, N))
    u = 0
    v = 0
    sum = 0
    for u in range(M):
        for v in range(N):
            sum = 0
            for x in range(M):
                for y in range(N):
                    sum += image[x, y]*cmath.exp(-1j*2*cmath.pi*((float(u*x) / M) + (float(v*y) / N)))
            fourier[u, v] = sum
    
def imagefilter(image, kernel):
    output = image * kernel
    return output

original = cv2.imread("woman.png")
gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
padded = padImage(gray)
centered = centerImage(padded)

# cv2.imshow("Padded Image", padded)
cv2.imshow("Centered", centered)