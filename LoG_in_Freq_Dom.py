# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 20:02:42 2019

@author: eagle
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def padImage(image):
    (iH, iW) = image.shape[:2]
    
    pad1 = iH
    pad2 = iW
    
    image = cv2.copyMakeBorder(image, 0, pad1, 0, pad2,
		cv2.BORDER_CONSTANT, 0) # you get a P x Q image
                                # P = 2M and Q = 2N
    return image
    
def normalize(image):
    image = image // np.amax(image)
    return image
    
# multiply by (-1)^(x+y)
def center(image):
    x, y = image.shape[:2]
    center = np.zeros((x, y))
    for i in range(x):
        for j in range(y):
            center[i, j] = (-1)**(i+j)
    centeredImage = image * center
    return centeredImage

def gaussiankernel(image, sigma):
    iH = image.shape[0]
    iW = image.shape[1]
    cu, cv = iH // 2, iW // 2 # center of the kernel
    kernel = np.zeros((iH, iW))
    for u in range(0, iH):
        for v in range(0, iW):
            D = np.sqrt((u - cu)**2 + (v - cv)**2)
            kernel[u, v] = np.exp(-(D**2) / (2 * (sigma**2)))
    return kernel
    
def laplaciankernel(image):
    iH = image.shape[0]
    iW = image.shape[1]
    cu, cv = iH // 2, iW // 2
    kernel = np.zeros((iH, iW))
    for u in range(0, iH):
        for v in range(0, iW):
            D = np.sqrt((u - cu)**2 + (v - cv)**2)
            kernel[u, v] = (-4)*((np.pi)**2)*(D**2)
    kernel = kernel // np.amin(kernel)
    return kernel
    
def multiply(fft, kernel1, kernel2):
    output = fft*kernel1*kernel2
    return output

def multiply2(fft, kernel):
    output = fft * kernel
    return output

def calculateFFT(image):
    ftimage = np.fft.fft2(image)
    return ftimage

# calculate IFFT after LoG
def calculateIFFT(image):
    ifft = np.abs(np.fft.ifft2(image))
    return ifft
    
# LoG given the fft of the image
def LoG(fft, sigma):
    gaussianKernel = gaussiankernel(fft, sigma) # first obtain the gaussian kernel calculated by this function
    laplacianKernel = laplaciankernel(fft) # then obtain the laplacian kernel calculated by this function
    result = multiply(fft, gaussianKernel, laplacianKernel) # multiply the fourier transform by the gaussian and laplacian kernel
    return result

def unPad(paddedImage):
    (iH, iW) = paddedImage.shape[:2] # get padded image dimensions
    unPaddedImage = paddedImage[0 : iH // 2, 0 : iW // 2] # slicing off the black areas that surround the image
    return unPaddedImage

def sharpen(image, unpadded, c):
    sharpened = image + (c * unpadded)
    return sharpened

original = cv2.imread("woman.png") # read in image

gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) # get grayscale of image

padded = padImage(gray) # pad the image

normalized = normalize(padded) # normalize the padded and centered image

centered = center(normalized) # center the padded image

fft = calculateFFT(centered) # calculate the fourier transform of the centered and normalized image

gaussianFourier = multiply2(fft, gaussiankernel(fft, 25))

gaussianIFFT = calculateIFFT(gaussianFourier)

unCenteredGaussian = center(gaussianIFFT)

unPaddedGaussian = unPad(unCenteredGaussian)

LoG = LoG(fft, 25)

ifft = calculateIFFT(LoG)

unCenter = center(ifft)

unPadded = unPad(unCenter)

sharpened = sharpen(gray, unPadded, 3)

plt.subplot(121), plt.imshow(gray, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

#plt.subplot(222), plt.imshow(unPaddedGaussian, cmap = 'gray')
#plt.title('Gaussian Image'), plt.xticks([]), plt.yticks([])

plt.subplot(122), plt.imshow(sharpened, cmap = 'gray')
plt.title('Sharpened Image'), plt.xticks([]), plt.yticks([])

plt.show()