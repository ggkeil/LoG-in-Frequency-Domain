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
    
def normalizeImage(image):
    (iH, iW) = image.shape[:2]
    for x in range(0, iH):
        for y in range(0, iW):
            image[x, y] = image[x, y] / 255
    
    return image

# shows the magnitude spectrum of the image
def plotMagnitudeSpectrum(image):
    dft = cv2.dft(np.float32(image), flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))
    
    return magnitude_spectrum
    
# Bring the image back
def backwardTransform(image):
    dft = cv2.dft(np.float32(image),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    rows, cols = image.shape
    crow,ccol = rows/2 , cols/2
    
    # create a mask first, center square is 1, remaining all zeros
    mask = np.zeros((rows,cols,2),np.uint8)
    mask[crow-30:crow+30, ccol-30:ccol+30] = 1
    
    # apply mask and inverse DFT
    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    return img_back

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
    kernel = np.zeros(iH, iW)
    for u in range(0, iH):
        for v in range(0, iW):
            D = np.sqrt((u - cu)**2 + (v - cv)**2)
            kernel[u, v] = -4*((np.pi)**2)*(D**2)
    return kernel
    
def multiply(image, kernel):
    output = np.multiply(image, kernel)
    return output

# Inverse Fourier to get image back
def showChangedImage(image, kernel):
    ftimage = np.fft.fft2(image)
    ftimage = np.fft.fftshift(ftimage)
    ftimagep = multiply(ftimage, kernel)
    imagep = np.fft.ifft2(ftimagep)
    return np.abs(imagep)

# multiply by (-1)^(x+y)
def unCenterImage(image):
    result = np.zeros((image.shape[0], image.shape[1]))
    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[1]):
            result[x, y] = image[x, y] * pow(-1, x + y)
    
    return result

def unPadImage(paddedImage):
    (iH, iW) = paddedImage.shape[:2]
    unPaddedImage = paddedImage[0 : int(iH / 2), 0 : int(iW / 2)]
    return unPaddedImage

original = cv2.imread("woman.png")
gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
padded = padImage(gray)
magSpecPlot = plotMagnitudeSpectrum(padded)
sigma = 25
gaussianOfFourierPlot = multiply(magSpecPlot, gaussiankernel(magSpecPlot, sigma))
changed_image = showChangedImage(padded, gaussiankernel(padded, sigma))
unpadded_changed = unPadImage(changed_image)

plt.subplot(231),plt.imshow(padded, cmap = 'gray')
plt.title('Input Padded Image'), plt.xticks([]), plt.yticks([])
plt.subplot(232),plt.imshow(magSpecPlot, cmap = 'gray')
plt.title('Magnitude Spectrum of Padded Image'), plt.xticks([]), plt.yticks([])
plt.subplot(233),plt.imshow(gaussianOfFourierPlot, cmap = 'gray')
plt.title('Gaussian Magnitude Spectrum of Padded Image'), plt.xticks([]), plt.yticks([])
plt.subplot(234),plt.imshow(changed_image, cmap = 'gray')
plt.title('Image Came Back'), plt.xticks([]), plt.yticks([])
plt.subplot(235),plt.imshow(unpadded_changed, cmap = 'gray')
plt.title('Unpadded'), plt.xticks([]), plt.yticks([])
plt.show()