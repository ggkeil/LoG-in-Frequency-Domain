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
    
# normalize the image so that pixel values [0, 1]
def normalize(image):
    image = image // np.amax(image)
    return image
    
# multiply by (-1)^(x+y) to center or uncenter the image
def center(image):
    x, y = image.shape[:2]
    center = np.zeros((x, y))
    for i in range(x):
        for j in range(y):
            center[i, j] = (-1)**(i+j)
    centeredImage = image * center
    return centeredImage

# calculate the gaussian kernel
def gaussiankernel(image, sigma):
    iH = image.shape[0]
    iW = image.shape[1]
    cu, cv = iH // 2, iW // 2 # center of the kernel
    kernel = np.zeros((iH, iW))
    for u in range(0, iH):
        for v in range(0, iW):
            D = np.sqrt((u - cu)**2 + (v - cv)**2)
            kernel[u, v] = np.exp(-(D**2) / (2 * (sigma**2)))
    kernel = kernel // np.amax(kernel)
    return kernel
    
# calculate the laplacian kernel
def laplaciankernel(image):
    iH = image.shape[0]
    iW = image.shape[1]
    cu, cv = iH // 2, iW // 2
    kernel = np.zeros((iH, iW))
    for u in range(0, iH):
        for v in range(0, iW):
            D = np.sqrt((u - cu)**2 + (v - cv)**2)
            kernel[u, v] = (-4)*((np.pi)**2)*(D**2)
    kernel = kernel // np.amin(kernel) # normalize the Laplacian kernel
    return kernel

# used to multiply a fourier image by two kernels  
def multiply3(fft, kernel1, kernel2):
    first = np.multiply(fft, kernel1)
    output = np.multiply(first, kernel2)
    return output

def multiply2(fft, kernel):
    output = fft * kernel
    return output

def calculateFFT(image):
    ftimage = np.fft.fft2(image)
    return ftimage

# used for plotting the fourier of the image
def plotMagnitudeSpectrum(image):
    dft = cv2.dft(np.float32(image), flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))
    
    return magnitude_spectrum

# calculate IFFT after LoG
def calculateIFFT(image):
    ifft = np.abs(np.fft.ifft2(image))
    return ifft
    
# LoG given the fft of the image
def LoG(fft, sigma):
    gaussianKernel = gaussiankernel(fft, sigma) # first obtain the gaussian kernel calculated by this function
    laplacianKernel = laplaciankernel(fft) # then obtain the laplacian kernel calculated by this function
    result = multiply3(fft, gaussianKernel, laplacianKernel) # multiply the fourier transform by the gaussian and laplacian kernel
    return result

def unPad(paddedImage):
    (iH, iW) = paddedImage.shape[:2] # get padded image dimensions
    unPaddedImage = paddedImage[0 : iH // 2, 0 : iW // 2] # slicing off the black areas that surround the image
    return unPaddedImage

def sharpen(image, unpadded, c):
    sharpened = image + (c * unpadded)
    return sharpened

original = cv2.imread("CameraMan.png") # read in image

gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) # get grayscale of image

padded = padImage(gray) # pad the image

normalized = normalize(padded) # normalize the padded and centered image

centered = center(normalized) # center the padded image

fft = calculateFFT(centered) # calculate the fourier transform of the centered and normalized image

magSpecPlot = plotMagnitudeSpectrum(padded) # plot the Magnitude Spectrum of the Padded Image

LoG = LoG(fft, 50) # find the Laplacian of Gaussian (LoG) of the padded, normalized, and centered image in fourier

ifft = calculateIFFT(LoG) # find the inverse fourier of the LoG

unCenter = center(ifft) # uncenter the inverse fourier

unPadded = unPad(unCenter) # unpad the uncentered inverse fourier

sharpened = sharpen(gray, unPadded, -300) # sharpen the resulting image

plt.subplot(221), plt.imshow(gray, cmap = 'gray') # plot the gray scale image
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(222), plt.imshow(magSpecPlot, cmap = 'gray') # plot the magnitude spectrum
plt.title('Magnitude Spectrum Plot of Padded Image'), plt.xticks([]), plt.yticks([])

plt.subplot(223), plt.imshow(sharpened, cmap = 'gray') # plot the sharpened
plt.title('Sharpened Image'), plt.xticks([]), plt.yticks([])

plt.show()