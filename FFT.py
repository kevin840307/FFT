from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import seaborn as sns
import pandas as pd

def get_w(len, oper):
    var = -2
    if oper != 1:
        var = 2
    w_complex = np.zeros([len, 2])
    for i in range(len):
        c = [np.cos((var * np.pi * i / len)), np.sin((var * np.pi * i / len))]
        w_complex[i] = c
    return w_complex

def complex_multi(com1, com2):
    r = com1[0] * com2[0] - com1[1] * com2[1]
    i = com1[0] * com2[1] + com1[1] * com2[0]
    return np.array([r, i])

def complex_add(com1, com2):
    r = com1[0] + com2[0]
    i = com1[1] + com2[1]
    return np.array([r, i])

def complex_sub(com1, com2):
    r = com1[0] - com2[0]
    i = com1[1] - com2[1]
    return np.array([r, i])


def FFT_1D(complexs):
    size = len(complexs)
    if size == 1:
        return complexs;

    w_len = size >> 1
    num1 = complexs[0::2]
    num2 = complexs[1::2]

    num_complex = np.zeros([size, 2])
    com1 = FFT_1D(num1)
    com2 = FFT_1D(num2)
    w_complex = get_w(size, 1)

    for i in range(w_len):
        com_mul = complex_multi(w_complex[i], com2[i])
        num_complex[i] = complex_add(com1[i], com_mul)
        num_complex[w_len + i] = complex_sub(com1[i], com_mul)
    return num_complex;

def FFT(matrix):
    width = matrix.shape[1]
    height = matrix.shape[0]
    fft = np.zeros((height, width, 2))
    fft[:,:,0] = matrix

    for col in range(height):
        fft[col,:,:] = FFT_1D(fft[col,:,:])

    fft = np.transpose(fft, [1, 0, 2])

    for row in range(width):
        fft[row,:,:] = FFT_1D(fft[row,:,:])

    fft = np.transpose(fft, [1, 0, 2])
    return fft

def IFFT_1D(complexs):
    size = len(complexs)
    if size == 1:
        return complexs;

    w_len = size >> 1
    num1 = complexs[0::2]
    num2 = complexs[1::2]

    num_complex = np.zeros([size, 2])
    com1 = IFFT_1D(num1)
    com2 = IFFT_1D(num2)
    w_complex = get_w(size, -1)

    for i in range(w_len):
        com_mul = complex_multi(w_complex[i], com2[i])
        num_complex[i] = complex_add(com1[i], com_mul)
        num_complex[w_len + i] = complex_sub(com1[i], com_mul)
    return num_complex;

def IFFT(fft):
    width = fft.shape[1]
    height = fft.shape[0]
    matrix = fft

    matrix = np.transpose(matrix, [1, 0, 2])

    for row in range(width):
        matrix[row,:,:] = IFFT_1D(matrix[row,:,:])

    matrix = np.transpose(matrix, [1, 0, 2])

    for col in range(height):
        matrix[col,:,:] = IFFT_1D(matrix[col,:,:])

    return matrix


#value = np.arange(10, 170, 10,dtype=np.float)
#value = np.reshape(value, (4, 4))
#fft = np.fft.fft2(value)
#print(fft)
#ifft = np.fft.ifft2(fft)
#print(ifft)

#print('--------------------------------------')

#value = np.arange(10, 170, 10,dtype=np.float)
#value = np.reshape(value, (4, 4))
#fft = FFT(value)
#print(fft)
#ifft = IFFT(fft) / 16
#print(ifft.astype(np.int))



#img = cv2.imread("C:/Users/USER/Desktop/1.jpg")
#img = img[:, :, 0]
#img = np.float32(img)
#img_dct = cv2.dct(img)
#plt.imshow(img_dct, cmap = plt.cm.gray)
#plt.show()
#img_recor = cv2.idct(img_dct)
#plt.imshow(img_recor, cmap = plt.cm.gray)
#plt.show()

#img_dct = FFT(img)
#plt.imshow(img_dct, cmap = plt.cm.gray)
#plt.show()
#img_recor = IFFT(img_dct)
#plt.imshow(img_recor, cmap = plt.cm.gray)
#plt.show()

