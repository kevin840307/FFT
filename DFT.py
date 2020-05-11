from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import seaborn as sns
import pandas as pd

def DFT_process(matrix, i, j):
    pi = np.pi * 2.
    width = matrix.shape[1]
    height = matrix.shape[0]
    value = np.array([0., 0.])

    for col in range(height):
        for row in range(width):
            angle = pi * ((i * col) / float(height) + (j * row) / float(width))
            value[0] += matrix[col, row] * np.cos(angle)
            value[1] -= matrix[col, row] * np.sin(angle)

    #return (1. / np.sqrt(height * width)) * value
    return value

def DFT(matrix):
    width = matrix.shape[1]
    height = matrix.shape[0]
    dct = np.zeros((height, width, 2))

    for col in range(height):
        for row in range(width):
            dct[col, row] = DFT_process(matrix, col, row)
    return dct


def IDFT_process(dft, i, j):
    pi = np.pi * 2.
    width = dft.shape[1]
    height = dft.shape[0]
    value = np.array([0., 0.])

    for col in range(height):
        for row in range(width):
            angle = pi * ((i * col) / float(height) + (j * row) / float(width))
            value[0] += dft[col, row, 0] * np.cos(angle) - dft[col, row, 1] * np.sin(angle)
            value[1] += dft[col, row, 0] * np.sin(angle) + dft[col, row, 1] * np.cos(angle)

    return value

def IDFT(dft):
    width = dft.shape[1]
    height = dft.shape[0]
    matrix = np.zeros_like(dft)

    for col in range(height):
        for row in range(width):
            matrix[col, row] = IDFT_process(dft, col, row)
    return matrix


value = np.arange(10, 260, 10,dtype=np.float)
matrix = np.reshape(value, (5, 5))
matrix_dct = cv2.dft(matrix, flags=cv2.DFT_COMPLEX_OUTPUT)
print(matrix_dct.astype(np.int))
idft = cv2.idft(matrix_dct)
print('')
print(idft[:,:,0]  / 25.) # sqrt(5 * 5)

print('--------------------------------------')

matrix_dct = DFT(matrix)
print(matrix_dct.astype(np.int))
print('\r\n\r\n')
print(IDFT(matrix_dct)[:,:,0] / 25. ) # sqrt(5 * 5)
matrix_dct[1:,:,] = 0
print('\r\n\r\n')
print(matrix_dct.astype(np.int))
print('\r\n\r\n')
print(IDFT(matrix_dct)[:,:,0] / 25. ) # sqrt(5 * 5)

#img = cv2.imread("C:/Users/USER/Desktop/1.jpg")
#img = img[:, :, 0]
#img = np.float32(img)
#img_dct = cv2.dct(img)
#plt.imshow(img_dct, cmap = plt.cm.gray)
#plt.show()
#img_recor = cv2.idct(img_dct)
#plt.imshow(img_recor, cmap = plt.cm.gray)
#plt.show()

#img_df = pd.DataFrame({"pixel": img.astype(np.int).reshape([-1])})
#sns.countplot(x="pixel", data=img_df)
#plt.show()

#img_df = pd.DataFrame({"pixel": img_recor.astype(np.int).reshape([-1])})
#sns.countplot(x="pixel", data=img_df)
#plt.show()

#img_dct = DFT(img)
#plt.imshow(img_dct, cmap = plt.cm.gray)
#plt.show()
#img_recor = IDFT(img_dct)
#plt.imshow(img_recor, cmap = plt.cm.gray)
#plt.show()

