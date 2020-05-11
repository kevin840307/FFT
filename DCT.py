from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import seaborn as sns
import pandas as pd

def DCT_process(matrix, i, j):
    width = matrix.shape[1]
    height = matrix.shape[0]
    value = 0.
    for col in range(height):
        for row in range(width):
            save = matrix[col, row] 
            save *= math.cos(math.pi * (2 * col + 1) * i / (2. * height))
            save *= math.cos(math.pi * (2 * row + 1) * j / (2. * width))
            value += save
    c = 1.
    if i == 0:
        c /= np.sqrt(2)
    if j == 0:
        c /= np.sqrt(2)

    return (2. / np.sqrt(height * width)) * c * value

def DCT(matrix):
    width = matrix.shape[1]
    height = matrix.shape[0]
    dct = np.zeros_like(matrix)

    for col in range(height):
        for row in range(width):
            dct[col, row] = DCT_process(matrix, col, row)
    return dct

def IDCT_process(dct, i, j):
    width = dct.shape[1]
    height = dct.shape[0]
    value = 0

    for col in range(height):
        for row in range(width):
            save = dct[col, row] 
            if col == 0:
                save /= np.sqrt(2)
            if row == 0:
                save /= np.sqrt(2)
            save *= math.cos(math.pi * (2 * i + 1) * col / (2. * height))
            save *= math.cos(math.pi * (2 * j + 1) * row / (2. * width))
            value += save

    return (2. / np.sqrt(height * width)) * value

def IDCT(dct):
    width = dct.shape[1]
    height = dct.shape[0]
    matrix = np.zeros_like(dct)

    for col in range(height):
        for row in range(width):
            matrix[col, row] = IDCT_process(dct, col, row)
    return matrix


value = np.arange(10, 160, 10,dtype=np.float)
matrix = np.reshape(value, (5, 3))
matrix = np.float32(matrix)
matrix_dct = cv2.dct(matrix)
print(matrix_dct.astype(np.int))
print(cv2.idct(matrix_dct))

matrix_dct = DCT(matrix)
print(IDCT(matrix_dct))
matrix_dct = matrix_dct.astype(np.int)
print(matrix_dct)


img = cv2.imread("C:/Users/USER/Desktop/Lena.jpg")
img = img[:, :, 0]
img = np.float32(img)
img_dct = cv2.dct(img)
plt.imshow(img_dct, cmap = plt.cm.gray)
plt.show()
img_recor = cv2.idct(img_dct)
plt.imshow(img_recor, cmap = plt.cm.gray)
plt.show()

img_df = pd.DataFrame({"pixel": img.astype(np.int).reshape([-1])})

img_df.hist('pixel', bins=20, grid=False)
#sns.countplot(x="pixel", data=img_df)
plt.show()

img_df = pd.DataFrame({"pixel": img_dct.astype(np.int).reshape([-1])})
img_df.hist('pixel', bins=20, grid=False)
#sns.countplot(x="pixel", data=img_df)
plt.show()

#img_dct = DCT(img)
#plt.imshow(img_dct, cmap = plt.cm.gray)
#plt.show()
#img_recor = IDCT(img_dct)
#plt.imshow(img_recor, cmap = plt.cm.gray)
#plt.show()

