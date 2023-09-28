import cv2
import numpy as np
import os

# MACROS
GRAY = 0
BLUE = 0
GREEN = 1
RED = 2

# Shape Type
shape = {
    "vert": [[0, 1, 0], [0, 0, 0], [0, 1, 0]],
    "dash": [[0, 0, 0], [1, 0, 1], [0, 0, 0]],
    "cross": [[1, 0, 1], [0, 0, 0], [1, 0, 1]],
    "add": [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
}


# Class rawImage
class rawImage:
    # Constructor
    def __init__(self, fileName, dataType, height, width, channel):
        self.h = height
        self.w = width
        self.ch = channel
        self.dtype = dataType
        # If file not exist, create a new one
        if not os.path.isfile(fileName):
            self.data = np.zeros((height, width, channel), dtype=dataType)
        else:
            self.data = np.fromfile(fileName, dtype=dataType)  # read image
            self.data = self.data.reshape(height, width, channel)  # reshape image

    # Function: Change dtype
    def change_dtype(self, dataType):
        self.data = self.data.astype(dataType)
        self.dtype = dataType

    # Function: Turn Gray Image to BGR Image
    def gray2rgb(self):
        self.data = np.repeat(self, 3, axis=2)
        self.data.ch = 3

    #  Function: Turn BGR Image to Gray Image
    def rgb2gray(self):
        self.data = (
            self.data[:, :, 0] * 0.299
            + self.data[:, :, 1] * 0.587
            + self.data[:, :, 2] * 0.114
        ).astype(int)
        self.data.ch = 1

    # Function: Get Neighbor Pixels Average
    def get_averageN(self, posH, posW, posC, shape):
        nNum = 0
        nSum = 0
        for h in range(3):
            for w in range(3):
                if shape[h][w] == 0:
                    continue
                if posH + h - 1 < 0 or posH + h - 1 >= self.h:
                    continue
                if posW + w - 1 < 0 or posW + w - 1 >= self.w:
                    continue
                nNum += 1
                nSum += self.data[posH + h - 1][posW + w - 1][posC]
        return nSum // nNum


# Debayer with Bilinear Interpolation
def bilinear_inter(bayerImg):
    newImg = rawImage("", "uint8", bayerImg.h, bayerImg.w, 3)
    for h in range(bayerImg.h):
        for w in range(bayerImg.w):
            if h % 2 == 0 and w % 2 == 0:  # Blue Block
                newImg.data[h][w][BLUE] = bayerImg.data[h][w][GRAY]
                newImg.data[h][w][GREEN] = bayerImg.get_averageN(
                    h, w, GRAY, shape["add"]
                )
                newImg.data[h][w][RED] = bayerImg.get_averageN(
                    h, w, GRAY, shape["cross"]
                )
            elif h % 2 == 1 and w % 2 == 1:  # Red Block
                newImg.data[h][w][BLUE] = bayerImg.get_averageN(
                    h, w, GRAY, shape["cross"]
                )
                newImg.data[h][w][GREEN] = bayerImg.get_averageN(
                    h, w, GRAY, shape["add"]
                )
                newImg.data[h][w][RED] = bayerImg.data[h][w][GRAY]
            else:  # Green Block
                if h % 2 == 0 and w % 2 == 1:  # Green Block 1
                    newImg.data[h][w][BLUE] = bayerImg.get_averageN(
                        h, w, GRAY, shape["dash"]
                    )
                    newImg.data[h][w][RED] = bayerImg.get_averageN(
                        h, w, GRAY, shape["vert"]
                    )
                else:  # Green Block 2
                    newImg.data[h][w][BLUE] = bayerImg.get_averageN(
                        h, w, GRAY, shape["vert"]
                    )
                    newImg.data[h][w][RED] = bayerImg.get_averageN(
                        h, w, GRAY, shape["dash"]
                    )
                newImg.data[h][w][GREEN] = bayerImg.data[h][w][GRAY]
    return newImg


# Setup Image
oriImg = rawImage("origin.raw", "uint16", 1349, 1999, 1)

# Right shift 4 bits
oriImg.data = oriImg.data >> 4
oriImg.change_dtype("uint8")
newImg = bilinear_inter(oriImg)

# Save the result image as "result.bmp"
cv2.imwrite("origin.bmp", oriImg.data)
cv2.imwrite("result.bmp", newImg.data)


# Demo the result image
cv2.imshow("origin", oriImg.data)  # show image
cv2.waitKey(0), cv2.destroyAllWindows()  # press any key to close the window
cv2.imshow("result", newImg.data)  # show image
cv2.waitKey(0), cv2.destroyAllWindows()  # press any key to close the window
