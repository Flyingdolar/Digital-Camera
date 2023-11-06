import cv2
import numpy as np
import os

# MACROS
GRAY = 0
BLUE = 0
GREEN = 1
RED = 2


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

    def demosaic(self, type):
        if type == "RG":
            self.data = cv2.cvtColor(self.data, cv2.COLOR_BAYER_RG2BGR)
        elif type == "GR":
            self.data = cv2.cvtColor(self.data, cv2.COLOR_BAYER_GR2BGR)
        elif type == "GB":
            self.data = cv2.cvtColor(self.data, cv2.COLOR_BAYER_GB2BGR)
        elif type == "BG":
            self.data = cv2.cvtColor(self.data, cv2.COLOR_BAYER_BG2BGR)
        else:
            print("Error: Wrong demosaic type")
            return
        self.ch = 3

    # Function: Crop Image
    def crop(self, startH=0, startW=0, endH=-1, endW=-1):
        endH = self.h if endH == -1 else endH
        endW = self.w if endW == -1 else endW
        self.data = self.data[startH:endH, startW:endW, :]
        self.h = endH - startH
        self.w = endW - startW
