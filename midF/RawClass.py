import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# MACROS
GRAY = 0
BGR = 1
HSV = 2
BLUE = 0
GREEN = 1
RED = 2


# Class RAWDATA
class RAWDATA:
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

    # Function: Demosaic
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

    # Function: Draw Histogram
    def draw_histogram(
        self,
        channel,
        isSaved=False,
        fileName="histogram.png",
        isShow=True,
    ):
        # Calculate min and max value
        if self.dtype == np.uint8:
            minVal = 0
            maxVal = 255
            bins = 256
        elif self.dtype == np.uint16:
            minVal = 0
            maxVal = 65535
            bins = 65536
        elif self.dtype == np.float32:
            minVal = 0
            maxVal = 1
            bins = 65536
        else:
            print("Error: Wrong dtype")
            return
        if channel == GRAY:
            plt.hist(self.data.ravel(), bins, [minVal, maxVal])
            plt.ylim(0, 6000)
        elif channel == BGR:
            # Draw 3 plots for B, G, R
            plt.subplot(131)
            plt.hist(self.data[:, :, BLUE].ravel(), bins, [minVal, maxVal])
            plt.subplot(132)
            plt.hist(self.data[:, :, GREEN].ravel(), bins, [minVal, maxVal])
            plt.subplot(133)
            plt.hist(self.data[:, :, RED].ravel(), bins, [minVal, maxVal])
        elif channel == HSV:
            # Draw 3 plots for H, S, V
            hsv = cv2.cvtColor(self.data, cv2.COLOR_BGR2HSV)
            plt.subplot(131)
            plt.hist(hsv[:, :, 0].ravel(), bins, [minVal, maxVal])
            plt.subplot(132)
            plt.hist(hsv[:, :, 1].ravel(), bins, [minVal, maxVal])
            plt.subplot(133)
            plt.hist(hsv[:, :, 2].ravel(), bins, [minVal, maxVal])

        if isSaved:
            plt.savefig(fileName)
        if isShow:
            plt.show()
        plt.clf()

    # Function: Crop Image
    def crop(self, startH=0, startW=0, endH=-1, endW=-1):
        endH = self.h if endH == -1 else endH
        endW = self.w if endW == -1 else endW
        self.data = self.data[startH:endH, startW:endW, :]
        self.h = endH - startH
        self.w = endW - startW

    # Function: Print Image Information
    def print_info(self, name):
        print("\033[1m")
        print("RAW file " + name + ":" + "\033[0;34m")
        print(" Height, Width, Channel: " + str(self.data.shape))
        print(" Data type: " + str(self.dtype) + "\033[33m")
        print(" Min: " + str(np.min(self.data)))
        print(" Max: " + str(np.max(self.data)))
        print(" Mean: " + str(np.mean(self.data)))
        print("\033[0m")

    # Function: Check Underexposure & Overexposure
    def check_exposure(self, threshold):
        overexposure = np.zeros((self.h, self.w, self.ch), dtype=np.uint8)
        overexposure[self.data >= threshold] = 255
        return overexposure

    # Function: Save Image
    def save_image(self, fileName):
        cv2.imwrite(fileName, self.data)
