import cv2
import numpy as np

import RawClass as rc


def fuse(imgLow, imgHigh):
    imgLow.change_dtype(np.float32)
    imgHigh.change_dtype(np.float32)
    # Change to log2
    imgLow.data = np.log2(imgLow.data + 1)
    imgHigh.data = np.log2(imgHigh.data + 1)
    # Get the ratio where imgHigh is < 64256, imgLow is > 2
    rateList = np.mean(imgHigh.data) / np.mean(imgLow.data)
    rate = np.mean(rateList)
    print("rate:", rate)
    # Fuse the two images
    imgFused = rc.rawImage("", np.float32, imgLow.h, imgLow.w, imgLow.ch)
    # If the imgHigh is too bright(>64256), use imgLow * ratio
    imgFused.data[imgHigh.data >= 16] = imgLow.data[imgHigh.data >= 16] * rate
    # Recover to 2^x
    imgFused.data = np.power(2, imgFused.data) - 1

    # Return the fused image
    return imgFused
