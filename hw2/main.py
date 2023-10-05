import cv2 as cv
import numpy as np
import os

folderName = "subset"
fileList = []

# Use OS to scan all files in the folder
for file in os.listdir(folderName):
    if file.endswith(".bmp"):
        fileList.append(file)

# Print all files in the folder
print("File List:")
for file in fileList:
    print(file)
