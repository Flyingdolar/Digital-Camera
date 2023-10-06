import cv2 as cv
import numpy as np
import os

folderName = "subset"  # The folder name of the images
fileList = []  # The list of the images
mindist = 10000  # The minimum distance
minfile = ""  # The file name of the best image
exposure = 0.18  # The exposure time of the best image
colorLevel = 256  # The color level of the image
centWeight = [  # The central weight of the image
    [2, 0, 2, 0, 2],
    [0, 2, 4, 2, 0],
    [0, 4, 16, 4, 0],
    [0, 2, 4, 2, 0],
    [2, 1, 2, 1, 2],
]

# Use OS to scan all files in the folder
for file in os.listdir(folderName):
    if file.endswith(".bmp"):
        fileList.append(file)

print("Process Start")
print("Using Central Weighting find the best image with exposure " + str(exposure))
print("Reading files...")
print("Total files: " + str(len(fileList)))

# Print all files in the folder
print("File List:")
for file in fileList:
    img = cv.imread(folderName + "/" + file)  # Read the image
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Change img from BGR to Gray
    height, width = img.shape  # Get the size of the image

    # Divide the image into 25 blocks, and calculate the mean of each block
    block = [[0 for i in range(5)] for j in range(5)]
    for i in range(5):
        for j in range(5):
            block[i][j] = np.mean(
                img[
                    int(height / 5 * i) : int(height / 5 * (i + 1)),
                    int(width / 5 * j) : int(width / 5 * (j + 1)),
                ]
            )

    # Calculate the dot product of the block and the central weight
    dotProduct = 0
    for i in range(5):
        for j in range(5):
            dotProduct += block[i][j] * centWeight[i][j]
    dotProduct /= 64
    dist = dotProduct - int(exposure * colorLevel)
    print(
        "Reading "
        + file
        + "..."
        + "bright = "
        + str(dotProduct)  # convert dotProduct to string
        + ", dist = "
        + str(dist)
        + "..."
    )
    dist = abs(dist)
    # Record if it is min-dist
    if dist < mindist:
        mindist = dist
        minfile = file

# Print the result
print("Process Finished")
print("The best image is " + minfile + " with distance " + str(mindist))
