import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os


# Define Macros Print function for different color
def print_m(str, color):
    if color == "err":
        print("\033[31m" + str + "\033[0m")
    elif color == "info":
        print("\033[3m\033[34m" + str + "\033[0m")
    # elif color == "input":
    #     print("\033[38;5;206m" + str + "\033[0m")
    elif color == "output":
        print("\033[1m\033[32m" + str + "\033[0m")
    else:
        print(str)


# Training Data Setting
folderName = "../hw2/subset"  # The folder name of training images
fileList = []  # The list of training images
colorLevel = 256  # The color level of the image
centWeight = np.array(
    [  # The central weight of the image
        [2, 0, 2, 0, 2],
        [0, 2, 4, 2, 0],
        [2, 4, 16, 4, 2],
        [0, 2, 4, 2, 0],
        [4, 1, 4, 1, 4],
    ]
)
# Predict Data Setting
idealExp = 0.18  # Ideal exposure level
fileID = 40  # The ID of the target image
targetFile = "../hw2/subset/000040.bmp"  # The file name of the target image


def getExposure(img, matrix):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Change img from BGR to Gray
    dotProduct = 0
    imgH, imgW = img.shape  # Get the size of the image
    matH, matW = matrix.shape  # Get the size of the matrix

    # Divide the image as matrix, and calculate the mean of each block
    block = [[0 for w in range(matW)] for h in range(matH)]
    for row in range(matH):
        for col in range(matW):
            stRow, edRow = row * (imgH // matH), (row + 1) * (imgH // matH)
            stCol, edCol = col * (imgW // matW), (col + 1) * (imgW // matW)
            block[row][col] = np.mean(img[stRow:edRow, stCol:edCol])

    # Calculate the dot product of the block and the central weight
    for row in range(matH):
        for col in range(matW):
            dotProduct += block[row][col] * matrix[row][col]

    return dotProduct / np.sum(matrix)


# Use OS to scan all files in the folder
for file in os.listdir(folderName):
    if file.endswith(".bmp"):
        fileList.append(file)

# Sort the file list
fileList.sort()

# The Array that stores the data
expData = np.zeros((len(fileList), 2))
dataString = ""


# Train & Learn the feature of the image
print_m("Start Training...", "info")
for imgID, file in enumerate(fileList):
    img = cv.imread(folderName + "/" + file)  # Read the image
    expValue = getExposure(img, centWeight)  # Get the exposure value

    # Record the data in the array
    expData[imgID][0] = imgID
    expData[imgID][1] = np.log2(expValue)
    dataString += str(imgID) + "," + str(expValue) + "\n"


# Save the train data as csv file
with open("trainData.csv", "w") as f:
    f.write("ID, Exposure Level\n")
    f.write(dataString)
    f.close()

# Print the result
print_m("Training Section Finished", "info")
print_m("Total files: " + str(len(fileList)), "input")
# Find the Linear Regression
print_m("Finding the Linear Regression...", "info")
# Calculate the mean of x and y
meanX = np.mean(expData[:, 0])
meanY = np.mean(expData[:, 1])
# Calculate the slope and intercept
slope = np.sum((expData[:, 0] - meanX) * (expData[:, 1] - meanY)) / np.sum(
    (expData[:, 0] - meanX) ** 2
)
intercept = meanY - slope * meanX
print_m("y = " + str(round(slope, 3)) + "x + " + str(round(intercept, 3)), "output")

# Predict the best image
print_m("Predicting the best image... ( " + targetFile + " )", "info")
img = cv.imread(targetFile)  # Read the image
expValue = np.log2(getExposure(img, centWeight))  # Get the exposure value
idealExp = np.log2(idealExp * colorLevel)  # Change the ideal exposure value to log2
print_m("Current Exp: " + str(round(2**expValue, 3)), "input")
print_m("Ideal Exp: " + str(round(2**idealExp, 3)), "input")
print()
idealID = ((idealExp - expValue) / slope) + fileID
print_m("The best image is " + str(int(idealID)) + ".bmp", "output")

# Draw the graph
print_m("Drawing the graph of the Linear Regression...", "info")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Draw the exponential version
ax1.plot(expData[:, 0], 2 ** (expData[:, 1]))
ax1.plot(expData[:, 0], 2 ** (slope * expData[:, 0] + intercept))
ax1.scatter(fileID, 2 ** (expValue), color="red")
ax1.scatter(idealID, 2 ** (idealExp), color="green")
ax1.set_xlabel("Exposure Level")
ax1.set_ylabel("Brightness")
ax1.set_title("Auto Exposure (Exponential)")

# Draw the normal version
ax2.plot(expData[:, 0], expData[:, 1])
ax2.plot(expData[:, 0], slope * expData[:, 0] + intercept)
ax2.scatter(fileID, expValue, color="red")
ax2.scatter(idealID, idealExp, color="green")
ax2.set_xlabel("Exposure Level")
ax2.set_ylabel("Brightness (log2)")
ax2.set_title("Auto Exposure (Normal)")

plt.show()
