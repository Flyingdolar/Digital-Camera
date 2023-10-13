import cv2 as cv
import numpy as np

# Read best exposure image
bestExp = cv.imread("bestExp.bmp")

# Method 1: Apply Gray World Algorithm

# 1 - Find the average of each channel
avgB = np.mean(bestExp[:, :, 0])
avgG = np.mean(bestExp[:, :, 1])
avgR = np.mean(bestExp[:, :, 2])

# 2 - Calculate the gain of each channel
avg = (avgB + avgG + avgR) / 3
gainB = avg / avgB
gainG = avg / avgG
gainR = avg / avgR

# 3 - apply the gain to the image
bestExp[:, :, 0] = bestExp[:, :, 0] * gainB
bestExp[:, :, 1] = bestExp[:, :, 1] * gainG
bestExp[:, :, 2] = bestExp[:, :, 2] * gainR

# 4 - Clip the value to 255
bestExp = np.clip(bestExp, 0, 255)

# Save & Show the image
cv.imwrite("grayWorld.bmp", bestExp)
print('Process Finished, "grayWorld.bmp" is created.')

# Method 2: Apply White Patch Algorithm

# 1 - Find the max of each channel
maxB = np.max(bestExp[:, :, 0])
maxG = np.max(bestExp[:, :, 1])
maxR = np.max(bestExp[:, :, 2])

# 2 - Calculate the gain of each channel
gainB = 255 / maxB
gainG = 255 / maxG
gainR = 255 / maxR

# 3 - apply the gain to the image
bestExp[:, :, 0] = bestExp[:, :, 0] * gainB
bestExp[:, :, 1] = bestExp[:, :, 1] * gainG
bestExp[:, :, 2] = bestExp[:, :, 2] * gainR

# 4 - Clip the value to 255
bestExp = np.clip(bestExp, 0, 255)

# Save & Show the image
cv.imwrite("whitePatch.bmp", bestExp)
print('Process Finished, "whitePatch.bmp" is created.')


# Method 3: Find edge and apply gray world only on the edge

# 1 - Find the edge of the image
bestExpGray = cv.cvtColor(bestExp, cv.COLOR_BGR2GRAY)
bestExpGray = cv.GaussianBlur(bestExpGray, (3, 3), 0)
bestExpEdge = cv.Canny(bestExpGray, 50, 150)

# Show the edge image
cv.imwrite("edge.bmp", bestExpEdge)
cv.imshow("Edge", bestExpEdge)
cv.waitKey(0)

# 2 - Find the average of each channel on the edge
sampleImg = cv.imread("bestExp.bmp")
pixelCount = 0
for row in range(bestExpEdge.shape[0]):
    for col in range(bestExpEdge.shape[1]):
        if bestExpEdge[row, col] != 0:
            avgB += sampleImg[row, col, 0]
            avgG += sampleImg[row, col, 1]
            avgR += sampleImg[row, col, 2]
            pixelCount += 1

avgB /= pixelCount
avgG /= pixelCount
avgR /= pixelCount

# 3 - Calculate the gain of each channel
avg = (avgB + avgG + avgR) / 3
gainB = avg / avgB
gainG = avg / avgG
gainR = avg / avgR

# 4 - apply the gain to the image
sampleImg[:, :, 0] = sampleImg[:, :, 0] * gainB
sampleImg[:, :, 1] = sampleImg[:, :, 1] * gainG
sampleImg[:, :, 2] = sampleImg[:, :, 2] * gainR

# 5 - Clip the value to 255
sampleImg = np.clip(sampleImg, 0, 255)

# Save & Show the image
cv.imwrite("edgeGrayWorld.bmp", sampleImg)
print('Process Finished, "edgeGrayWorld.bmp" is created.')
cv.imshow("Edge Gray World", sampleImg)
cv.waitKey(0)
