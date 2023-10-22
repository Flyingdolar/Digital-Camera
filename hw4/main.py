import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

inf = 1e10

imgName = "img1"  # image name
img_path = f"Pictures/{imgName}.jpg"  # image path
output_path = f"Res/{imgName}.jpg"  # output path
outHist_path = f"Res/{imgName}Hist.jpg"  # output histogram path
its = 3  # iteration times
lambda_ = 0.2  # lambda
#                set to 0 for linear distribution
#                set to infinity for Histogram Equalization
step = 0.1  # step size
#              set as Percentage of full dynamic range of L channel
#              suggest to be 0.01~0.1


# Functions
# - Find desired c value to cut into two intervals
def cutDyamicRange(ch_L: np.ndarray, lambda_: float, step: float, its: int):
    minL, maxL = np.min(ch_L), np.max(ch_L)
    meanL = (maxL - minL) / 2
    desC, minE = minL, inf

    # Find the best c value
    for cdx in np.arange(minL, maxL, step):
        E1 = ((cdx - meanL) / (maxL - minL)) ** 2
        E2 = ((np.sum(ch_L[ch_L < cdx]) - np.sum(ch_L) / 2) / np.sum(ch_L)) ** 2
        E = E1 + lambda_ * E2
        desC, minE = (cdx, E) if E < minE else (desC, minE)

    # Divide into two intervals for next iteration
    if desC == minL or ch_L[ch_L < desC].size == 0:  # No need to divide
        return np.array([])
    elif desC == maxL or ch_L[ch_L >= desC].size == 0:  # No need to divide
        return np.array([])
    elif its == 1:  # Last iteration
        return np.array([desC])
    else:  # Divide into two intervals
        ch_L1, ch_L2 = ch_L[ch_L < desC], ch_L[ch_L >= desC]
        arrC1 = cutDyamicRange(ch_L1, lambda_, step, its - 1)
        arrC2 = cutDyamicRange(ch_L2, lambda_, step, its - 1)
        # Combine arrC1 & arrC2 & desC as 1D array
        arr = np.concatenate((arrC1, arrC2))
        arr = np.concatenate(([desC], arr))
        arr = np.sort(arr)
        return arr


# - Enhance Saturation
def enhance_SbyV(oriImg, resImg):
    # Convert to HSV
    oriImg = cv.cvtColor(oriImg, cv.COLOR_BGR2HSV)
    resImg = cv.cvtColor(resImg, cv.COLOR_BGR2HSV)

    print("Enhancing Saturation...")
    # Enhance s by delta value of v
    delV = np.float64(resImg[:, :, 2]) - np.float64(oriImg[:, :, 2])
    resS = resImg[:, :, 1] + delV * 0.15
    resImg[:, :, 1] = np.clip(resS, 0, 255)

    # Convert to BGR
    resImg = cv.cvtColor(resImg, cv.COLOR_HSV2BGR)
    return resImg


# Read Image on LAB color space
oriImg = cv.imread(img_path).astype(np.float32) / 255.0
labImg = cv.cvtColor(oriImg, cv.COLOR_BGR2Lab)

# Extract L channel manually
ch_L = labImg[..., 0].astype(np.float64)

# Split LAB channels
_, ch_A, ch_B = cv.split(labImg)
ch_L = ch_L.flatten()

print("Dividing Dynamic Range...")
maxL, minL = np.max(ch_L), np.min(ch_L)
listC = cutDyamicRange(ch_L, lambda_, step, its)
listC = np.concatenate(([minL], listC, [maxL]))
listC = np.sort(listC)
parts, ptLen = len(listC) - 1, (maxL - minL) / (len(listC) - 1)

# Histogram Equalization
print("Histogram Equalization...")
print("List of c values:", listC)
for pdx in range(len(ch_L)):
    for cdx in range(len(listC) - 1):
        if listC[cdx] <= ch_L[pdx] < listC[cdx + 1]:
            pos = (ch_L[pdx] - listC[cdx]) / (listC[cdx + 1] - listC[cdx])
            ch_L[pdx] += ptLen * (cdx + pos) - ch_L[pdx]
            break


# Merge LAB channels
ch_L = ch_L.reshape((oriImg.shape[0], oriImg.shape[1]))
ch_L = ch_L.astype(np.float32)
labImg = cv.merge([ch_L, ch_A, ch_B])  # Merge LAB channels
# Change LAB to BGR
resImg = cv.cvtColor(labImg, cv.COLOR_Lab2BGR)
# Enhance Saturation
resImg = enhance_SbyV(oriImg, resImg)

# Draw Histogram
oriGray = cv.cvtColor(oriImg, cv.COLOR_BGR2GRAY)
resGray = cv.cvtColor(resImg, cv.COLOR_BGR2GRAY)
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.hist(oriGray.flatten(), bins=256, range=(0, 1), density=True)
plt.subplot(122)
plt.hist(resGray.flatten(), bins=256, range=(0, 1), density=True)

# Save Histogram
plt.savefig(outHist_path)

# Show Histogram & Image
cv.imshow("Histogram", plt.imread("tmp/Histogram.jpg"))
cv.imshow("Origin", oriImg)
cv.imshow("Result", resImg)
cv.waitKey(0)

# Save Image
cv.imwrite(output_path, resImg * 255)
