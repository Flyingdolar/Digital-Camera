import RawClass as rc
import numpy as np
import cv2

# MACROS
GRAY = 0
BGR = 1
HSV = 2
BLUE = 0
GREEN = 1
RED = 2

imgW, imgH = 3296, 2472


def cutDyamicRange(ch_L: np.ndarray, lambda_: float, step: float, its: int):
    minL, maxL = np.min(ch_L), np.max(ch_L)
    meanL = (maxL - minL) / 2
    desC, minE = minL, 1e10

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


# Read all 5 RAW files
rawImage = []
for i in range(5):
    rawImage.append(rc.RAWDATA("raw/" + str(i + 1) + ".raw", np.uint16, imgH, imgW, 1))
    print("Read RAW file " + str(i + 1) + " successfully")

# Adjust Image to Avoid Noise
for i in range(5):
    # Remove offset as 512
    rawImage[i].data = rawImage[i].data - 512
    # Clip image to 0~2**14-1
    rawImage[i].data = np.clip(rawImage[i].data, 0, 2**14 - 1)
    # Normalize
    rawImage[i].data = rawImage[i].data / (2**14 - 1) * 65535
    rawImage[i].change_dtype(np.uint16)
    # # Demosaic
    # rawImage[i].demosaic("RG")
    # rawImage[i].change_dtype(np.uint16)
    print("Adjust RAW file " + str(i + 1) + " successfully")
    # Save image
    rawImage[i].save_image("preProcess/OR" + str(i + 1) + ".png")

# Fuse Images
print("Fusing...")
# Using cv2 Fuse
mergeMerten = cv2.createMergeMertens()
imgFused = mergeMerten.process(
    [
        rawImage[0].data,
        rawImage[1].data,
        rawImage[2].data,
        rawImage[3].data,
        rawImage[4].data,
    ]
)
# # Calculate Exposure Ratio of each image
# expRatio = []
# for i in range(5):
#     expRatio.append(np.mean(rawImage[i].data[rawImage[i].data < 2**14 * 0.9]))
# # Fuse
# imgFused = rawImage[4].data
# expRatio = expRatio[4] / expRatio
# print(expRatio)
# for i in range(4, -1, -1):
#     print("...Fusing", i + 1, "...")
#     imgFused = imgFused + rawImage[i].data * expRatio[i]

# Normalize
lmin, lmax = np.min(imgFused), np.max(imgFused)
print(lmin, lmax)
imgFused = (imgFused - lmin) / (lmax - lmin) * 65535
imgFused = imgFused.astype(np.uint16)

# Save image
cv2.imwrite("fused.png", imgFused)

# Demoasic
print("Demosaic...")
# RG to BGR
imgFused = cv2.cvtColor(imgFused, cv2.COLOR_BAYER_RG2BGR)
imgFused = imgFused.astype(np.uint16)

# Save image
cv2.imwrite("demosaic.png", imgFused)

# Auto White Balance
print("Auto White Balance...")
# Calculate mean value of each channel
meanB, meanG, meanR = (
    np.mean(imgFused[:, :, BLUE]),
    np.mean(imgFused[:, :, GREEN]),
    np.mean(imgFused[:, :, RED]),
)
# Calculate gain of each channel
gainB, gainG, gainR = (
    meanG / meanB,
    meanG / meanG,
    meanG / meanR,
)
# Apply gain
imgFused = imgFused.astype(np.float32)
imgFused[:, :, BLUE] *= gainB
imgFused[:, :, GREEN] *= gainG
imgFused[:, :, RED] *= gainR
imgFused = np.clip(imgFused, 0, 65535)
imgFused = imgFused.astype(np.uint16)
cv2.imwrite("awb.png", imgFused)

# Tone Reproduction
print("Tone Reproduction...")
# Change dtype to float32
imgFused = imgFused.astype(np.float32) / 65535
imgLAB = cv2.cvtColor(imgFused, cv2.COLOR_BGR2LAB)
imgL, imgA, imgB = cv2.split(imgLAB)
# Flatten L channel
imgL = imgL.flatten()
minL, maxL = 0, 100
# Divide Dynamic Range
listC = cutDyamicRange(imgL, 0.2, 1, 3)
listC = np.concatenate(([minL], listC, [maxL]))
listC = np.sort(listC)
parts, ptLen = listC.size - 1, (maxL - minL) / listC.size

# Histogram Equalization
print("Histogram Equalization...")
print("List of c values:", listC)
for pdx in range(len(imgL)):
    for cdx in range(len(listC) - 1):
        if listC[cdx] <= imgL[pdx] < listC[cdx + 1]:
            pos = (imgL[pdx] - listC[cdx]) / (listC[cdx + 1] - listC[cdx])
            imgL[pdx] += ptLen * (cdx + pos) - imgL[pdx]
            break

# Color Enhancement
print("Color Enhancement...")
imgA *= 1.5
imgB *= 1.5
# Clip
imgA = np.clip(imgA, -100, 100)
imgB = np.clip(imgB, -100, 100)

imgL = imgL.reshape((imgH, imgW))
imgLAB = cv2.merge((imgL, imgA, imgB))
# Convert FLOAT32 LAB to UINT16 RGB
imgTone = cv2.cvtColor(imgLAB, cv2.COLOR_LAB2BGR)
imgTone *= 65535
imgTone = imgTone.astype(np.uint16)

cv2.imwrite("tone.png", imgTone)

# Gamma Correction
print("Gamma Correction...")
gamma = 1.5
imgGamma = np.power(imgTone / 65535, 1 / gamma) * 65535
imgGamma = imgGamma.astype(np.uint16)

# Save image
cv2.imwrite("gamma.png", imgGamma)
