import cv2
import numpy as np
import matplotlib.pyplot as plt

import RawClass as rc

imgW, imgH = 3296, 2472
inf = 1e10
lambda_ = 0.5  # lambda
step = 0.1  # step size
its = 4  # iteration times
bgr_to_xyz = np.array(
    [[0.4002, 0.7075, 0.1655], [0.2457, 0.5866, 0.1043], [0.0553, 0.1564, 0.7920]]
)


def rgb_to_lab(imgBGR):
    # Convert RGB to XYZ
    imgXYZ = np.matmul(imgBGR, bgr_to_xyz)
    # Let X/Xn, Y/Yn, Z/Zn be the normalized XYZ values
    Xn, Yn, Zn = 95.047, 100.000, 108.883  # D65

    # Define f(t) = t^(1/3) if t > 0.008856 else 7.787 * t + 16/116
    def f(t):
        return np.where(t > 0.008856, t ** (1 / 3), 7.787 * t + 16 / 116)

    # Calculate L, a, b
    X, Y, Z = imgXYZ[:, :, 0], imgXYZ[:, :, 1], imgXYZ[:, :, 2]
    L = 116 * f(Y / Yn) - 16
    a = 500 * (f(X / Xn) - f(Y / Yn))
    b = 200 * (f(Y / Yn) - f(Z / Zn))
    imgLAB = cv2.merge((L, a, b))
    return imgLAB


def lab_to_rgb(imgLAB):
    # Let X/Xn, Y/Yn, Z/Zn be the normalized XYZ values
    Xn, Yn, Zn = 95.047, 100.000, 108.883  # D65

    # Define f(t) = t^(1/3) if t > 0.008856 else 7.787 * t + 16/116
    def f(t):
        return np.where(t > 0.008856, t ** (1 / 3), 7.787 * t + 16 / 116)

    # Calculate X, Y, Z
    L, a, b = imgLAB[:, :, 0], imgLAB[:, :, 1], imgLAB[:, :, 2]
    Y = Yn * f((L + 16) / 116)
    X = Xn * f((L + 16) / 116 + a / 500)
    Z = Zn * f((L + 16) / 116 - b / 200)
    imgXYZ = cv2.merge((X, Y, Z))

    # Convert XYZ to RGB
    xyz_to_bgr = np.linalg.inv(bgr_to_xyz)
    imgBGR = np.matmul(imgXYZ, xyz_to_bgr)
    return imgBGR


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


# Read 5 raw images to imgList
imgList = []
for i in range(1, 6):
    imgList.append(rc.rawImage("raw/" + str(i) + ".raw", np.uint16, imgH, imgW, 1))
    imgList[i - 1].change_dtype(np.float64)

# ? Save 5 raw images to 5 jpg images
for i in range(5):
    # Normalize
    lmin, lmax = np.min(imgList[i].data), np.max(imgList[i].data)
    saveImg = (imgList[i].data - lmin) / (lmax - lmin) * 255
    saveImg = saveImg.astype(np.uint8)
    # Save
    cv2.imwrite("raw/" + str(i + 1) + ".jpg", saveImg)

# Fuse Images
print("Fusing...")
# mergeMerten = cv2.createMergeMertens()
# imgFused = mergeMerten.process(
#     [
#         imgList[0].data,
#         imgList[1].data,
#         imgList[2].data,
#         imgList[3].data,
#         imgList[4].data,
#     ]
# )
imgFused = imgList[0].data
for i in range(1, 5):
    print("...Fusing", i + 1, "...")
    imgList[i].data = np.log2(imgList[i].data + 1)
    imgFused = np.log2(imgFused + 1)
    rateList = []
    rateList.append(
        np.mean(imgList[i].data[imgList[i].data < 15] / imgFused[imgList[i].data < 15])
    )
    rate = np.mean(rateList)
    print("...rate: ", rate)
    for row in range(imgH):
        for col in range(imgW):
            if imgList[i].data[row][col] < 15:
                imgFused[row][col] = imgList[i].data[row][col]
            else:
                imgFused[row][col] = (
                    rate * imgFused[row][col] * 0.3 + 0.7 * imgList[i].data[row][col]
                )
    imgList[i].data = np.power(2, imgList[i].data) - 1
    imgFused = np.power(2, imgFused) - 1

# Normalize
lmin, lmax = np.min(imgFused), np.max(imgFused)
imgFused = (imgFused - lmin) / (lmax - lmin) * 65025
imgFused = imgFused.astype(np.uint16)

# ? Save fused image
# Normalize
lmin, lmax = np.min(imgFused), np.max(imgFused)
saveImg = (imgFused - lmin) / (lmax - lmin) * 255
saveImg = saveImg.astype(np.uint8)
# Save
cv2.imwrite("fused.jpg", saveImg)

# Demosaic
print("Demosaic...")
imgBGR = cv2.cvtColor(imgFused, cv2.COLOR_BAYER_RG2BGR)

# ? Save demosaiced image
# Normalize
lmin, lmax = np.min(imgBGR), np.max(imgBGR)
saveImg = (imgBGR - lmin) / (lmax - lmin) * 255
saveImg = saveImg.astype(np.uint8)
# Save
cv2.imwrite("demosaic.jpg", saveImg)

# Gray World
print("Gray World...")
# Get the average of each channel
avgCh = np.mean(imgBGR, axis=(0, 1))
# Get the average of the average
avgAvg = np.mean(avgCh)
# Get the ratio of each channel
ratioCh = avgAvg / avgCh
# Multiply the ratio to each channel
imgBGR = imgBGR * ratioCh

# ? Save gray world image
# Normalize
lmin, lmax = np.min(imgBGR), np.max(imgBGR)
saveImg = (imgBGR - lmin) / (lmax - lmin) * 255
saveImg = saveImg.astype(np.uint8)
# Save
cv2.imwrite("grayWorld.jpg", saveImg)


# Tone Reproduction
# Convert UINT16 RGB to FLOAT32 LAB
imgBGR = imgBGR.astype(np.float32) / 65025.0
imgLAB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2LAB)
# Split into L, A, B channels
ch_L, ch_A, ch_B = cv2.split(imgLAB)
# Flatten L channel
ch_L = ch_L.flatten()
minL, maxL = 0, 100
# Divide dynamic range
print("Divide dynamic range...")
listC = cutDyamicRange(ch_L, lambda_, step, its)
listC = np.concatenate(([minL], listC, [maxL]))
listC = np.sort(listC)
parts, ptLen = len(listC) - 1, (maxL - minL) / (len(listC) - 1)

# Draw Histogram of L channel
plt.hist(ch_L, bins=100)
# Draw c values as vertical lines
for cdx in range(len(listC)):
    plt.axvline(listC[cdx], color="r")
# Save Histogram
plt.savefig("originHist.jpg")


# Histogram Equalization
print("Histogram Equalization...")
print("List of c values:", listC)
for pdx in range(len(ch_L)):
    for cdx in range(len(listC) - 1):
        if listC[cdx] <= ch_L[pdx] < listC[cdx + 1]:
            pos = (ch_L[pdx] - listC[cdx]) / (listC[cdx + 1] - listC[cdx])
            ch_L[pdx] += ptLen * (cdx + pos) - ch_L[pdx]
            break

# Draw Histogram of L channel
# Clear the figure
plt.clf()
plt.hist(ch_L, bins=100)
# Draw lines at every 100 / parts interval
for cdx in range(parts):
    plt.axvline(ptLen * (cdx + 1), color="r")

# Enhance Saturation
print("Enhance Saturation...")
ch_A *= 1.3
ch_B *= 1.3
ch_A = np.clip(ch_A, -100, 100)
ch_B = np.clip(ch_B, -100, 100)

# Merge LAB channels
ch_L = ch_L.reshape((imgH, imgW))
imgLAB = cv2.merge((ch_L, ch_A, ch_B))
# Convert FLOAT32 LAB to UINT16 RGB
imgBGR = cv2.cvtColor(imgLAB, cv2.COLOR_LAB2BGR)
imgBGR *= 255
imgBGR = imgBGR.astype(np.uint8)
# print("Convert FLOAT32 LAB to UINT16 RGB...")
print("...L:", np.min(imgLAB[:, :, 0]), np.max(imgLAB[:, :, 0]))
print("...A:", np.min(imgLAB[:, :, 1]), np.max(imgLAB[:, :, 1]))
print("...B:", np.min(imgLAB[:, :, 2]), np.max(imgLAB[:, :, 2]))
# imgBGR = lab_to_rgb(imgLAB)
# # Print the range of B G R channels
print("...B:", np.min(imgBGR[:, :, 0]), np.max(imgBGR[:, :, 0]))
print("...G:", np.min(imgBGR[:, :, 1]), np.max(imgBGR[:, :, 1]))
print("...R:", np.min(imgBGR[:, :, 2]), np.max(imgBGR[:, :, 2]))
# imgBGR = imgBGR.astype(np.uint16)

# ? Save tone reproduction image
# Normalize
lmin, lmax = np.min(imgBGR), np.max(imgBGR)
saveImg = (imgBGR - lmin) / (lmax - lmin) * 255
saveImg = saveImg.astype(np.uint8)
# Save
cv2.imwrite("toneReproduction.jpg", saveImg)


# Gamma Correction
print("Gamma Correction...")
imgBGR = imgBGR.astype(np.float32) / 255
imgBGR = np.power(imgBGR, 1 / 2.2)
imgBGR *= 255
imgBGR = imgBGR.astype(np.uint8)


# Save
print("Saving...")
cv2.imwrite("result.jpg", imgBGR)
cv2.imshow("result", imgBGR)
cv2.waitKey(0)
cv2.destroyAllWindows()
