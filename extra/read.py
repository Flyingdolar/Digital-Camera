import rawpy
import imageio

# pip install rawpy
# pip install imageio


# read image
img = rawpy.imread("dataset1/IMG_1161.CR2")

# Turn into CFA (Color Filter Array) image
cfa = img.raw_image_visible

# Save CFA image
imageio.imwrite("dataset1/IMG_1161_cfa.png", cfa)
