import cv2 as cv
import numpy as np

images = np.zeros((9, 256, 256), dtype=np.uint8)

for i in range(9):
    images[i] = cv.imread(f'RS0030_yz_0625_0_0_0_{i}.png', 0)

mean = np.mean(images, axis=0)
mean = cv.blur(mean, (5, 5))

cv.imwrite('test.png', mean)