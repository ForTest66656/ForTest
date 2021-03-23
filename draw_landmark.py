import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import imageio
import cv2

img = (np.zeros((256, 256, 3))+255).astype(np.float32)
tpts = np.load('data/init_landmark.npy')   
tpts -= 56
tpts *= 1.25
tpts += 56
tpts *= (256 / 112)
tpts = tpts.astype(np.uint8)
for k in range(tpts.shape[0]):
    cv2.circle(img, (tpts[k, 0], tpts[k, 1]), 3, [0, 255, 0], -1)
imageio.imwrite('meanface/300w_68.jpg', img.astype(np.uint8))
print('meanface/300w_68.jpg')

img = (np.zeros((256, 256, 3))+255).astype(np.float32)
index = [0, 2, 4, 6, 8, 10, 12, 14, 16, 17, 19, 21, 22, 24, 26,
        27, 28, 29, 30, 31, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        48, 50, 52, 54, 55, 57, 59, 60, 62, 64, 65, 67]
tpts = tpts[index]
tpts = tpts.astype(np.uint8)
for k in range(tpts.shape[0]):
    cv2.circle(img, (tpts[k, 0], tpts[k, 1]), 3, [0, 255, 0], -1)
imageio.imwrite('meanface/300w_46.jpg', img.astype(np.uint8))
print('meanface/300w_46.jpg')

img = (np.zeros((256, 256, 3))+255).astype(np.float32)
tpts = np.load('data/wflw/init_landmark.npy')
tpts *= (256 / 112)
tpts = tpts.astype(np.uint8)
for k in range(tpts.shape[0]):
    cv2.circle(img, (tpts[k, 0], tpts[k, 1]), 3, [0, 255, 0], -1)
imageio.imwrite('meanface/wflw_98.jpg', img.astype(np.uint8))
print('meanface/wflw_98.jpg')

img = (np.zeros((256, 256, 3))+255).astype(np.float32)
index = list(np.arange(0, 17, 2)) + list(np.arange(18, 33, 2)) + [33, 35, 37, 38, 40] +\
        [42, 44, 46, 48, 50] + list(np.arange(51, 55)) + list(np.arange(55, 60, 2)) +\
        list(np.arange(60, 68, 2)) + list(np.arange(68, 76, 2)) + list(np.arange(76, 83, 2)) +\
        list(np.arange(83, 88, 2)) + list(np.arange(88, 93, 2)) + [93, 95]
tpts = tpts[index]
tpts = tpts.astype(np.uint8)
for k in range(tpts.shape[0]):
    cv2.circle(img, (tpts[k, 0], tpts[k, 1]), 3, [0, 255, 0], -1)
imageio.imwrite('meanface/wflw_54.jpg', img.astype(np.uint8))
print('meanface/wflw_54.jpg')

img = (np.zeros((256, 256, 3))+255).astype(np.float32)
tpts = scipy.io.loadmat('data/300w/images/helen/Helen_meanShape_256_1_5x.mat')['Helen_meanShape_256_1_5x']
tpts = tpts.astype(np.uint8)
for k in range(tpts.shape[0]):
    cv2.circle(img, (tpts[k, 0], tpts[k, 1]), 3, [0, 255, 0], -1)
imageio.imwrite('meanface/helen_194.jpg', img.astype(np.uint8))
print('meanface/helen_194.jpg')

img = (np.zeros((256, 256, 3))+255).astype(np.float32)
index = list(np.arange(0, 21, 2)) + [21] + list(np.arange(24, 41, 2)) + list(np.arange(41, 58, 2)) +\
        list(np.arange(58, 72, 2)) + list(np.arange(72, 86, 2)) + list(np.arange(86, 100, 2)) +\
        list(np.arange(100, 114, 2)) + list(np.arange(114, 134, 2)) + list(np.arange(134, 154, 2)) +\
        list(np.arange(154, 174, 2)) + list(np.arange(174, 194, 2))
tpts1 = tpts[index]
tpts1 = tpts1.astype(np.uint8)
for k in range(tpts1.shape[0]):
    cv2.circle(img, (tpts1[k, 0], tpts1[k, 1]), 3, [0, 255, 0], -1)
imageio.imwrite('meanface/helen_98.jpg', img.astype(np.uint8))
print('meanface/helen_98.jpg')

img = (np.zeros((256, 256, 3))+255).astype(np.float32)
index = list(np.arange(0, 21, 3)) + [20, 21] + list(np.arange(25, 41, 3)) + [41, 44, 47, 49, 51, 54, 57] +\
        [58, 61, 64, 65, 68, 71] + [72, 75, 78, 79, 82, 85] + [86, 89, 92, 93, 96, 99] +\
        [100, 103, 106, 107, 110, 113] + [114, 116, 119, 122, 124, 126, 129, 132] +\
        [134, 136, 139, 142, 144, 146, 149, 152] + [154, 156, 159, 162, 164, 166, 169, 162] +\
        [174, 176, 179, 182, 184, 186, 189, 192]
tpts1 = tpts[index]
tpts1 = tpts1.astype(np.uint8)
for k in range(tpts1.shape[0]):
    cv2.circle(img, (tpts1[k, 0], tpts1[k, 1]), 3, [0, 255, 0], -1)
imageio.imwrite('meanface/helen_78.jpg', img.astype(np.uint8))
print('meanface/helen_78.jpg')
