import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

video_name = '../data/videos/2018-11-14-2002-36Collateral.mp4'
#dir_path = os.path.dirname(os.path.realpath(__file__))
#video_name = dir_path +'/' +image_path

image_size = 256

grad_sequence = []

maxsumgrad = 0

success = True
n = 1
cumulative_mean = 0
while success:
    vidObj = cv2.VideoCapture(video_name)
    success, img = vidObj.read()
    if not success:
        break
    img = cv2.resize(img, (image_size, image_size),0,0, cv2.INTER_LINEAR)

    #img = cv2.imread(image)
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    grad_sum = abs(np.sum(laplacian))

    if grad_sum > maxsumgrad:
        maxsumgrad = grad_sum
        maxgrad = img
        maxgrad_laplacian = laplacian
    grad_sequence.append(grad_sum)
    cumulative_mean += grad_sum
    n += 1
    if n == 2300:
        break

cumulative_mean /= n

print cumulative_mean
plt.imshow(maxgrad)
plt.show()
