import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

video_name = '../data/videos/2018-11-14-2002-36Collateral.mp4'

# video 2018-11-14-2002-36Collateral.mp4
# Total Frames: 244

#dir_path = os.path.dirname(os.path.realpath(__file__))
#video_name = dir_path +'/' +image_path

image_size = 256

grad_sequence = []

maxsumgrad = 0
maxgradList = []
topN = 6

success = True
n = 1
cumulative_mean = 0
vidObj = cv2.VideoCapture(video_name)
success, img = vidObj.read()
while success:
    img = cv2.resize(img, (image_size, image_size),0,0, cv2.INTER_LINEAR)

    #img = cv2.imread(image)
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    grad_sum = abs(np.sum(laplacian))
    print grad_sum, maxsumgrad
    if grad_sum > maxsumgrad:
        maxsumgrad = grad_sum
        maxgradList.append(img)
        maxgrad_laplacian = laplacian
        if len(maxgradList) > topN:
            del maxgradList[0]
    grad_sequence.append(grad_sum)
    cumulative_mean += grad_sum
    n += 1
    if n == 2300:
        break
    success, img = vidObj.read()
print n

cumulative_mean /= n

print cumulative_mean

print len(maxgradList)
for im in maxgradList:
    plt.imshow(im)
    plt.show()
