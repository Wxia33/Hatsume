import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

video_name = '../data/videos/2018-11-14-2002-36Collateral.mp4'
# video 2018-11-14-2002-36Collateral.mp4
# Total Frames: 244

video_name = '../data/videos/2018-11-14-1951-49_Trim.mp4'
video_name = '../data/videos/hs-game-ipad-1.mp4'
#dir_path = os.path.dirname(os.path.realpath(__file__))
#video_name = dir_path +'/' +image_path

image_size = 256

grad_sequence = []

maxsumgrad = 0
topN = 6

success = True
cumulative_mean = 0
vidObj = cv2.VideoCapture(video_name)
success, img = vidObj.read()

img_list = []
while success:
    img = cv2.resize(img, (image_size, image_size),0,0, cv2.INTER_LINEAR)
    img_list.append(img)

    #img = cv2.imread(image)
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    grad_sum = abs(np.sum(laplacian))

    grad_sequence.append(grad_sum)
    cumulative_mean += grad_sum

    success, img = vidObj.read()

cumulative_mean /= len(grad_sequence)

# Print number of frames
print len(grad_sequence)

# Find sub-sequence with largest sum
dplist = grad_sequence - cumulative_mean

# Using variation on Kadane's Algorithm
max_sum = 0
max_list = [-1]
max_temp = []
max_until_here = 0

for (ind, energy) in enumerate(dplist):
    max_until_here += energy
    max_temp.append(ind)
    if (max_until_here < 0):
        max_until_here = 0
        max_temp = []
    if (max_sum < max_until_here):
        max_sum = max_until_here
        #print max_list[-1],max_temp[0]
        if max_list[-1] == max_temp[0] - 1:
            max_list += max_temp
            #print 'append'
        else:
            max_list = max_temp
            #print 'new arr'
        max_temp = []

#print max_list, max_sum

# Show Max Contiguous Energy Frames
for frame_ind in max_list:
    cv2.imshow('Frame', img_list[frame_ind])
    if cv2.waitKey(25) and 0xFF == ord("q"):
        break
