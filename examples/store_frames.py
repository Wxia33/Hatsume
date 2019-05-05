import cv2

video_name = '../data/videos/2018-11-14-2002-36Collateral.mp4'

frame_q = []

vidObj = cv2.VideoCapture(video_name)
success, img = vidObj.read()
while success:
    frame_q.append(img)
    success, img = vidObj.read()

print(frame_q)
