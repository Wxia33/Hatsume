import cv2

image_size = 256
video_name = '../data/videos/2018-11-14-2002-36Collateral.mp4'

frame_q = []

vidObj = cv2.VideoCapture(video_name)
success, img = vidObj.read()
while success:
    img = cv2.resize(img, (image_size, image_size),0,0, cv2.INTER_LINEAR)
    frame_q.append(img)
    success, img = vidObj.read()

# Play frame 120 to 180
del frame_q[0:120]
del frame_q[181 - 100:len(frame_q)]

print(len(frame_q))
for frame in frame_q:
    cv2.imshow('Frame', frame)
    if cv2.waitKey(25) and 0xFF == ord("q"):
        break
