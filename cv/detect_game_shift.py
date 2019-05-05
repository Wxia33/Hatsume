import cv2
import numpy as np
import tensorflow as tf
from collections import defaultdict


def read_labels(filename='labels.txt'):
    # Read label file
    label_file = open(filename, 'r')
    labels = label_file.read().split()
    label_file.close()
    return labels


def game_change_detect(
        video_name,
        model_path='../tensorflow/vg-classifier-model/vg-classifier-model.meta',
        image_size=256,
        num_channels=3
):
    image_size = 256
    num_channels = 3

    # Read Label File
    labels = read_labels()

    # Read video file
    vidObj = cv2.VideoCapture(video_name)
    success, img = vidObj.read()

    # Start tensorflow session
    sess = tf.Session()
    saver = tf.train.import_meta_graph(model_path)
    saver.restore(sess, tf.train.latest_checkpoint('../tensorflow/vg-classifier-model/'))
    graph = tf.get_default_graph()

    # Moving average of previous 20 frames
    prevClass = []
    frameChange = []
    classDict = defaultdict(lambda: 0)
    prevMajClass = -1

    n = 0
    while success:
        image = []
        img = cv2.resize(img, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
        image.append(img)
        img = np.array(image, dtype=np.uint8)
        img = img.astype('float32')
        img = np.multiply(img, 1.0 / 255.0)
        x_batch = img.reshape(1, image_size, image_size, num_channels)
        y_pred = graph.get_tensor_by_name('y_pred:0')
        x = graph.get_tensor_by_name('x:0')
        y_true = graph.get_tensor_by_name('y_true:0')
        y_test_images = np.zeros((1, len(labels)))
        feed_dict_testing = {x: x_batch, y_true: y_test_images}
        result = sess.run(y_pred, feed_dict=feed_dict_testing)
        res = max(result[0])
        for i, j in enumerate(result[0]):
            if j == res:
                print(labels[i])
                prevClass.append(i)
                classDict[i] += 1
                if len(prevClass) > 20:
                    classDict[prevClass[0]] -= 1
                    del prevClass[0]
                break
        maxClass = max(prevClass, key=classDict.get)
        if prevMajClass == -1:
            prevMajClass = maxClass
        if prevMajClass != maxClass:
            prevMajClass = maxClass
            frameChange.append(n)
        n += 1
        success, img = vidObj.read()
    return frameChange


def main():
    video_name = '../data/videos/blops_and_hstone.mp4'

    frames_changed = game_change_detect(video_name)

    print('Games changed at frames: ', frames_changed)


if __name__ == "__main__":
    main()
