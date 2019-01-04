import os
import cv2
import tensorflow as tf
import numpy as np

path = raw_input('What directory would you like me to clean up? ')

vidFiles = []
for f in os.listdir(path):
    if not f.startswith('.'):
        vidFiles.append(f)

print('Cleaning...')


image_size = 256
num_channels = 3

print('Retrieving Labels...')
labelFile = open('labels.txt','r')
labels = labelFile.read().split()
labelFile.close()

print('Reading Frames extracted from video...')
print(vidFiles)

for vid in vidFiles:
    vidObj = cv2.VideoCapture(path + vid)
    success, image = vidObj.read()
    #dir_path = os.path.dirname(os.path.realpath(__file__))
    #image_path=sys.argv[1]
    #filename = dir_path +'/' +image_path
    images = []
    image_size=256
    num_channels=3
    #image = cv2.imread(filename)
    image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
    images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = np.multiply(images, 1.0/255.0)

    print('Retrieving Classifier Model...')

    x_batch = images.reshape(1, image_size,image_size,num_channels)

    model_path = '../tensorflow/vg-classifier-model/vg-classifier-model.meta'
    sess = tf.Session()
    saver = tf.train.import_meta_graph(model_path)
    saver.restore(sess, tf.train.latest_checkpoint('../tensorflow/vg-classifier-model/'))
    graph = tf.get_default_graph()
    y_pred = graph.get_tensor_by_name('y_pred:0')
    x = graph.get_tensor_by_name('x:0')
    y_true = graph.get_tensor_by_name('y_true:0')
    y_test_images = np.zeros((1, 3))

    feed_dict_testing = {x: x_batch, y_true: y_test_images}

    result = sess.run(y_pred, feed_dict = feed_dict_testing)
    res = max(result[0])
    for i,j in enumerate(result[0]):
        if j == res:
            print vid, labels[i]
            break



print('Cleaned!')
