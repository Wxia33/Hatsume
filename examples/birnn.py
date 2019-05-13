import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import sklearn.model_selection as sk

X_train = 2
Y_train = 2
X_val = 2
Y_val = 2

hm_epochs = 2
n_classes = 2
batch_size = 2
batch_size_val = 2
chunk_size = 2
n_chunks = 2
rnn_size = 2

with tf.name_scope('Inputs'):
    x = tf.placeholder('float', [None, None, chunk_size], name='Features')
    y = tf.placeholder('float', name='Labels')


def bi_directional_lstm(x):
    x = tf.unstack(x, n_chunks, 1)

    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True)
    lstm_fw_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)

    lstm_cell_3 = tf.contrib.rnn.BasicLSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_4 = tf.contrib.rnn.BasicLSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True)
    lstm_bw_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell_3, lstm_cell_4], state_is_tuple=True)

    weights = tf.Variable(tf.random_normal([2 * rnn_size, n_classes]), name='weights1')

    biases = tf.Variable(tf.random_normal([n_classes]), name='biases1')
    return tf.matmul(x[-1], weights) + biases


def train_bilstm(x):
    prediction = bi_directional_lstm(x)

    best_accurracy = 0.0

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits
                          (logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    with tf.Session() as sess:
        tf.device('/gpu:0')
        sess.run(tf.global_variables_initializer())

        kk = 0
        for epoch in range(hm_epochs):
            epoch_loss = 0
            valdd = []
            k = 0
            for _ in range(int(trainSamples / batch_size)):
                epoch_x = X_train[k:k + batch_size, :]
                epoch_y = Y_train[k:k + batch_size, :]
                epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))
        _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
        epoch_loss += c
        k = k + batch_size
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
    loss.append(epoch_loss)
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    accuracy_out = (accuracy.eval({x: X_Validation.reshape((-1, n_chunks, chunk_size)), y: Y_Validation}))
    Val_Accuracy.append(accuracy_out)
    print('Validation Accuracy : ', accuracy_out, '  ||| Best Accuracy :', best_accuracy)
    return

if __name__ == '__main__':
    train_bilstm(x)
