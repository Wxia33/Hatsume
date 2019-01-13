import tflearn
import pickle
import tensorflow as tf
import numpy as np

filename = ''

# One-layer LSTM
def build_rnn(frames, input_size, num_classes):
    net = tflearn.input_data(shape=[None, frames, input_size])
    net = tflearn.lstm(net, 256, dropout=0.2)
    net = tflearn.fully_connected(net, num_classes, activation='softmax')
    net = tflearn.regression(net, optimizer='adam',
                             loss='categorical_crossentropy', name='output1')
    return net

net = build_rnn(,3)

# Train the model.
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(X_train, y_train, validation_set=(X_test, y_test),
          show_metric=True, batch_size=batch_size, snapshot_step=100,
          n_epoch=4)

# Save it.
model.save('checkpoints/rnn.tflearn')

# Get our model.
model = tflearn.DNN(net, tensorboard_verbose=0)
model.load('checkpoints/rnn.tflearn')

# Evaluate. Note that although we're passing in "train" data,
# this is actually our holdout dataset, so we never actually
# used it to train the model. Bad variable semantics.
print(model.evaluate(X_train, y_train))
