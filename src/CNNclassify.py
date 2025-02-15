import tensorflow as tf
import time
import math
import sys
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Load the data set

from keras.datasets import cifar10

(xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
xTrain = xTrain.astype(np.float)
yTrain = np.squeeze(yTrain)

yTest = np.squeeze(yTest)
xTest = xTest.astype(np.float)

# Show dimension for each variable

#print ('Train image shape:    {0}'.format(xTrain.shape))
#print ('Train label shape:    {0}'.format(yTrain.shape))
#print ('Test image shape:     {0}'.format(xTest.shape))
#print ('Test label shape:     {0}'.format(yTest.shape))

# Pre processing data
# Normalize the data by subtract the mean image
meanImage = np.mean(xTrain, axis=0)
xTrain -= meanImage
xTest -= meanImage

# set hyperparameter
lr = 5e-4
model_path = os.path.join('model', 'model')
epochs = 10
batchSize = 128



# Create model

x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])

# set 1

conv1_1 = tf.layers.conv2d(inputs=x, filters=32, padding='same', kernel_size=5, strides=1, activation=tf.nn.relu)
conv1_2 = tf.layers.conv2d(inputs=conv1_1, filters=32, padding='same', kernel_size=5, strides=1, activation=tf.nn.relu)
max_pool1 = tf.layers.max_pooling2d(inputs=conv1_2, pool_size=[2, 2], strides=2)
drop_out1 = tf.layers.dropout(max_pool1, 0.5)

# set2


conv2_1 = tf.layers.conv2d(inputs=drop_out1, filters=64, padding='same', kernel_size=5, strides=1, activation=tf.nn.relu)
conv2_2 = tf.layers.conv2d(inputs=conv2_1, filters=64, padding='same', kernel_size=5, strides=1, activation=tf.nn.relu)
max_pool2 = tf.layers.max_pooling2d(inputs=conv2_2, pool_size=[2, 2], strides=2)
drop_out2 = tf.layers.dropout(max_pool2, 0.5)


# layer : 2d convolution with activation function as RELU

conv3_1 = tf.layers.conv2d(inputs=drop_out2, filters=128, padding='same', kernel_size=5, strides=1, activation=tf.nn.relu)
conv3_1 = tf.layers.conv2d(inputs=conv3_1, filters=128, padding='same', kernel_size=5, strides=1, activation=tf.nn.relu)
max_pooling_3 = tf.layers.max_pooling2d(inputs=conv3_1, pool_size=[2, 2], strides=2)
drop_out3 = tf.layers.dropout(max_pooling_3, 0.5)


# flatten to provide as input to fully connected layer
reshape_vector = tf.reshape(max_pooling_3, [-1, 2048])
fully_connected_layer1 = tf.layers.dense(inputs=reshape_vector, units=512, activation=tf.nn.relu)

# batch normalization
batch_normalization = tf.layers.batch_normalization(inputs=fully_connected_layer1, training=True)

# fully connected layer
yOut = tf.layers.dense(inputs=batch_normalization, units=10, activation=None)

# Define Loss
totalLoss = tf.losses.hinge_loss(tf.one_hot(y, 10), logits=yOut)
meanLoss = tf.reduce_mean(totalLoss)

# Define Optimizer
optimizer = tf.train.AdamOptimizer(lr)
trainStep = optimizer.minimize(meanLoss)

# Define correct Prediction and accuracy
correctPrediction = tf.equal(tf.argmax(yOut, 1), y)
accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

def train():

    print("Loop\t\tTrain Loss\t\tTrain Acc %\t\tTest loss\t\tTest Acc %")
    start = time.time()
    # Train

    trainIndex = np.arange(xTrain.shape[0])
    save_model = tf.train.Saver()
    #np.random.shuffle(trainIndex)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):

            # For each batch in training data
            for i in range(int(math.ceil(xTrain.shape[0] / batchSize))):
                # Get the batch data for training
                startIndex = (i * batchSize) % xTrain.shape[0]
                idX = trainIndex[startIndex:startIndex + batchSize]
                train_loss, train_acc, _ = sess.run([meanLoss, accuracy, trainStep], feed_dict={x: xTrain[idX, :], y: yTrain[idX]})
            test_loss, test_acc = sess.run([meanLoss, accuracy], feed_dict={x: xTest, y: yTest})
            print('{0}/{1}\t\t{2:0.6f}\t\t{3:0.6f}\t\t{4:0.6f}\t\t{5:0.6f}'.format(int(e) + 1, epochs, train_loss, train_acc * 100, test_loss, test_acc * 100))
        save_path = save_model.save(sess, model_path)
    end = time.time()
    print("Model saved in file: ", save_path)
    #print("Model Train Time %0.2f seconds" % (end - start))
    sess.close()


labels = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def visualize_filters(units):
    filters = 32
    plt.figure(1, figsize=(20,20))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        plt.axis('off')
        plt.imshow(np.array(units)[0, 0, :, :, i], interpolation="nearest", cmap="gray")
    plt.savefig('CONV_rslt' + '.png')


def test(image_path):
    input_image = cv2.imread(image_path)
    test_image = np.expand_dims(input_image, axis=0)
    if np.size(test_image) != 3072:
        print("Image size is not 32 x 32 x 3")
        return
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    save_model = tf.train.Saver()
    save_model.restore(sess, model_path)
    test_loss = sess.run([yOut], feed_dict={x: test_image})
    print(labels[np.argmax(test_loss)])

    # visualize first layer

    visualize_first_layer = sess.run([conv1_1], feed_dict={x: test_image})
    visualize_filters(visualize_first_layer)
    sess.close()

if sys.argv[1] == "train":
    train()
elif sys.argv[1] == "test" or sys.argv[1] == "predict":
    test(sys.argv[2])
else:
    print("Invalid Syntax")