import numpy as np
from PIL import Image
#import tensorflow library
import tensorflow as tf
#import mnist database
from tensorflow.examples.tutorials.mnist import input_data
#assign a variable to the input_data
#data is read using one_hot_encoding to rep labels(actual digits)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#checks the value of the dataset
n_train = mnist.train.num_examples
n_validation = mnist.validation.num_examples
n_test = mnist.test.num_examples
#store the no of units per layer as global variable
#neural network architecture part1
n_input = 784
n_hidden1 = 512
n_hidden2 = 256
n_hidden3 = 128
n_output = 10
#neural network architecture part2
#set the hyperparameters
#how much parameters adjust at each learning step
learning_rate = 1e-4
#how much time spent on each training
n_iterations = 1000
#number of training examples used at each step
batch_size = 128
#variable shows the number of units eliminated randomly
dropout = 0.5
#tensorflow graph
#we set tensors as placeholders
#none represents any amount
X =tf.placeholder("float",[None, n_input])
Y = tf.placeholder("float",[None, n_output])
#controls the droupout rate when traning droupout=0.5 and testing droupout=0.1
keep_prob = tf.placeholder(tf.float32)
#add weight and bias as these values are optimized during training
#important during learning as they activate functions of the neurons and
#represents the strength of the connections between units
weights = {
	"w1": tf.Variable(tf.truncated_normal([n_input, n_hidden1], stddev=0.1)),
	"w2": tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2], stddev=0.1)),
	"w3": tf.Variable(tf.truncated_normal([n_hidden2, n_hidden3], stddev=0.1)),
	"out": tf.Variable(tf.truncated_normal([n_hidden3, n_output], stddev=0.1)),
}
#bias
biases = {
	"b1": tf.Variable(tf.constant(0.1, shape = [n_hidden1])),
	"b2": tf.Variable(tf.constant(0.1, shape = [n_hidden2])),
	"b3": tf.Variable(tf.constant(0.1, shape = [n_hidden3])),
	"out": tf.Variable(tf.constant(0.1, shape = [n_output])),
}
#layers that will define manipulation of the tensors
layer_1 = tf.add(tf.matmul(X, weights["w1"]), biases["b1"])
layer_2 = tf.add(tf.matmul(layer_1, weights["w2"]), biases["b2"])
layer_3 = tf.add(tf.matmul(layer_2, weights["w3"]), biases["b3"])
layer_drop = tf.nn.dropout(layer_3, keep_prob)
output_layer = tf.matmul(layer_3, weights["out"]) + biases["out"]
#loss function using adam opyimizer algorithm
cross_entropy = tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits(
		labels=Y, logits=output_layer

))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#training and testing
#arg_max compares which images are predicted correctly
#looks at the output_layer(predictions) and labels(y)
#equal function returns a list of boolean
correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#initialize a session for training the graph
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
#mini batches training
for i in range(n_iterations):
	batch_x, batch_y = mnist.train.next_batch(batch_size)
	sess.run(train_step, feed_dict={
		X: batch_x, Y:batch_y, keep_prob: dropout
	})
#prints loss and accuracy per minibatch
	if i % 100 == 0:
		minibatch_loss, minibatch_accuracy = sess.run(
			[cross_entropy, accuracy],
			feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0})
		print(
			"Iteration",
			str(i),
			"\t| Loss = ",
			str(minibatch_loss),
			"\t| Acurracy =",
			str(minibatch_accuracy)
			)
#run the session on the testing images
test_accuracy = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0})
print("\nAccuracy on test set:", test_accuracy)
#loads the test images of the hndwritten digits
img = np.invert(Image.open("test_img.png").convert('L')).ravel()
#np.squeeze returns a single element from the array
prediction = sess.run(tf.argmax(output_layer, 1), feed_dict={X: [img]})
print ("Prediction for test image:", np.squeeze(prediction))


