import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import math

class CNNRawImplementation():

	def __init__(self):
		self.sess = tf.Session()
		self.X = tf.placeholder(tf.float32, [None, 28, 28, 1], name='X')
		self.pkeep = tf.placeholder(tf.float32)

	def train(self):
		mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=False)

		# initializing placeholders and variables
		X = self.X

		# softmax layer
		W = tf.Variable(tf.truncated_normal([200, 10], stddev=0.1))
		b = tf.Variable(tf.ones([10])/10)

		# first convolution layer
		W1 = tf.Variable(tf.truncated_normal([5, 5, 1, 4], stddev=0.1))
		b1 = tf.Variable(tf.ones([4])/10)

		# second convolution layer
		W2 = tf.Variable(tf.truncated_normal([5, 5, 4, 8], stddev=0.1))
		b2 = tf.Variable(tf.ones([8])/10)

		# third convolution layer
		W3 = tf.Variable(tf.truncated_normal([5, 5, 8, 12], stddev=0.1))
		b3 = tf.Variable(tf.ones([12])/10)

		# fully connected layer
		W4 = tf.Variable(tf.truncated_normal([7*7*12, 200], stddev=0.1))
		b4 = tf.Variable(tf.ones([200])/10)

		pkeep = self.pkeep

		# applying first convolution
		Y1conv = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding = "SAME") + b1)

		# applying second convolution
		Y2conv = tf.nn.relu(tf.nn.conv2d(Y1conv, W2, strides=[1, 2, 2, 1],  padding="SAME") + b2)

		# applying third convolution
		Y3conv = tf.nn.relu(tf.nn.conv2d(Y2conv, W3, strides=[1, 2, 2, 1], padding="SAME") + b3)

		# applying fully connected
		Yfully = tf.nn.relu(tf.matmul(tf.reshape(Y3conv, [-1, 7*7*12]), W4) + b4)

		# applying softmax
		Y = tf.nn.softmax(tf.matmul(Yfully, W) + b)

		# placeholder for correct answer
		Y_ = tf.placeholder(tf.float32, [None, 10])

		# loss function
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Y, labels=Y_)
		cross_entropy = tf.reduce_mean(cross_entropy)*100

		# % of correct answers found in batch
		is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
		accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

		lr = tf.placeholder(tf.float32);
		# preparing training data
		optimizer = tf.train.AdamOptimizer(lr)
		train_step = optimizer.minimize(cross_entropy)

		init = tf.global_variables_initializer()

		# initialize tensorflow session
		sess = self.sess
		sess.run(init)

		# learning rate decay
		max_learning_rate = 0.003
		min_learning_rate = 0.0001
		decay_speed = 2000.0 # 0.003-0.0001-2000=>0.9826 done in 5000 iterations

		for i in range(5000):
			learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)

			# load batch of images and correct answers
			batch_X, batch_Y = mnist.train.next_batch(100)
			train_data = {X: batch_X, Y_: batch_Y, lr: learning_rate, pkeep: 0.75}

			# train
			sess.run(train_step, feed_dict=train_data)

			if i%100 == 0:
				a,c = sess.run([accuracy, cross_entropy], feed_dict=train_data)
				print("Train | Step: %5d, Accuracy: %5f, Cost: %10f." % (i, a, c))

				# success on test data ?
				test_data = {X: mnist.test.images, Y_: mnist.test.labels, pkeep: 1}
				a,c = sess.run([accuracy, cross_entropy], feed_dict=test_data)
				print("Test  | Step: %5d, Accuracy: %5f, Cost: %10f." % (i, a, c))

		self.model = tf.argmax(Y, 1)

	def fit(self, img):
		return self.sess.run([self.model], feed_dict={self.X: [img], self.pkeep: 1})[0]

