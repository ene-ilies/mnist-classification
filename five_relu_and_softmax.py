import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import math

class FiveReLUAndSoftmax():

	def __init__(self):
		self.sess = tf.Session()
		self.X = tf.placeholder(tf.float32, [None, 28, 28, 1], name='X')

	def train(self):
		mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=False)

		# initializing placeholders and variables
		X = self.X
		W = tf.Variable(tf.truncated_normal([30, 10], stddev=0.1))
		b = tf.Variable(tf.ones([10])/10)

		W1 = tf.Variable(tf.truncated_normal([784, 200], stddev=0.1))
		b1 = tf.Variable(tf.ones([200])/10)

		W2 = tf.Variable(tf.truncated_normal([200, 100], stddev=0.1))
		b2 = tf.Variable(tf.ones([100])/10)

		W3 = tf.Variable(tf.truncated_normal([100, 60], stddev=0.1))
		b3 = tf.Variable(tf.ones([60])/10)

		W4 = tf.Variable(tf.truncated_normal([60, 30], stddev=0.1))
		b4 = tf.Variable(tf.ones([30])/10)

		# creating the model
		XX = tf.reshape(X, [-1, 784])
		Y1 = tf.nn.relu(tf.matmul(XX, W1) + b1)
		Y2 = tf.nn.relu(tf.matmul(Y1, W2) + b2)
		Y3 = tf.nn.relu(tf.matmul(Y2, W3) + b3)
		Y4 = tf.nn.relu(tf.matmul(Y3, W4) + b4)

		Y = tf.nn.softmax(tf.matmul(Y4, W) + b)

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
			train_data = {X: batch_X, Y_: batch_Y, lr: learning_rate}

			# train
			sess.run(train_step, feed_dict=train_data)

			a,c = sess.run([accuracy, cross_entropy], feed_dict=train_data)
			print("Train | Step: %5d, Accuracy: %5f, Cost: %10f." % (i, a, c))

			# success on test data ?
			test_data = {X: mnist.test.images, Y_: mnist.test.labels}
			a,c = sess.run([accuracy, cross_entropy], feed_dict=test_data)
			print("Test  | Step: %5d, Accuracy: %5f, Cost: %10f." % (i, a, c))

		self.model = tf.argmax(Y, 1)

	def fit(self, img):
		return self.sess.run([self.model], feed_dict={self.X: [img]})[0]

