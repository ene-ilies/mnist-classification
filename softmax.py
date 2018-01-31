import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class MNISTSoftmax():

	def __init__(self):
		self.sess = tf.Session()
		self.X = tf.placeholder(tf.float32, [None, 28, 28, 1], name='X')

	def train(self):
		mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=False)

		# initializing placeholders and variables
		X = self.X
		W = tf.Variable(tf.zeros([784, 10]))
		b = tf.Variable(tf.zeros([10]))

		init = tf.global_variables_initializer()

		# creating the model
		Y = tf.nn.softmax(tf.matmul(tf.reshape(X, [-1, 784]), W) + b)

		# placeholder for correct answer
		Y_ = tf.placeholder(tf.float32, [None, 10])

		# loss function
		cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))

		# % of correct answers found in batch
		is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
		accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

		# preparing training data
		optimizer = tf.train.GradientDescentOptimizer(0.003)
		train_step = optimizer.minimize(cross_entropy)

		# initialize tensorflow session
		sess = self.sess
		sess.run(init)

		for i in range(1000):
			# load batch of images and correct answers
			batch_X, batch_Y = mnist.train.next_batch(100)
			train_data = {X: batch_X, Y_: batch_Y}

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

