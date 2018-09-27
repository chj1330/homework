from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

X, c = mnist.train.next_batch(64)

print(c)