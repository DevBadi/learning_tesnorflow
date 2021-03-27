import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# Initialization => there are many ways to create a Tensor

# x = tf.constant(4, shape=(1, 1), dtype=tf.float32)
# x = tf.constant([[1, 2, 3], [4, 5, 6]])
#
# x = tf.zeros((2, 3))
# print(x)

# do maths operations and casting and indexing
# y = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9])
# print(y[:])
# print(y[1:])
# print(y[1:3])
# print(y[::3])
# # take a value from an indices of an array
# indices = tf.constant([0, 3])
# y_ind = tf.gather(y, indices)
# print(y_ind)
# Reshaping a Tensor
x = tf.range(9)
x = tf.reshape(x, (3, 3))
print(x)