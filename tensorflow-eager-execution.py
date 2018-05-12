from __future__ import absolute_import, division, print_function

import tensorflow as tf

tf.enable_eager_execution()

tf.executing_eagerly()        # => True

x = [[2.]]
m = tf.matmul(x, x)  # matrix multiplication
# m = x * x // can't multiply sequence by non-int of type 'list'
print("hello, {}".format(m))  # => "hello, [[4.]]"

a = tf.constant([[1, 2],
                 [3, 4]])
print(a)
# => tf.Tensor([[1 2]   tf operation returns a tensor
#               [3 4]], shape=(2, 2), dtype=int32)

# Broadcasting support
b = tf.add(a, 1)
print(b)
# => tf.Tensor([[2 3]  tf operation returns a tensor
#               [4 5]], shape=(2, 2), dtype=int32)

# Operator overloading is supported
print(a * b)
# => tf.Tensor([[ 2  6]  element-wise multiplication
#               [12 20]], shape=(2, 2), dtype=int32)
# tensor operations return a tensor, the * here is a tensor operation, * is overloaded


print(tf.multiply(a, b))
# element-wise multiplication
# non overloading version, save as the above *


print(tf.matmul(a, b))
#[[10 13]
# [22 29]], shape=(2, 2), dtype=int32)
# matrix multiplication

# Use NumPy values
import numpy as np

c = np.multiply(a, b)
print(c)
# => [[ 2  6]
#     [12 20]]
# numpy operation always return numpy array

# Obtain numpy value from a tensor:
print(a.numpy())
# => [[1 2]
#     [3 4]]
# a is a tensor, however a tensor's numpy() method always returns a numpy array

# Keras, https://keras.io/
# Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow,
# CNTK, or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea
# to result with the least possible delay is key to doing good research.
# Check here to understand, Writing your own Keras layers https://keras.io/layers/writing-your-own-keras-layers/

