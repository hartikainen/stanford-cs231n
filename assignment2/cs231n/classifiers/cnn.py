import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

from cs231n.classifiers.fc_net import (
  affine_batchnorm_relu_forward, affine_batchnorm_relu_backward
)

def conv_batchnorm_relu_pool_forward(x, w, b, gamma, beta,
                                     conv_param, pool_param, bn_param):
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)

  an, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)

  s, relu_cache = relu_forward(an)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, bn_cache, relu_cache, pool_cache)
  return out, cache

def conv_batchnorm_relu_pool_backward(dout, cache):
  conv_cache, bn_cache, relu_cache, pool_cache = cache

  ds = max_pool_backward_fast(dout, pool_cache)
  dan = relu_backward(ds, relu_cache)
  da, dgamma, dbeta = spatial_batchnorm_backward(dan, bn_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db, dgamma, dbeta

class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:

  conv - relu - 2x2 max pool - affine - relu - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, use_batchnorm=False):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.use_batchnorm = use_batchnorm
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    self.params['W1'] = weight_scale * np.random.rand(
      num_filters, input_dim[0], filter_size, filter_size)
    self.params['b1'] = np.zeros(num_filters)

    W2_dim = (num_filters * input_dim[1]/2 * input_dim[2]/2, hidden_dim)
    self.params['W2'] = weight_scale * np.random.rand(*W2_dim)
    self.params['b2'] = np.zeros(hidden_dim)

    self.params['W3'] = weight_scale * np.random.rand(hidden_dim, num_classes)
    self.params['b3'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    self.bn_params = {}
    if self.use_batchnorm:
      self.params['gamma1'] = np.ones(num_filters)
      self.params['beta1'] = np.zeros(num_filters)
      self.params['gamma2'] = np.ones(hidden_dim)
      self.params['beta2'] = np.zeros(hidden_dim)

      self.bn_params[1] = { 'mode': 'train' }
      self.bn_params[2] = { 'mode': 'train' }

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    if self.use_batchnorm:
      for key, bn_param in self.bn_params.iteritems():
        bn_param['mode'] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    if self.use_batchnorm:
      gamma1, beta1 = self.params['gamma1'], self.params['beta1']
      bn_param1 = self.bn_params[1]
      crp_out, crp_cache = conv_batchnorm_relu_pool_forward(
        X, W1, b1, gamma1, beta1, conv_param, pool_param, bn_param1)


      gamma2, beta2 = self.params['gamma2'], self.params['beta2']
      bn_param2 = self.bn_params[2]
      ar_out, ar_cache = affine_batchnorm_relu_forward(crp_out, W2, b2,
                                                       gamma2, beta2, bn_param2)
    else:
      crp_out, crp_cache = conv_relu_pool_forward(
        X, W1, b1, conv_param, pool_param)

      ar_out, ar_cache = affine_relu_forward(crp_out, W2, b2)

    scores, scores_cache = affine_forward(ar_out, W3, b3)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if mode == "test":
      return scores

    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dscores = softmax_loss(scores, y)
    loss += 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))

    da, dW3, db3 = affine_backward(dscores, scores_cache)
    dW3 += self.reg * W3

    if self.use_batchnorm:
      dar, dW2, db2, dgamma2, dbeta2 = affine_batchnorm_relu_backward(
        da, ar_cache)
      dcrp, dW1, db1, dgamma1, dbeta1 = conv_batchnorm_relu_pool_backward(
        dar, crp_cache)
    else:
      dar, dW2, db2 = affine_relu_backward(da, ar_cache)
      dcrp, dW1, db1 = conv_relu_pool_backward(dar, crp_cache)

    dW2 += self.reg * W2

    dW1 += self.reg * W1

    grads = {
      "W1": dW1, "b1": db1,
      "W2": dW2, "b2": db2,
      "W3": dW3, "b3": db3
    }

    if self.use_batchnorm:
      grads.update({
        "gamma1": dgamma1, "beta1": dbeta1,
        "gamma2": dgamma2, "beta2": dbeta2
      })
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


pass
