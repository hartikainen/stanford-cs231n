import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.

  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """

  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg

    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################
    self.params["W1"] = np.random.randn(input_dim, hidden_dim) * weight_scale
    self.params["W2"] = np.random.randn(hidden_dim, num_classes) * weight_scale
    self.params["b1"] = np.zeros(hidden_dim)
    self.params["b2"] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']

    h1, h1_cache = affine_relu_forward(X, W1, b1)
    scores, scores_cache = affine_forward(h1, W2, b2)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    loss, dscores = softmax_loss(scores, y)
    loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))

    dh1, dW2, db2 = affine_backward(dscores, scores_cache)
    dW2 += self.reg * W2
    dX, dW1, db1 = affine_relu_backward(dh1, h1_cache)
    dW1 += self.reg * W1

    grads = {
      "W1": dW1, "b1": db1,
      "W2": dW2, "b2": db2
    }
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads

def W_key(layer_idx):
  return "W" + str(layer_idx)


def b_key(layer_idx):
  return "b" + str(layer_idx)


def gamma_key(layer_idx):
  return "gamma" + str(layer_idx)


def beta_key(layer_idx):
  return "beta" + str(layer_idx)


def affine_batchnorm_relu_forward(x, w, b, gamma, beta, bn_param):
  """
  Forward pass for the affine-batchnorm-relu convenience layer
  """
  fc, fc_cache = affine_forward(x, w, b)
  bn, bn_cache = batchnorm_forward(fc, gamma, beta, bn_param)
  out, relu_cache = relu_forward(bn)
  cache = (fc_cache, bn_cache, relu_cache)

  return out, cache


def affine_batchnorm_relu_backward(dout, cache):
  """
  Backward pass for the affine-batchnorm-relu convenience layer
  """
  fc_cache, bn_cache, relu_cache = cache
  drelu = relu_backward(dout, relu_cache)
  dbn, dgamma, dbeta = batchnorm_backward_alt(drelu, bn_cache)
  dx, dw, db = affine_backward(dbn, fc_cache)

  return dx, dw, db, dbn, dgamma, dbeta

class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be

  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.

  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.

    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    self.layers = []

    for l in xrange(self.num_layers-1):
      if self.use_batchnorm:
        layer = ["affine_batchnorm_relu"]
      else:
        layer = ["affine_relu"]

      if self.use_dropout:
        layer.append("dropout")

      self.layers.append(layer)

    self.layers.append(["affine"])

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    dims = [input_dim] + hidden_dims + [num_classes]
    for l in xrange(1, len(dims)):
      n1, n2 = dims[l-1:l+1]
      self.params[W_key(l)] = np.random.randn(n1, n2) * weight_scale
      self.params[b_key(l)] = np.zeros(n2)

    if self.use_batchnorm:
      for l in xrange(1, len(dims)-1):
        n = dims[l]
        self.params[gamma_key(l)] = np.ones(n)
        self.params[beta_key(l)] = np.zeros(n)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed

    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]

    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    cache = {}

    out = X
    for l, layer in enumerate(self.layers[:-1], 1):

      layer_cache = []

      for k, layer_type in enumerate(layer):

        if layer_type == "affine_batchnorm_relu":
          W, b = self.params[W_key(l)], self.params[b_key(l)]
          gamma, beta = self.params[gamma_key(l)], self.params[beta_key(l)]
          bn_param = self.bn_params[l-1]
          out, out_cache = affine_batchnorm_relu_forward(out, W, b,
                                                         gamma, beta, bn_param)

        elif layer_type == "affine_relu":
          W, b = self.params[W_key(l)], self.params[b_key(l)]
          out, out_cache = affine_relu_forward(out, W, b)

        elif layer_type == "dropout":
          out, out_cache = dropout_forward(out, self.dropout_param)

        else:
          print("WARNING: No layer found", layer_type)

        layer_cache.append(out_cache)

      cache[l] = layer_cache

    l = self.num_layers
    W, b = self.params[W_key(l)], self.params[b_key(l)]
    scores, scores_cache = affine_forward(out, W, b)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    loss, dscores = softmax_loss(scores, y)
    weight_sums = np.sum(
      np.sum(self.params[W_key(l)] ** 2) for l in xrange(1, self.num_layers+1)
    )
    loss += 0.5 * self.reg * (weight_sums)

    l = self.num_layers
    dout, dW, db = affine_backward(dscores, scores_cache)
    dW += self.reg * self.params[W_key(l)]
    grads.update({ W_key(l): dW, b_key(l): db })

    for l, layer in reversed(list(enumerate(self.layers[:-1], 1))):

      layer_cache = cache[l]

      for k, layer_type in reversed(list(enumerate(layer))):

        if layer_type == "affine_batchnorm_relu":
          dout, dW, db, dbn, dgamma, dbeta = affine_batchnorm_relu_backward(
            dout, layer_cache[k]
          )
          dW += self.reg * self.params[W_key(l)]
          grads.update({
            W_key(l): dW, b_key(l): db,
            gamma_key(l): dgamma, beta_key(l): dbeta
          })

        elif layer_type == "affine_relu":
          dout, dW, db = affine_relu_backward(dout, layer_cache[k])
          dW += self.reg * self.params[W_key(l)]
          grads.update({ W_key(l): dW, b_key(l): db })

        elif layer_type == "dropout":
          dout = dropout_backward(dout, layer_cache[k])

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
