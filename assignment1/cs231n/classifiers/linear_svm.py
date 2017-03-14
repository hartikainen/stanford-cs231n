import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg, delta=1.0):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0

  for i in xrange(num_train):
    y_ = y[i]
    scores = X[i].dot(W)
    correct_class_score = scores[y_]
    for j in xrange(num_classes):
      if j == y_:
        continue
      margin = scores[j] - correct_class_score + delta
      if margin > 0:
        loss += margin
        dW[:,j]  += X[i]
        dW[:,y_] -= X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= float(num_train)
  dW   /= float(num_train)

  # Add L2 regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW   += reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg, delta=1.0):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = X.shape[0]

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = np.dot(X, W)
  correct_class_scores = np.choose(y, scores.T).reshape(-1, 1)

  hinge_difference = np.maximum(scores - correct_class_scores + delta, 0.0)
  # We currently have num_train * delta error in our matrix, since we should
  # not add delta to the correct class indices. For all i=[0..num_train],
  # j=y[i] set hinge_difference[i,j] = 0.0.
  hinge_difference[np.arange(num_train), y] = 0.0

  # Right now the loss is a matrix of all training examples. We want it to be
  # scalar average instead, so we sum and we divide by num_train.
  loss = np.sum(hinge_difference) / float(num_train)

  # Add L2 regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  grad_mask = (hinge_difference > 0).astype(int)
  grad_mask[np.arange(y.shape[0]), y] = - np.sum(grad_mask, axis=1)
  dW = np.dot(X.T, grad_mask)

  dW /= float(num_train)
  dW += reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
