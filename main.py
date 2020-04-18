from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    batch_size,dims = X.shape
    dims,classes = W.shape

    for batch in range(batch_size):
        exp_scores = np.exp(np.dot(X[batch,:],W))
        prob_scores = exp_scores/np.sum(exp_scores)

        for dim in range(dims):
            for c in range(classes):
                if y[batch] == c:
                    dW[dim,c] += X[batch,dim] * (prob_scores[c] - 1)
                else:
                    dW[dim,c] += X[batch,dim] * prob_scores[c]

        loss += -np.log(prob_scores[y[batch]])
    loss = loss/batch_size + 0.5 * reg * np.sum(W**2)

    dW = dW/batch_size + reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    batch_size,dims = X.shape
    dims,classes = W.shape

    exp_scores = np.exp(np.matmul(X,W))
    prob_scores = exp_scores/np.sum(exp_scores,axis = 1,keepdims= True)
    loss = np.sum(-np.log(prob_scores[range(batch_size),y]))/batch_size + 0.5 * reg * np.sum(W**2)

    #gradient
    prob_scores[range(batch_size),y] -= 1
    dW = np.dot(X.T, prob_scores)/batch_size + reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

