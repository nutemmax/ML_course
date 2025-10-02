# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Stochastic Gradient Descent
"""
from helpers import batch_iter
from costs import compute_loss


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from a data sample batch of size B, where B < N, and their corresponding labels.

    Args:
        y: numpy array of shape=(B, )
        tx: numpy array of shape=(B,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        A numpy array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """
    # same as before, gradient computation, just on minibatch of size b = len(y)
    e = y - tx @ w
    N = y.shape[0]
    grad_L = -1/N * tx.T @ e
    return grad_L


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma, loss_type = "MSE"):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
        loss_type (optional) : string, 'MSE' or 'MAE', denoting the type of loss we compute over the dataset

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """

    # Define parameters to store w and loss
    ws = [initial_w]
    losses = [compute_loss(y, tx, initial_w, loss_type)]  # include initial loss
    w = initial_w.copy()

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y,tx,batch_size, num_batches=1, shuffle=True) :

            # compute stochastic gradient on bathc 
            g =  compute_stoch_gradient(y_batch, tx_batch, w)

            # update weights 
            w = w - gamma * g

            # compute loss on full batch with current set of weights
            loss = compute_loss(y,tx, w, loss_type)

            losses.append(loss)
            ws.append(w)

        print(
            "SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )
    return losses, ws
