# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Gradient Descent
"""


def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    N = y.shape[0]
    e = y - tx @ w
    grad_L = - 1/N * tx.T @ e
    return grad_L


def gradient_descent(y, tx, initial_w, max_iters, gamma, loss_type = "MSE"):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        loss_type (optional) : string, 'MSE' or 'MAE', denoting the type of loss we compute over the dataset

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters + 1 containing the model parameters as numpy arrays of shape (2, ),
            for each iteration of GD (as well as the final weights)
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = [compute_loss(y,tx, initial_w, loss_type)] # compute loss for initial_w as well
    w = initial_w

    for n_iter in range(max_iters):
        # compute gradient
        grad_L = compute_gradient(y,tx, w)

        # update weights
        w = w - gamma* grad_L

        # compute loss with new weights
        loss = compute_loss(y,tx, w, loss_type)

        # store w and loss
        ws.append(w)
        losses.append(loss)
        print(
            "GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )

    return losses, ws
