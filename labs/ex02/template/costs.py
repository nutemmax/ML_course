# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np


def compute_loss(y, tx, w, loss_type = "MSE"):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.
        loss_type (optional) : string, 'MSE' or 'MAE', denoting the type of loss we compute 

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = y - tx @ w
    N = len(y)
        
    MAE = np.mean(np.abs(e))
    MSE = 1 / (2*N) * (e.T @ e) 
    
    if loss_type == "MSE":
        return MSE
    elif loss_type == "MAE":
        return MAE
    else:
        raise ValueError("loss_type must be 'MSE' or 'MAE'")
