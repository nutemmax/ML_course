# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """calculate the least squares."""
    # normal equations : 
    A = tx.T @ tx # gram matrix
    b = tx.T @ y  # right hand side

    # solve for b using linalg solve
    w = np.linalg.solve(A, b)

    # error and MSE
    e = y - tx @ w
    mse = np.mean(e ** 2) / 2

    return w, mse
