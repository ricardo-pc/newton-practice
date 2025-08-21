# NEW CHANGE FOR PR 

import numpy as np
from scipy.optimize import approx_fprime, _numdiff


# SECOND NEW HEADER
def grad_fd(f, x, eps=1e-8):
    """Finite-difference gradient ∇f(x)."""
    x = np.asarray(x, dtype=float)
    return approx_fprime(x, f, epsilon=eps)

def hess_fd(f, x, rel_step=1e-6):
    """Finite-difference Hessian H_f(x)."""
    x = np.asarray(x, dtype=float)

    # First get gradient function
    def grad_fun(y):
        return grad_fd(f, y)

    # Then differentiate gradient with approx_derivative
    return _numdiff.approx_derivative(grad_fun, x, rel_step=rel_step)



# LET ME CHANGE THIS TOO
def newton(f, x0, tol=1e-8, max_iter=50):
    """
    Multivariate Newton:
        x_{t+1} = x_t - H_f(x_t)^{-1} ∇f(x_t)
    """
    x = np.asarray(x0, dtype=float)
    for _ in range(max_iter):
        g = grad_fd(f, x)
        H = hess_fd(f, x)
        step = np.linalg.solve(H, g)
        x_new = x - step
        if np.linalg.norm(x_new - x) <= tol:
            return x_new
        x = x_new
    return x


# CHECK THIS TOO