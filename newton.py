# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%


def deriv(f, x, eps=1e-16):
    """
    Function to obtain the derivative of a function

    Parameters:
    ------------
    f: function
    x: next x in the iteration
    eps: epsilon (default = 1e-16)

    Output:
    ------------
    Derivative of the function
    """
    # if x < 5:
    #    raise RuntimeError("fake error")
    return (f(x + eps) - f(x)) / eps


def deriv2(f, x, eps=1e-16):
    """
    Function to obtain the second derivate of a function

    Parameters:
    f: function
    x: next x in the iteration
    eps: epsilon (default = 1e-16)
    Function deriv(f, x, eps=1e-16)

    Output:
    ------------
    Second derivative of the function
    """
    return (deriv(f, x + eps, eps) - deriv(f, x, eps)) / eps


def optimize(x0, f, tol=1e-4):
    """
    Function that runs Newton's method to minimize function

    Parameters:
    ------------
    x0: starting value
    f: function
    tol: tolerance value (default = 1e-4)

    Output:
    ------------
    x_new: new x after iteration
    """
    # Apply Newton's Method iterative process to obtain the next value
    x_new = x0 - deriv(f, x0) / deriv2(f, x0)
    x = x0
    # Calculate delta (x new - x0) and compare against the established threshold (tol)
    # Delta must be less than tolerance, otherwise make x equal to xnew and repeat the process
    while abs(x_new - x) < tol:
        x = x_new
        x_new = x0 - deriv(f, x0) / deriv2(f, x0)
    # if delta is less than tolerance, then output xnew
    return {"x": x_new, "value": f(x_new)}

# suggested change right here 

# %%
