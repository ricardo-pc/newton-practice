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

def deriv(f, x, eps = 1e-16):
    # if x < 5:
    #    raise RuntimeError("fake error")
    return (f(x+eps) - f(x)) / eps


def deriv2(f, x, eps = 1e-16):
    return (deriv(f, x+eps, eps) - deriv(f, x, eps)) / eps


def optimize(x0, f, tol =1e-4):
    x_new = x0 - deriv(f, x0)/    deriv2(f, x0)
    x = x0
    while abs(x_new - x) < tol:
        x = x_new
        x_new = x0 - deriv(f, x0) / deriv2(f, x0)
    return {"x": x_new,
            'value': f(x_new)}

# %%
