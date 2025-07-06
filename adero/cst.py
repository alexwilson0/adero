import numpy as np
from scipy.special import comb

def class_function(x, N1=0.5, N2=1.0):
    return np.power(x, N1) * np.power(1 - x, N2)

def bernstein(i, n, x):
    return comb(n, i) * np.power(x, i) * np.power(1 - x, n - i)

def cst_model(x, *A, N1=0.5, N2=1.0):
    x = np.asarray(x)
    n = len(A) - 1
    S = sum(A[i] * bernstein(i, n, x) for i in range(n + 1))
    return class_function(x, N1, N2) * S

def average_thickness_ratio(upper_coeffs, lower_coeffs, num_points=200):
    x = np.linspace(0, 1, num_points)
    up = cst_model(x, *upper_coeffs)
    low= cst_model(x, *lower_coeffs)
    thickness = up - low
    # integral over x [0,1] is already mean since width=1
    return float(np.trapz(thickness, x))