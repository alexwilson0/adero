import os
import numpy as np
from scipy.optimize import curve_fit
from cst import cst_model

def fit_cst(x_data, y_data, initial_guess):
    popt, _ = curve_fit(lambda x, *A: cst_model(x, *A), x_data, y_data, p0=initial_guess)
    return popt

def load_airfoil(file_path):
    airfoil_name = os.path.splitext(os.path.basename(file_path))[0]
    
    data = np.loadtxt(file_path, skiprows=1)
    x_all = data[:, 0]
    y_all = data[:, 1]
    
    le_index = np.argmin(x_all)
    x_upper, y_upper = x_all[:le_index+1], y_all[:le_index+1]
    x_lower, y_lower = x_all[le_index:], y_all[le_index:]
    
    chord = x_all.max() - x_all.min()
    x_upper_norm = (x_upper - x_all.min()) / chord
    x_lower_norm = (x_lower - x_all.min()) / chord
    
    return airfoil_name, x_upper_norm, y_upper, x_lower_norm, y_lower

def dat_to_CST(file_path):
    order = 2  # 3 coefficients
    initial_guess = [0.0] * (order + 1)

    airfoil_name, x_upper, y_upper, x_lower, y_lower = load_airfoil(file_path)

    popt_upper = fit_cst(x_upper, y_upper, initial_guess)
    popt_lower = fit_cst(x_lower, y_lower, initial_guess)

    return popt_upper, popt_lower
