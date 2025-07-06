"""
Airfoil Optimization Script

This script takes a Reynolds number (Re) as input and outputs the optimized airfoil parameters.
It uses a hybrid optimization approach:
  1. A Genetic Algorithm (using pymoo) to get an initial candidate.
  2. A Simplex (COBYLA) optimization (using scipy.optimize.minimize) to refine the solution.

Finally, the optimized candidate is passed to a neural network to obtain the corresponding outputs.
"""

import os
import pickle
import numpy as np
from scipy.integrate import quad
from tensorflow import keras
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize as pymoo_minimize
from scipy.optimize import minimize as scipy_minimize

from adero.cst import cst_model

_BASE_DIR     = os.path.join(os.getcwd(), 'adero/Airfoil_Optimizer')
_MODEL        = keras.models.load_model(os.path.join(_BASE_DIR, 'best_aerofoil_nn.keras'))
_SCALER_X     = pickle.load(open(os.path.join(_BASE_DIR, 'scaler_X.pkl'), 'rb'))
_SCALER_Y     = pickle.load(open(os.path.join(_BASE_DIR, 'scaler_y.pkl'), 'rb'))

#--------------------------------------------------------------------
# Utility function: cross-sectional area of the airfoil between x_start and x_end
# using CST definitions

def X_Section_Area(upper_coeffs, lower_coeffs, x_start=0.0, x_end=1.0):
    def integrand(x):
        return cst_model(x, *upper_coeffs) - cst_model(x, *lower_coeffs)
    area, _ = quad(integrand, x_start, x_end)
    return area

#--------------------------------------------------------------------
# Neural Network surrogate call

def NN_Call(x_input):
    """now just use the globals, no re‐loading."""
    x_arr = np.array(x_input, ndmin=2)
    x_scaled = _SCALER_X.transform(x_arr)
    pred_scaled = _MODEL.predict(x_scaled)
    pred = _SCALER_Y.inverse_transform(pred_scaled)
    return pred[0] if pred.shape[0] == 1 else pred

#--------------------------------------------------------------------
# Benchmark reference from NACA0009

def Objective_Benchmark(Re):
    naca_input = [Re, 0.136496419, 0.112719347, 0.1201507089,
                  -0.109578871, -0.131592109, -0.067587853]
    out = NN_Call(naca_input)
    CL0_ref = out[0]
    CD0_ref = out[6]
    CLst_ref = out[1]
    area_ref = X_Section_Area([0.13408799,0.113120489,0.103381083],
                               [-0.117176292,-0.103004856,-0.099195319])
    return CL0_ref/CD0_ref, CLst_ref, area_ref

#--------------------------------------------------------------------
# Objective function combining mapped weights and performance

def Objective(x, output, refs, weights):
    CLCD_ref, CLst_ref, A_ref = refs
    # Fixed weight distribution in mapper

    W = np.array([
        float(weights['lift']),
        float(weights['drag']),
        float(weights['lift_at_stall']),
        float(weights['aerofoil_area'])
    ])

    W1, W2, W3, W4 = W

    def single_obj(xi, outi):
        CL0, CLst, CD0 = outi[0], outi[1], outi[6]
        area = X_Section_Area(xi[1:4], xi[-3:])
        s = CL0 + CD0 + CLst + area
        m1 = W1 * s / CL0
        m2 = W2 * s / CD0
        m3 = W3 * s / CLst
        m4 = W4 * s / area
        m_sum = m1 + m2 + m3 + m4
        w1o, w2o, w3o, w4o = m1/m_sum, m2/m_sum, m3/m_sum, m4/m_sum
        return -(w1o*CL0 + w2o/CD0 + w3o*CLst + w4o*area)

    x_arr = np.array(x)
    if x_arr.ndim == 1:
        return single_obj(x_arr, output)
    else:
        return np.array([single_obj(x_arr[i], output[i]) for i in range(len(x_arr))])

#--------------------------------------------------------------------
# PyMOO problem definition

class AirfoilOptimizationProblem(Problem):
    def __init__(self, Re, weights, n_var=7, **kwargs):
        lb = np.array([Re, 0.08217587, 0.30071102, 0.11121146,
                       -0.30071102, -0.3361913, -0.199198])
        ub = np.array([Re, 0.10051946, 0.49846749, 0.21794427,
                       -0.0821759, 0.28573566, -0.1105485])
        super().__init__(n_var=n_var, n_obj=1, xl=lb, xu=ub, vtype=float, **kwargs)
        self.weights = weights
        self.refs = Objective_Benchmark(Re)

    def _evaluate(self, x, out, *args, **kwargs):
        out['F'] = Objective(x, NN_Call(x), self.refs, self.weights)

#--------------------------------------------------------------------
# Genetic Algorithm phase

def full_hybrid_opt(Re, weights):
    problem = AirfoilOptimizationProblem(Re, weights)
    algorithm = NSGA2(pop_size=12, save_history=True)
    res = pymoo_minimize(problem, algorithm, ('n_gen', 8), verbose=False)
    return res.X

#--------------------------------------------------------------------
# Simplex (COBYLA) refinement phase

def simp_opt(GA_best, Re, weights):
    problem = AirfoilOptimizationProblem(Re, weights)
    refs = problem.refs
    # Construct inequality constraints from bounds
    lb, ub = problem.xl, problem.xu
    constraints = []
    for i in range(len(lb)):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - lb[i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: ub[i] - x[i]})

    history = []

    res = scipy_minimize(
        lambda xi: Objective(xi, NN_Call(xi), refs, weights),
        GA_best,
        method='COBYLA',
        constraints=constraints,
        callback=lambda xk: history.append(-Objective(xk, NN_Call(xk), refs, weights)),
        options={'maxiter':50, 'disp':True}
    )
    return res.x, -res.fun, history

#--------------------------------------------------------------------
# Main optimization entry

def optimize_airfoil(Re, weights):
    ga = full_hybrid_opt(Re, weights)
    opt_params, perf, hist = simp_opt(ga, Re, weights)
    nn_out = NN_Call(opt_params)
    return opt_params, nn_out, perf, hist
