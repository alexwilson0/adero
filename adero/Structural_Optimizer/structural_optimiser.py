#!/usr/bin/env python3
"""
Hybrid Optimization Script for Wing Section Structural Parameters

This script fixes aerodynamic dimensions (Root Chord, Tip Chord, Span, Sweep)
and optimizes structural sizing (Rib Thickness, Spar Thickness, Skin Thickness,
Number of Ribs) to balance multiple performance metrics.

Regression models:
  - Neural Network predicts: [Total_Deformation, Shear_Stress_XY, Mass,
    Rib_Stress, Spar_Stress, Airfoil_Stress]
  - Gaussian Process predicts: [Root_Bending_Moment]

Objective: Minimize deformation, stresses, mass; maximize root bending moment.
Weights are equal by default but can be updated later.
"""

import numpy as np
import os
import pickle
from scipy.optimize import minimize as scipy_minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize as pymoo_minimize
from tensorflow import keras

_BASE_DIR        = os.path.dirname(__file__)
_NN_PATH         = os.path.join(_BASE_DIR, 'subset_models', 'NeuralNetwork.keras')
_GP_PATH         = os.path.join(_BASE_DIR, 'subset_models', 'GPR.pkl')
_SCALER_X_PATH   = os.path.join(_BASE_DIR, 'scaler_X.pkl')
_SCALER_Y_PATH   = os.path.join(_BASE_DIR, 'scaler_y.pkl')

# load once at import time
_nn_model   = keras.models.load_model(_NN_PATH)
_gp_model   = pickle.load(open(_GP_PATH, 'rb'))
_scaler_X   = pickle.load(open(_SCALER_X_PATH, 'rb'))
_scaler_y   = pickle.load(open(_SCALER_Y_PATH, 'rb'))

# Track simplex performance history for diagnostics
simplex_perf_history = []

#--------------------------------------------------------------------
# Combined Model Call
#--------------------------------------------------------------------
def CombinedModel_Call(x_input):

    x_arr = np.array(x_input, ndmin=2)
    x_scaled = _scaler_X.transform(x_arr)

    nn_out = _nn_model.predict(x_scaled)
    gp_out = _gp_model.predict(x_scaled) 

    gp_out = np.array(gp_out).reshape(-1, 1)  

    combined_scaled = np.zeros((x_scaled.shape[0], 7))

    combined_scaled[:,  :7] = nn_out
    combined_scaled[:, 7:] = gp_out

    combined = _scaler_y.inverse_transform(combined_scaled)
    return combined[0]

#--------------------------------------------------------------------
# Reference Benchmarks Function
#--------------------------------------------------------------------
def Objective_Benchmarks():
    """
    Returns normalization references for each output metric.
    Update with real benchmark values as needed.
    """
    ref = {
        'Total_Deformation': 0.05,      # m
        'Shear_Stress_XY': 1e6,         # Pa
        'Mass': 500.0,                  # kg
        'Rib_Stress': 1e7,              # Pa
        'Spar_Stress': 1e7,             # Pa
        'Airfoil_Stress': 1e7,          # Pa
        'Root_Bending_Moment': 1e5      # Nm
    }
    return ref

#--------------------------------------------------------------------
# Objective Function
#--------------------------------------------------------------------
def Objective(x, outputs, weights, refs):
    """
    Weighted sum: minimize most outputs, maximize Root_Bending_Moment.
    """
    # Extract values
    TD, SS, M, RS, SPS, AS, RBM = outputs

    W = np.array([
        float(weights['total_deformation']),
        float(weights['shear_stress_(xy)']),
        float(weights['mass']),
        float(weights['rib_stress']),
        float(weights['spar_stress']),
        float(weights['airfoil_stress']),
        float(weights['root_bending_moment'])
    ])
    W1, W2, W3, W4, W5, W6, W7 = W

    abs_sum = abs(TD) + abs(SS) + abs(M) + abs(RS) + abs(SPS) + abs(AS) +abs(RBM)

    W1_mapper = (W1 * abs_sum) / TD
    W2_mapper = (W2 * abs_sum) / SS
    W3_mapper = (W3 * abs_sum) / M
    W4_mapper = (W4 * abs_sum) / RS
    W5_mapper = (W5 * abs_sum) / SPS
    W6_mapper = (W6 * abs_sum) / AS
    W7_mapper = (W7 * abs_sum) / RBM

    mapper_sum = W1_mapper + W2_mapper + W3_mapper + W4_mapper

    W1_obj = W1_mapper / mapper_sum
    W2_obj = W2_mapper / mapper_sum
    W3_obj = W3_mapper / mapper_sum
    W4_obj = W4_mapper / mapper_sum
    W5_obj = W5_mapper / mapper_sum
    W6_obj = W6_mapper / mapper_sum
    W7_obj = W7_mapper / mapper_sum

    # Normalize
    val = (
        W1_obj * (TD) +
        W2_obj * (SS) +
        W3_obj * (M) +
        W4_obj * (RS) +
        W5_obj * (SPS) +
        W6_obj * (AS) -
        W7_obj * (RBM)
    )
    return val

#--------------------------------------------------------------------
# pymoo Problem Definition
#--------------------------------------------------------------------
class SectionOptimizationProblem(Problem):
    def __init__(self, fixed_root, fixed_tip, fixed_span, fixed_sweep, weights):
        # Variables: Rib_Th, Spar_Th, Skin_Th, No_Ribs
        lb = np.array([0.003, 0.005, 0.003, 4])
        ub = np.array([0.008, 0.015, 0.009, 13])
        super().__init__(n_var=4, n_obj=1, n_ieq_constr=0, xl=lb, xu=ub)
        self.refs = Objective_Benchmarks()
        self.fixed = (fixed_root, fixed_tip, fixed_span, fixed_sweep)
        self.weights = weights

    def _evaluate(self, X, out, *args, **kwargs):
        F = []
        for x in X:
            # x = [Rib_Th, Spar_Th, Skin_Th, No_Ribs]
            full = np.hstack((x, self.fixed))
            preds = CombinedModel_Call(full)
            F.append(Objective(x, preds, self.weights, self.refs))
        out['F'] = np.array(F).reshape(-1,1)

def full_hybrid_opt(fixed_root, fixed_tip, fixed_span, fixed_sweep,
                    weights, Pop, Gen, Simplex_iter):
    """
    Hybrid GA + Simplex for section sizing.
    """
    #print(f"Starting GA: Pop={Pop}, Generations={Gen}")
    problem = SectionOptimizationProblem(
        fixed_root, fixed_tip, fixed_span, fixed_sweep, weights
    )
    algo   = NSGA2(pop_size=Pop, save_history=True)
    res_ga = pymoo_minimize(problem, algo, ('n_gen', Gen), verbose=False)
    best   = res_ga.X
    #print(f"GA complete. Obj={res_ga.F[0]:.4f}\n")

    # Build COBYLA constraints from bounds
    lb, ub = problem.xl, problem.xu
    cons = []
    for i in range(len(lb)):
        cons.append({'type':'ineq', 'fun': lambda x,i=i: x[i] - lb[i]})
        cons.append({'type':'ineq', 'fun': lambda x,i=i: ub[i] - x[i]})

    # --- Define simplex objective and callback *inside* this scope ---
    def simplex_objective(xk):
        # rebuild problem to use the same fixed params
        prob_s = SectionOptimizationProblem(
            fixed_root, fixed_tip, fixed_span, fixed_sweep, weights
        )
        out = {}
        prob_s._evaluate(np.array([xk]), out)
        return out['F'][0, 0]

    simplex_perf_history.clear()
    def simplex_callback(xk):
        simplex_perf_history.append(simplex_objective(xk))

    #print(f"Starting Simplex: maxiter={Simplex_iter}\n")
    res_sim = scipy_minimize(
        simplex_objective,
        best,
        method='COBYLA',
        constraints=cons,
        options={'maxiter': Simplex_iter, 'disp': True}
    )
    #print(f"Simplex complete. Obj={res_sim.fun:.4f}")
    return res_sim.x

def optimize_section(fixed_root, fixed_tip, fixed_span, fixed_sweep, weights, Pop=9, Gen=11, Simplex_Iter=30):
    best_vars = full_hybrid_opt(fixed_root, fixed_tip, fixed_span, fixed_sweep,
                                weights, Pop, Gen, Simplex_Iter)
    full_design = np.hstack((best_vars,
                             [fixed_root, fixed_tip, fixed_span, fixed_sweep]))
    preds = CombinedModel_Call(full_design)
    return best_vars, preds
