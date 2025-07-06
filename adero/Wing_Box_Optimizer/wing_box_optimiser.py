#!/usr/bin/env python3
"""
Structural Component Optimization Script using a Hybrid GA + Simplex Approach.

This script takes as input:
  - Fixed structural parameters: tc_avg (average thickness-to-chord ratio) and Wing_Loading (kg/m^2).
  - The optimizer has freedom to vary: Chord, Span, and Taper.

It optimizes to minimize:
  - Tip_Deflection
  - Max_Stress
  - Stress_Ratio
  - Root_Bending_Moment

The regression models are a combination of a Neural Network and a Gaussian Process.
The Neural Network predicts: [Tip_Deflection, Max_Stress, Stress_Ratio]
The Gaussian Process predicts: [Root_Bending_Moment]

The objective is a weighted sum of normalized outputs; default weights are equal but can be adjusted.
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
_nn_path         = os.path.join(_BASE_DIR, 'subset_models', 'NeuralNetwork.keras')
_gp_path         = os.path.join(_BASE_DIR, 'subset_models', 'GPR.pkl')
_scaler_x_path   = os.path.join(_BASE_DIR, 'scaler_X.pkl')
_scaler_y_path   = os.path.join(_BASE_DIR, 'scaler_y.pkl')

_nn_model        = keras.models.load_model(_nn_path)
_gp_model        = pickle.load(open(_gp_path, 'rb'))
_scaler_X        = pickle.load(open(_scaler_x_path, 'rb'))
_scaler_y        = pickle.load(open(_scaler_y_path, 'rb'))

# History of simplex objective values (for diagnostics)
simplex_perf_history = []

#--------------------------------------------------------------------
# Combined Model Call Function
#--------------------------------------------------------------------
def CombinedModel_Call(x_input):
    x_arr = np.array(x_input, ndmin=2)
    x_scaled = _scaler_X.transform(x_arr)

    nn_preds = _nn_model.predict(x_scaled)
    gp_preds = _gp_model.predict(x_scaled)

    combined = np.zeros((x_scaled.shape[0], 4))
    combined[:, :3] = nn_preds
    combined[:, 3]  = gp_preds.flatten()
    return _scaler_y.inverse_transform(combined)[0]

#--------------------------------------------------------------------
# Benchmark Reference Function
#--------------------------------------------------------------------
def Objective_Benchmark_Structural():
    """
    Returns reference normalization values for structural outputs.
    Update these based on realistic design benchmarks.
    """
    ref_Tip_Def = 1.0          # meters
    ref_Max_Stress = 1000.0    # MPa
    ref_Stress_Ratio = 1.0     # unitless
    ref_Root_BM = 10000.0      # Nm
    return ref_Tip_Def, ref_Max_Stress, ref_Stress_Ratio, ref_Root_BM

#--------------------------------------------------------------------
# Objective Function
#--------------------------------------------------------------------
def Objective(x, output, weights, refs):
    """
    Weighted sum of normalized outputs to minimize.
    """
    ref_Tip_Def, ref_Max_Stress, ref_Stress_Ratio, ref_Root_BM = refs
    tip_def, max_stress, stress_ratio, root_bm = output
    chord = float(x[0])
    span = float(x[1])
    inv_AR = chord / span

    W = np.array([
        float(weights['tip_deflection']),
        float(weights['max_stress']),
        float(weights['stress_ratio']),
        float(weights['root_bending_moment']),
        float(weights['aspect_ratio'])
    ])

    W1, W2, W3, W4, W5 = W

    abs_sum = abs(tip_def) + abs(max_stress) + abs(stress_ratio) + abs(root_bm) + abs(inv_AR)

    W1_mapper = (W1 * abs_sum) / tip_def
    W2_mapper = (W2 * abs_sum) / max_stress
    W3_mapper = (W3 * abs_sum) / stress_ratio
    W4_mapper = (W4 * abs_sum) / root_bm
    W5_mapper = (W5 * abs_sum) / inv_AR

    mapper_sum = W1_mapper + W2_mapper + W3_mapper + W4_mapper + W5_mapper

    W1_obj = W1_mapper / mapper_sum
    W2_obj = W2_mapper / mapper_sum
    W3_obj = W3_mapper / mapper_sum
    W4_obj = W4_mapper / mapper_sum
    W5_obj = W5_mapper / mapper_sum

    obj = (
        W1_obj * (tip_def) +
        W2_obj * (max_stress) +
        W3_obj * (stress_ratio) -
        W4_obj * (root_bm) +
        W5_obj * (inv_AR))
    
    return obj

#--------------------------------------------------------------------
# Define the Optimization Problem for pymoo
#--------------------------------------------------------------------
class StructuralOptimizationProblem(Problem):
    def __init__(self, fixed_tc_avg, fixed_MTOW, weights):
        # Variable bounds: [Chord (m), Span (m), Taper (unitless)]
        lower_bounds = np.array([0.5, 2.18, 0.05])
        upper_bounds = np.array([8.0, 30.0, 1.0])
        super().__init__(n_var=3, n_obj=1, n_ieq_constr=0,
                         xl=lower_bounds, xu=upper_bounds)
        self.refs = Objective_Benchmark_Structural()
        self.fixed_tc = fixed_tc_avg
        self.fixed_MTOW = fixed_MTOW
        self.weights = weights

    def _evaluate(self, X, out, *args, **kwargs):
        F = []
        for x in X:
            # x = [Chord, Span, Taper]
            full_design = np.hstack((x, [self.fixed_tc, self.fixed_MTOW]))
            preds = CombinedModel_Call(full_design)
            val = Objective(full_design, preds, self.weights, self.refs)
            F.append(val)
        out['F'] = np.array(F).reshape(-1, 1)

# History of simplex objective values (for diagnostics)
simplex_perf_history = []

# … [CombinedModel_Call, Objective_Benchmark_Structural, Objective,
#      and StructuralOptimizationProblem definitions remain unchanged] …

def full_hybrid_opt(fixed_tc_avg, fixed_MTOW, weights, Pop, Gen, Simplex_iter):
    """
    Performs a hybrid GA + Simplex optimization for the structural problem.
    Returns the best [Chord, Span, Taper].
    """
    #print(f"\nStarting GA with {Gen} generations and population size {Pop}.\n")
    problem   = StructuralOptimizationProblem(fixed_tc_avg, fixed_MTOW, weights)
    algorithm = NSGA2(pop_size=Pop, save_history=True)
    result_ga = pymoo_minimize(problem,
                               algorithm,
                               ('n_gen', Gen),
                               verbose=False)
    best = result_ga.X
    #print(f"GA done. Best objective: {result_ga.F[0]:.4f}\n")

    # Build inequality constraints from the problem bounds
    lb = problem.xl
    ub = problem.xu
    constraints = []
    for i in range(len(lb)):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - lb[i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: ub[i] - x[i]})

    # Define simplex objective in this scope, capturing fixed_tc_avg & fixed_MTOW
    def simplex_objective(xk):
        prob_s = StructuralOptimizationProblem(fixed_tc_avg, fixed_MTOW, weights)
        out = {}
        prob_s._evaluate(np.array([xk]), out)
        return out['F'][0, 0]

    # Clear and collect simplex history via callback
    simplex_perf_history.clear()
    def simplex_callback(xk):
        simplex_perf_history.append(simplex_objective(xk))

    print(f"Starting Simplex with maxiter={Simplex_iter}.\n")
    result_sim = scipy_minimize(
        simplex_objective,
        best,
        method='COBYLA',
        constraints=constraints,
        options={'maxiter': Simplex_iter, 'disp': True}
    )

    print("Simplex done. Optimized parameters:", result_sim.x)
    return result_sim.x

def optimize_structure(fixed_tc_avg, fixed_MTOW, weights, Pop=14, Gen=7, Simplex_Iter=50):
    best_vars = full_hybrid_opt(fixed_tc_avg, fixed_MTOW, weights, Pop, Gen, Simplex_Iter)
    full_design = np.hstack((best_vars, [fixed_tc_avg, fixed_MTOW]))
    outputs = CombinedModel_Call(full_design)
    return best_vars, outputs
