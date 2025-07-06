#!/usr/bin/env python3
"""
Planform Optimization Script using Genetic Algorithm (GA) only.

This script takes as input:
  - Desired CL0 (minimum required lift coefficient)
  - Fixed planform parameters: AR, Area, and aerofoil CST parameters 
    (Up A0, Up A1, Up A2, Lo A0, Lo A1, Lo A2)
It then optimizes the remaining two variables, Taper and Sweep,
to find the best design that meets or exceeds the desired CL0 while:
    - Minimizing CD_min and CD_CL0,
    - Maximizing CL_min.
The full design vector is:
  [AR, Area, Taper, Sweep, Up A0, Up A1, Up A2, Lo A0, Lo A1, Lo A2].
The combined dual-model (NN & GP) produces 10 outputs:
  [CL0, CD_CL0, alpha0, a0, CD_min, CL_min, CD0, k, CM0, dCM_dalpha].
  
The objective function applies a heavy penalty if predicted CL0 is below the desired value,
and then uses a weighted sum to:
  - Minimize CD_min and CD_CL0,
  - Maximize CL_min.
"""

import numpy as np
import os
import pickle
from scipy.optimize import minimize as scipyminimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize as pymoo_minimize
from tensorflow import keras

_BASE_DIR       = os.path.dirname(__file__)
_NN_PATH        = os.path.join(_BASE_DIR, 'subset_models', 'NeuralNetwork.keras')
_GP_PATH        = os.path.join(_BASE_DIR, 'subset_models', 'GPR.pkl')
_SCALER_X_PATH  = os.path.join(_BASE_DIR, 'scaler_X.pkl')
_SCALER_Y_PATH  = os.path.join(_BASE_DIR, 'scaler_y.pkl')

_nn_model       = keras.models.load_model(_NN_PATH)
_gp_model       = pickle.load(open(_GP_PATH, 'rb'))
_scaler_X       = pickle.load(open(_SCALER_X_PATH, 'rb'))
_scaler_y       = pickle.load(open(_SCALER_Y_PATH, 'rb'))

simplex_perf_history = []

#--------------------------------------------------------------------
# Combined Model Call Function
def CombinedModel_Call(x_input):
    """
    Scale input, run both the NN & GP, stitch their outputs together,
    inverse‐scale, and return the 10 planform predictions:

      [CL0, CD_CL0, alpha0, a0, CD_min, CL_min, CD0, k, CM0, dCM_dalpha]
    """
    x_arr = np.array(x_input, ndmin=2)
    x_scaled = _scaler_X.transform(x_arr)

    #    NN_MODEL was trained on [CD_min, CL_min, CD0, k]
    nn_out = _nn_model.predict(x_scaled)       # shape (n_samples, 4)
    #    GP_MODEL was trained on [CL0, CD_CL0, alpha0, a0, CM0, dCM_dalpha]
    gp_out = _gp_model.predict(x_scaled)       


    combined_scaled = np.zeros((x_scaled.shape[0], 10))
    combined_scaled[:, 4:8] = nn_out
    combined_scaled[:, [0,1,2,3,8,9]] = gp_out

    combined = _scaler_y.inverse_transform(combined_scaled)

    return combined[0]

#--------------------------------------------------------------------
# Benchmark Function for Planform Optimization
def Objective_Benchmark_Planform(desired_CL0):
    """
    Returns benchmark reference values for planform optimization.
    (Placeholder values – update with realistic references as needed.)
    
    Parameters:
        desired_CL0 (float): The target minimum lift coefficient.
    
    Returns:
        Tuple: (desired_CL0, ref_CD_min, ref_CD_CL0, ref_CL_min)
    """
    ref_CD_min = 0.02   # Placeholder for reference minimum drag coefficient
    ref_CD_CL0 = 0.03   # Placeholder for reference CD_CL0
    ref_CL_min = 0.1    # Placeholder for reference minimum value for CL_min
    
    return desired_CL0, ref_CD_min, ref_CD_CL0, ref_CL_min

#--------------------------------------------------------------------
# Objective Function for Planform Optimization
def Objective(x, output, refs, weights, desired_CL0):
    """
    Computes the objective value based on design variables and model output.
    The objective penalizes designs where the predicted CL0 is below the desired value,
    and seeks to maximize CL_min while minimizing CD_min and CD_CL0.
    
    Parameters:
        x (array): Full design variables (10 values).
        output (array): Combined model predictions (10 outputs):
                        [CL0, CD_CL0, alpha0, a0, CD_min, CL_min, CD0, k, CM0, dCM_dalpha]
        refs (tuple): Benchmark references (desired_CL0, ref_CD_min, ref_CD_CL0, ref_CL_min).
    
    Returns:
        float: Objective function value.
    """
    desired_CL0 = desired_CL0

    predicted_CL0 = output[0]
    CD_CL0 = output[1]
    CD_min = output[4]
    CL_min = output[5]
    
    # Impose a heavy penalty if predicted CL0 is below desired.
    penalty = 0.0
    if predicted_CL0 < desired_CL0:
        penalty = 1000.0 * (desired_CL0 - predicted_CL0)
    
    # Weights for the objective components.
    W1, W2, W3, W4 = float(weights['drag_minimum']), float(weights['drag_at_lift_minimum']), float(weights['lift_minimum']), float(weights['lift'])

    abs_sum = CD_min + CD_CL0 + CL_min + predicted_CL0

    W1_mapper = (W1 * abs_sum) / CD_min
    W2_mapper = (W2 * abs_sum) / CD_CL0
    W3_mapper = (W3 * abs_sum) / CL_min
    W4_mapper = (W4 * abs_sum) / predicted_CL0

    mapper_sum = W1_mapper + W2_mapper + W3_mapper + W4_mapper
    
    W1_obj = W1_mapper / mapper_sum
    W2_obj = W2_mapper / mapper_sum
    W3_obj = W3_mapper / mapper_sum
    W4_obj = W4_mapper / mapper_sum

    # The terms are normalized by their reference values.
    # Note: We subtract the reward terms so that maximizing CL_min and CL0 lowers the objective.
    obj = (W1_obj * (CD_min) +
           W2_obj * (CD_CL0) -
           W3_obj * (CL_min) -
           W4_obj * (predicted_CL0) +
           penalty)
    
    return obj

#--------------------------------------------------------------------
# Define the Optimization Problem for pymoo
class PlanformOptimizationProblem(Problem):
    def __init__(self, desired_CL0, fixed_AR, fixed_Area, fixed_CST, weights, **kwargs):
        """
        fixed_AR: float, fixed aspect ratio.
        fixed_Area: float, fixed wing area.
        fixed_CST: list/array of 6 values [Up A0, Up A1, Up A2, Lo A0, Lo A1, Lo A2].
        """
        # Only Taper and Sweep are variables.
        # Their bounds (from previous full bounds for indices 2 and 3):
        # Taper: [0.050030964, 0.999990688]
        # Sweep: [0.001876309, 69.99869678]
        lower_bounds = np.array([0.050030964, 0.001876309])
        upper_bounds = np.array([0.999990688, 69.99869678])
        super().__init__(n_var=2, n_obj=1, n_ieq_constr=0, xl=lower_bounds, xu=upper_bounds, **kwargs)
        self.refs = Objective_Benchmark_Planform(desired_CL0)
        self.fixed_AR = fixed_AR
        self.fixed_Area = fixed_Area
        if len(fixed_CST) != 6:
            raise ValueError("fixed_CST must have 6 values: [Up A0, Up A1, Up A2, Lo A0, Lo A1, Lo A2]")
        self.fixed_CST = fixed_CST
        self.weights = weights
        self.desired_CL0 = desired_CL0
    
    def _evaluate(self, x, out, *args, **kwargs):
        n_samples = x.shape[0]
        residuals = []
        for i in range(n_samples):
            # x[i] contains [Taper, Sweep].
            # Construct full design vector:
            # [AR, Area, Taper, Sweep, Up A0, Up A1, Up A2, Lo A0, Lo A1, Lo A2]
            full_design = np.hstack((
                [self.fixed_AR, self.fixed_Area],
                x[i],
                self.fixed_CST
            ))
            output = CombinedModel_Call(full_design)
            res = Objective(full_design, output, self.refs, self.weights, self.desired_CL0)
            residuals.append(res)
        out["F"] = np.array(residuals).reshape(-1, 1)

def simplex_objective(x, desired_CL0, fixed_AR, fixed_Area, fixed_CST, weights):
    out = {}
    PlanformOptimizationProblem(desired_CL0, fixed_AR, fixed_Area, fixed_CST, weights)._evaluate(np.array([x]), out)
    return out["F"][0, 0]

#--------------------------------------------------------------------
# Genetic Algorithm Optimization using pymoo
def full_hybrid_opt(desired_CL0, fixed_AR, fixed_Area, fixed_CST, weights, Pop=9, Gen=6, Simplex_iter=30):
    """
    Performs the GA optimization and returns the best candidate design.
    
    Parameters:
        desired_CL0 (float): Target minimum CL0.
        fixed_AR (float): Fixed aspect ratio.
        fixed_Area (float): Fixed wing area.
        fixed_CST (list/array of 6 values): Fixed aerofoil CST parameters.
    
    Returns:
        array: Best candidate design (2 values: [Taper, Sweep]).
    """

    #print(f"\nStarting Genetic Algorithm with {Gen} generations and population size {Pop}.\n")
    problem = PlanformOptimizationProblem(desired_CL0, fixed_AR, fixed_Area, fixed_CST, weights)
    algorithm = NSGA2(pop_size=Pop, save_history=True)
    resultGA = pymoo_minimize(problem, algorithm, ('n_gen', Gen), verbose=False)
    best_candidate = resultGA.X
    best_performance = resultGA.F[0]
    #print(f"GA optimization complete. Best performance (objective value): {best_performance:.4f}\n")

    lower_bounds = np.array([0.050030964, 0.001876309])
    upper_bounds = np.array([0.999990688, 69.99869678])

    # Define inequality constraint functions (g(x) >= 0)
    constraints = []
    for i in range(len(lower_bounds)):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - lower_bounds[i]})  # x[i] >= lower_bounds[i]
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: upper_bounds[i] - x[i]})  # x[i] <= upper_bounds[i]
    
    simplex_perf_history = []

    def callback(xk):
    # Compute the objective value at the current iteration - FOR PLOTTING ONLY, OMIT IN FINAL VERSION
        simplex_performance = simplex_objective(xk, desired_CL0, fixed_AR, fixed_Area, fixed_CST, weights)
        simplex_perf_history.append(-simplex_performance)

    #print("Starting the simplex optimisation. Number of iterations = ", Simplex_iter, "\n")
    # Optimize using COBYLA
    resultSIM = scipyminimize(lambda x: simplex_objective(x, desired_CL0, fixed_AR, fixed_Area, fixed_CST, weights), 
                            best_candidate, 
                            method='COBYLA', 
                            constraints=constraints, 
                            options={'maxiter': Simplex_iter, 'disp': True})

    final_performance = -resultSIM.fun
    #print("\nSimplex optimization complete.")
    #print("Optimized Parameters:", resultSIM.x)
    #print("Performance:", final_performance)

    return resultSIM.x

#--------------------------------------------------------------------
# Main Optimization Function
def optimize_planform(desired_CL0, fixed_AR, fixed_Area, fixed_CST, weights):
    """
    Runs the optimization for a given desired CL0 and fixed parameters.
    
    Parameters:
        desired_CL0 (float): Target minimum CL0.
        fixed_AR (float): Fixed aspect ratio.
        fixed_Area (float): Fixed wing area.
        fixed_CST (list/array of 6 values): Fixed CST parameters.
    
    Returns:
        Tuple: (optimized design variables [Taper, Sweep], combined model predictions)
    """
    #print(f"Starting planform optimization for desired CL0: {desired_CL0}\n")
    best_vars = full_hybrid_opt(desired_CL0, fixed_AR, fixed_Area, fixed_CST, weights)
    # Construct full design vector from fixed and optimized values.
    full_design = np.hstack(([fixed_AR, fixed_Area], best_vars, fixed_CST))
    #print(full_design)
    outputs = CombinedModel_Call(full_design)
    return best_vars, outputs

