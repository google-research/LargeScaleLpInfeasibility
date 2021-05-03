# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import FirstOrderLp

include("primal_dual_hybrid_gradient.jl")
include("utils.jl")

# This script runs a experiment on the itest2 instance from the netlib database.
# It assumes the file is located in data/itest2.mps, see the README for details
# about how to obtain this file.
output_csv_path = "/tmp/blinear_current_stats.csv"
output_plots_dir = "/tmp"
print("Creating LP instance\n")
""" Returns a 2D bilinear problem.
"""
lambda_min = 1e-1
simple_lp = FirstOrderLp.linear_programming_problem(
  [-Inf, -Inf],  # variable_lower_bound
  [Inf, Inf],  # variable_upper_bound
  [0.0, 0.0],  # objective_vector
  0.0,                 # objective_constant
  [
    1.0 0.0
    0.0 lambda_min
  ],           # constraint_matrix
  [0.0, 0.0],      # right_hand_side
  2,                     # num_equalities
)

params = PdhgParameters(
  0.8, # step_size (forces the solver to use a provably correct step size)
  1.0, # primal_weight
  100, # printing frequency
  true, # verbose
  500, # iteration limit
  TwoDimensionalSubspace([0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
  0.0, # eps_primal_infeasible
  0.0, # eps_dual_infeasible
  false, # store_iterates
  nothing, # reference_point
)
print("About to start optimizing\n")
solver_output = optimize(params, simple_lp, [1.0, 1.0], [1.0, 1.0])
save_stats_to_csv(solver_output, output_csv_path)
generate_plots_from_stats_csv(output_csv_path, output_plots_dir)
