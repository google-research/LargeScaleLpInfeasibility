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
using Plots
using LaTeXStrings

include("primal_dual_hybrid_gradient.jl")
include("utils.jl")

function generate_problem(alpha::Float64, beta::Float64)
  matrix_two_variables = [
    -1.0 -2.0
    -3.0 -1.0
    1.0 1.0
  ]
  matrix_three_variables = [
    -1.0 -2.0 0.0
    -3.0 -1.0 0.0
    1.0 1.0 0.0
  ]
  variable_lower_bounds = alpha == 0.0 ? [-Inf, -Inf] : [-Inf, -Inf, -Inf]
  variable_upper_bounds = alpha == 0.0 ? [Inf, Inf] : [Inf, Inf, Inf]
  objective_vector = alpha == 0.0 ? [1.0, 1.0] : [1.0, 1.0, -1.0]
  constraint_matrix =
    alpha == 0.0 ? matrix_two_variables : matrix_three_variables
  right_hand_side = [-2.0, -2.0, beta]
  return FirstOrderLp.linear_programming_problem(
    variable_lower_bounds,
    variable_upper_bounds,
    objective_vector,
    0.0, # objective_constant
    constraint_matrix,
    right_hand_side,
    0, #num_equalities
  )
end

function generate_component_wise_plots(
  primal_iterates,
  dual_iterates,
  results_dir::String,
  prefix::String,
)
  gr()
  fntsm = font("serif-roman", pointsize = 18)
  fntlg = font("serif-roman", pointsize = 22)
  default(
    titlefont = fntlg,
    guidefont = fntlg,
    tickfont = fntsm,
    legendfont = fntsm,
  )
  colors = [1, 2, 4]
  # Plot primal
  plot(
    [primal_iterates[j][1] for j in 1:length(primal_iterates)],
    label = L"x_0",
    line = (3, :solid),
    color = colors[1],
  )
  plot!(
    [primal_iterates[j][2] for j in 1:length(primal_iterates)],
    label = L"x_1",
    line = (3, :dot),
    color = colors[2],
  )
  if length(primal_iterates[1]) > 2
    plot!(
      [primal_iterates[j][3] for j in 1:length(primal_iterates)],
      label = L"x_2",
      line = (3, :dashdot),
      color = colors[3],
    )
  end
  xaxis!("Iteration count")
  yaxis!("Value of the variables")
  savefig(joinpath(results_dir, prefix * "p.pdf"))

  # Plot dual
  plot(
    [dual_iterates[j][1] for j in 1:length(dual_iterates)],
    label = L"y_0",
    line = (3, :solid),
    color = colors[1],
  )
  plot!(
    [dual_iterates[j][2] for j in 1:length(dual_iterates)],
    label = L"y_1",
    line = (3, :dot),
    color = colors[2],
  )
  if length(dual_iterates[1]) > 2
    plot!(
      [dual_iterates[j][3] for j in 1:length(dual_iterates)],
      label = L"y_2",
      line = (3, :dashdot),
      color = colors[3],
    )
  end
  xaxis!("Iteration count")
  yaxis!("Value of the variables")
  savefig(joinpath(results_dir, prefix * "d.pdf"))
end


function main()
  params = PdhgParameters(
    nothing, # step_size (forces the solver to use a provably correct step size)
    1.0, # primal_weight
    1000, # printing frequency
    false, # verbose
    1000, # iteration limit
    nothing, # two_dimensional_subspace
    0.0, # eps_primal_infasible
    0.0, # eps_dual_infeasible
    true, # store_iterates
    nothing, # reference_point
  )
  both_feasible_problem = generate_problem(0.0, 1.0)
  both_infeasible_problem = generate_problem(1.0, 2.0)
  dual_unbounded_problem = generate_problem(0.0, 2.0)
  primal_unbounded_problem = generate_problem(1.0, 1.0)
  output = optimize(params, both_feasible_problem)
  generate_component_wise_plots(
    output.primal_iterates,
    output.dual_iterates,
    ARGS[1],
    "pfdf",
  )
  output = optimize(params, both_infeasible_problem)
  generate_component_wise_plots(
    output.primal_iterates,
    output.dual_iterates,
    ARGS[1],
    "pidi",
  )
  output = optimize(params, dual_unbounded_problem)
  generate_component_wise_plots(
    output.primal_iterates,
    output.dual_iterates,
    ARGS[1],
    "pidf",
  )
  output = optimize(params, primal_unbounded_problem)
  generate_component_wise_plots(
    output.primal_iterates,
    output.dual_iterates,
    ARGS[1],
    "pfdi",
  )
end

main()
