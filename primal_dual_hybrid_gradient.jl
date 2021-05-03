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
#
# This is a simplified version of primal-dual hybrid gradient implementation, of
# the package FirstOrderLp (see primal_dual_hybrid_gradient.jl), aimed to be
# used for numerical experiments.

import FirstOrderLp
import Printf
import LinearAlgebra
import Random
const norm = LinearAlgebra.norm
const dot = LinearAlgebra.dot
const randn = Random.randn

"""
Information for recording iterates on two dimensional subspace.
"""
struct TwoDimensionalSubspace
  """
  Subspace basis vector 1
  """
  basis_vector1::Vector{Float64}
  """
  Subspace basis vector 2, should be orthognal to vector 1
  """
  basis_vector2::Vector{Float64}
end

"""
A SimplePdhgParameters struct specifies the parameters for solving the saddle
point formulation of an problem using primal-dual hybrid gradient.

It solves a problem of the form (see quadratic_programmming.jl in FirstOrderLp)
minimize objective_vector' * x

s.t. constraint_matrix[1:num_equalities, :] * x =
     right_hand_side[1:num_equalities]

     constraint_matrix[(num_equalities + 1):end, :] * x >=
     right_hand_side[(num_equalities + 1):end, :]

     variable_lower_bound <= x <= variable_upper_bound

We use notation from Chambolle and Pock, "On the ergodic convergence rates of a
first-order primal-dual algorithm"
(http://www.optimization-online.org/DB_FILE/2014/09/4532.pdf).
That paper doesn't explicitly use the terminology "primal-dual hybrid gradient"
but their Theorem 1 is analyzing PDHG. In this file "Theorem 1" without further
reference refers to that paper.

Our problem is equivalent to the saddle point problem:
    min_x max_y L(x, y)
where
    L(x, y) = y' K x + g(x) - h*(y)
    K = -constraint_matrix
    g(x) = objective_vector' x if variable_lower_bound <= x <=
                                                            variable_upper_bound
                               otherwise infinity
    h*(y) = -right_hand_side' y if y[(num_equalities + 1):end] >= 0
                                otherwise infinity

Note that the places where g(x) and h*(y) are infinite effectively limits the
domain of the min and max. Therefore there's no infinity in the code.

We parametrize the primal and dual step sizes (tau and sigma in Chambolle and
Pock) as:
    primal_step_size = step_size / primal_weight
    dual_step_size = step_size * primal_weight.
The algoritm converges if
    primal_stepsize * dual_stepsize * norm(constraint_matrix)^2 < 1.
"""
struct PdhgParameters
  """
  Constant step size used in the algorithm. If nothing is specified, the solver
  computes a provably correct step size.
  """
  step_size::Union{Float64,Nothing}
  """
  Weight relating primal and dual step sizes.
  """
  primal_weight::Float64
  """
  Prints iteration stats at this frequency (in iterations) if verbose is true.
  """
  printing_frequency::Int64
  """
  If true a line of debugging info is printed every printing_frequency
  iterations. Otherwise some info is printed about the final solution.
  """
  verbosity::Bool
  """
  Number of loop iterations to run. Must be postive.
  """
  iteration_limit::Int64
  """
  A subspace where projections of iterates are recorded.
  If the value is nothing then the NaNs are recorded instead of a
  projection of the iterates.
  """
  two_dimensional_subspace::Union{TwoDimensionalSubspace,Nothing}
  """
  Accuracy to determine if a candidate point is a primal infeasibility
  certificate. See FirstOrderLp for documentation.
  """
  eps_primal_infeasible::Float64
  """
  Accuracy to determine if a candidate point is a dual infeasibility
  certificate. See FirstOrderLp for documentation.
  """
  eps_dual_infeasible::Float64
  """
  Determines whether or not to save the iterates of the run
  """
  store_iterates::Bool
  """
  If not Nothing, the solver will output three arrays containing the distance
  between this point and the normalized average, the normalized iterate, and the
  difference of iterates, respectively. The solver will include distances to
  this point in the returned PdhgStats.
  """
  reference_point::Union{Vector{Float64},Nothing}
end

"""
Statistics of the execution.
"""
struct PdhgStats
  """
  Primal objectives of the iterates; the ith entry corresponds to the primal
  objective of the ith iterate.
  """
  primal_objectives::Vector{Float64}
  """
  Primal objectives of the iterates; the ith entry corresponds to the dual
  objective of the ith iterate.
  """
  dual_objectives::Vector{Float64}
  """
  Primal norms of the iterates; the ith entry corresponds to the primal
  norm of the ith iterate.
  """
  primal_solution_norms::Vector{Float64}
  """
  Dual norms of the iterates; the ith entry corresponds to the dual
  norm of the ith iterate.
  """
  dual_solution_norms::Vector{Float64}
  """
  Primal delta norms of the iterates; the ith entry corresponds to the primal
  delta norm of the ith iterate.
  """
  primal_delta_norms::Vector{Float64}
  """
  Dual delta norms of the iterates; the ith entry corresponds to the dual
  delta norm of the ith iterate.
  """
  dual_delta_norms::Vector{Float64}
  """
  First coordinate of a subspace that the current iterates are projected onto.
  """
  current_subspace_coordinate1::Vector{Float64}
  """
  Second coordinate of a subspace that the current iterates are projected onto.
  """
  current_subspace_coordinate2::Vector{Float64}
  """
  First coordinate of a subspace that the average iterates are projected onto.
  """
  average_subspace_coordinate1::Vector{Float64}
  """
  Second coordinate of a subspace that the average iterates are projected onto.
  """
  average_subspace_coordinate2::Vector{Float64}
  """
  Euclidean distance between current_iterate/iteration_number and the
  input reference point at each iteration. If no reference point is given,
  this field is left as Nothing.
  """
  distance_to_normalized_current::Union{Vector{Float64},Nothing}
  """
  Euclidean distance between 2/(iteration_count + 1) * average_iterate and the
  input reference point at each iteration. If no reference point is given, this
  field is left as Nothing.
  """
  distance_to_normalized_average::Union{Vector{Float64},Nothing}
  """
  Euclidean distance between current_iterate - previous_iterate and the input
  reference point at each iteration. If no reference point is given, this field
  is left as Nothing.
  """
  distance_to_difference::Union{Vector{Float64},Nothing}
  """
  Maximum certificate error of the primal infeasibility certificate from
  current_iterate/iteration_number, scaled so that the dual ray objective is 1.
  If the dual ray objective is zero, this will be infinity, and if the dual ray
  objective is negative, this will be negative.
  """
  max_scaled_primal_certificate_error_of_normalized_current::Vector{Float64}
  """
  Maximum certificate error of the primal infeasibility certificate from
  2/(iteration_count + 1) * average_iterate, scaled so that the dual ray
  objective is 1. If the dual ray objective is zero, this will be infinity, and
  if the dual ray objective is negative, this will be negative.
  """
  max_scaled_primal_certificate_error_of_normalized_average::Vector{Float64}
  """
  Maximum certificate error of the primal infeasibility certificate from
  current_iterate - previous_iterate, scaled so that the dual ray objective
  is 1. If the dual ray objective is zero, this will be infinity, and if the
  dual ray objective is negative, this will be negative.
  """
  max_scaled_primal_certificate_error_of_difference::Vector{Float64}
end

"""
Output of the solver.
"""
struct PdhgOutput
  primal_solution::Vector{Float64}
  dual_solution::Vector{Float64}
  primal_delta::Vector{Float64}
  dual_delta::Vector{Float64}
  iteration_stats::PdhgStats
  """
  Records the first iteration in which a PointType detects infeasibility.
  """
  first_infeasibility_detection::Dict{FirstOrderLp.PointType,Int64}
  """
  Vector with all the primal iterates generated in the run.
  """
  primal_iterates::Union{Vector{Vector{Float64}},Nothing}
  """
  Vector with all the dual iterates generated in the run.
  """
  dual_iterates::Union{Vector{Vector{Float64}},Nothing}
  """
  The last iteration in which the active set of the primal and dual variables change.
  """
  last_active_set_change::Int64
end

"""
Computes statistics for the current iteration. The arguments primal_delta and
dual_delta correspond to the difference between the last two primal and dual
iterates, respectively.
"""
function compute_stats(
  problem::FirstOrderLp.QuadraticProgrammingProblem,
  iteration::Int64,
  two_dimensional_subspace::Union{TwoDimensionalSubspace,Nothing},
  stats::PdhgStats,
  primal_solution::Vector{Float64},
  dual_solution::Vector{Float64},
  primal_delta::Vector{Float64},
  dual_delta::Vector{Float64},
  solution_weighted_avg::FirstOrderLp.SolutionWeightedAverage,
)
  dual_stats =
    FirstOrderLp.compute_dual_stats(problem, primal_solution, dual_solution)
  stats.primal_objectives[iteration] =
    FirstOrderLp.primal_obj(problem, primal_solution)
  stats.dual_objectives[iteration] = dual_stats.dual_objective
  stats.primal_solution_norms[iteration] = norm(primal_solution)
  stats.dual_solution_norms[iteration] = norm(dual_solution)
  stats.primal_delta_norms[iteration] = norm(primal_delta)
  stats.dual_delta_norms[iteration] = norm(dual_delta)
  if nothing != two_dimensional_subspace
    z = [primal_solution; dual_solution]
    stats.current_subspace_coordinate1[iteration] =
      dot(z, two_dimensional_subspace.basis_vector1)
    stats.current_subspace_coordinate2[iteration] =
      dot(z, two_dimensional_subspace.basis_vector2)
    primal_avg, dual_avg = FirstOrderLp.compute_average(solution_weighted_avg)
    z_avg = [primal_avg; dual_avg]
    stats.average_subspace_coordinate1[iteration] =
      dot(z_avg, two_dimensional_subspace.basis_vector1)
    stats.average_subspace_coordinate2[iteration] =
      dot(z_avg, two_dimensional_subspace.basis_vector2)
  else
    stats.current_subspace_coordinate1[iteration] = NaN
    stats.current_subspace_coordinate2[iteration] = NaN
    stats.average_subspace_coordinate1[iteration] = NaN
    stats.average_subspace_coordinate2[iteration] = NaN
  end
end

function compute_infeasibility_distance_stats(
  iteration::Int64,
  stats::PdhgStats,
  reference_point::Vector{Float64},
  primal_solution::Vector{Float64},
  dual_solution::Vector{Float64},
  solution_weighted_avg::FirstOrderLp.SolutionWeightedAverage,
  primal_delta::Vector{Float64},
  dual_delta::Vector{Float64},
)
  primal_avg, dual_avg = FirstOrderLp.compute_average(solution_weighted_avg)
  stats.distance_to_normalized_current[iteration] =
    norm(reference_point - (1 / iteration) * [primal_solution; dual_solution])
  stats.distance_to_normalized_average[iteration] =
    norm(reference_point - (2 / (iteration + 1)) * [primal_avg; dual_avg])
  stats.distance_to_difference[iteration] =
    norm(reference_point - [primal_delta; dual_delta])
end

"""
Logging while the algorithm is running.
"""
function pdhg_log(
  problem::FirstOrderLp.QuadraticProgrammingProblem,
  iteration::Int64,
  stats::PdhgStats,
)

  Printf.@printf(
    "   %5d objectives=(%9g, %9g) norms=(%9g, %9g) res_norm=(%9g, %9g)\n",
    iteration,
    stats.primal_objectives[iteration],
    stats.dual_objectives[iteration],
    stats.primal_solution_norms[iteration],
    stats.dual_solution_norms[iteration],
    stats.primal_delta_norms[iteration],
    stats.dual_delta_norms[iteration],
  )
end

"""
Prints infeasibility information
"""
function infeasibility_log(
  iteration::Int64,
  infeasibility_information::FirstOrderLp.InfeasibilityInformation,
  point_type::FirstOrderLp.PointType,
)
  Printf.@printf(
    "   %5d %s max_dual: %9g dual_obj: %9g max_primal: %9g primal_obj %9g \n",
    iteration,
    string(point_type),
    infeasibility_information.max_dual_ray_infeasibility,
    infeasibility_information.dual_ray_objective,
    infeasibility_information.max_primal_ray_infeasibility,
    infeasibility_information.primal_ray_linear_objective,
  )
end

"""
Checks if the normalized current iterate, normalized average iterate, and
iterate difference provide (approximate) certificates of infeasibility.

# Returns
A dictionary from point type to the maximum infeasibility of the corresponding
scaled dual ray, scaled so that the dual ray objective is 1.0. This will be
infinity if the dual ray objective is 0.0, and negative is the dual ray
objective is negative.
"""
function check_infeasibility(
  problem::FirstOrderLp.QuadraticProgrammingProblem,
  iteration::Int64,
  first_infeasibility_detection::Dict{FirstOrderLp.PointType,Int64},
  primal_solution::Vector{Float64},
  primal_delta::Vector{Float64},
  dual_solution::Vector{Float64},
  dual_delta::Vector{Float64},
  solution_weighted_avg::FirstOrderLp.SolutionWeightedAverage,
  eps_primal_infeasible::Float64,
  eps_dual_infeasible::Float64,
  params::PdhgParameters,
)
  scaled_infeasibility = Dict{FirstOrderLp.PointType,Float64}()
  for point_type in instances(FirstOrderLp.PointType)
    primal_candidate = nothing
    dual_candidate = nothing
    if point_type == FirstOrderLp.POINT_TYPE_AVERAGE_ITERATE
      primal_candidate, dual_candidate =
        FirstOrderLp.compute_average(solution_weighted_avg)
    elseif point_type == FirstOrderLp.POINT_TYPE_CURRENT_ITERATE
      primal_candidate = primal_solution
      dual_candidate = dual_solution
    elseif point_type == FirstOrderLp.POINT_TYPE_ITERATE_DIFFERENCE
      primal_candidate = primal_delta
      dual_candidate = dual_delta
    else
      continue
    end
    infeasibility_information = FirstOrderLp.compute_infeasibility_information(
      problem,
      primal_candidate,
      dual_candidate,
      point_type,
    )
    scaled_infeasibility[point_type] =
      infeasibility_information.max_dual_ray_infeasibility /
      infeasibility_information.dual_ray_objective
    if params.verbosity && mod(iteration - 1, params.printing_frequency) == 0
      infeasibility_log(iteration, infeasibility_information, point_type)
    end
    if haskey(first_infeasibility_detection, point_type)
      continue
    end
    if FirstOrderLp.primal_infeasibility_criteria_met(
      eps_primal_infeasible,
      infeasibility_information,
    ) || FirstOrderLp.dual_infeasibility_criteria_met(
      eps_dual_infeasible,
      infeasibility_information,
    )
      infeasibility_log(iteration, infeasibility_information, point_type)
      first_infeasibility_detection[point_type] = iteration
      Printf.@printf(
        "   %5d Infeasibility detected using %s\n",
        iteration,
        string(point_type)
      )
    end
  end
  return scaled_infeasibility
end

"""
Counts the changes of primal active constraints for a given primal vector.
"""
function count_primal_active_constraint_changes(
  primal::Vector{Float64},
  next_primal::Vector{Float64},
  problem::FirstOrderLp.QuadraticProgrammingProblem,
)
  counter = 0
  for idx in 1:length(primal)
    counter +=
      (
        (primal[idx] == problem.variable_upper_bound[idx]) !=
        (next_primal[idx] == problem.variable_upper_bound[idx])
      ) || (
        (primal[idx] == problem.variable_lower_bound[idx]) !=
        (next_primal[idx] == problem.variable_lower_bound[idx])
      )
  end
  return counter
end

""" Counts the number of dual active constraints for a given dual vector."""
function count_dual_active_constraint_changes(
  dual::Vector{Float64},
  next_dual::Vector{Float64},
  problem::FirstOrderLp.QuadraticProgrammingProblem,
)
  counter = 0
  for idx in FirstOrderLp.inequality_range(problem)
    dual_active = 0
    next_dual_active = 0
    if dual[idx] == 0.0
      dual_active = 1
    end
    if next_dual[idx] == 0.0
      next_dual_active = 1
    end
    counter += abs(dual_active - next_dual_active)
  end
  return counter
end

function take_pdhg_step(
  problem::FirstOrderLp.QuadraticProgrammingProblem,
  current_primal_solution::Vector{Float64},
  current_dual_solution::Vector{Float64},
  primal_weight::Float64,
  step_size::Float64,
)
  # The next lines compute the primal portion of the PDHG algorithm:
  # argmin_x [g(x) + current_dual_solution' K x
  #          + (0.5 * primal_weight / step_size)
  #             * norm(x - current_primal_solution)^2]
  # See Sections 2-3 of Chambolle and Pock and the comment above
  # SimplePdhgParameters.
  # This minimization is easy to do in closed form since it can be separated
  # into independent problems for each of the primal variables. The
  # projection onto the primal feasibility set comes from the closed form
  # for the above minimization and the cases where g(x) is infinite - there
  # isn't officially any projection step in the algorithm.
  primal_gradient = FirstOrderLp.compute_primal_gradient(
    problem,
    current_primal_solution,
    current_dual_solution,
  )
  next_primal =
    current_primal_solution .- primal_gradient * (step_size / primal_weight)
  FirstOrderLp.project_primal!(next_primal, problem)

  # The next two lines compute the dual portion:
  # argmin_y [H*(y) - y' K (2.0*next_primal - current_primal_solution)
  #           + (0.5 / (primal_weight * step_size))
  #.             * norm(y-current_dual_solution)^2]
  dual_gradient = FirstOrderLp.compute_dual_gradient(
    problem,
    2.0 * next_primal - current_primal_solution,
  )
  next_dual =
    current_dual_solution .+ dual_gradient * (step_size * primal_weight)
  FirstOrderLp.project_dual!(next_dual, problem)

  return next_primal, next_dual
end


"""
`optimize(params::PdhgParameters,
          problem::QuadraticProgrammingProblem)`

Solves a linear program using primal-dual hybrid gradient. If the step_size
specified in params is negative, picks a step size that ensures
step_size^2 * norm(constraint_matrix)^2 < 1,
a condition that guarantees provable convergence.

# Arguments
- `params::PdhgParameters`: parameters.
- `original_problem::QuadraticProgrammingProblem`: the QP to solve.

# Returns
A SaddlePointOutput struct containing the solution found.
"""
function optimize(
  params::PdhgParameters,
  problem::FirstOrderLp.QuadraticProgrammingProblem,
  initial_primal_solution::Vector{Float64},
  initial_dual_solution::Vector{Float64},
)
  primal_size = length(problem.variable_lower_bound)
  dual_size = length(problem.right_hand_side)

  primal_weight = params.primal_weight
  reference_point = params.reference_point

  if isnothing(params.step_size)
    desired_relative_error = 0.2
    maximum_singular_value, number_of_power_iterations =
      FirstOrderLp.estimate_maximum_singular_value(
        problem.constraint_matrix,
        probability_of_failure = 0.001,
        desired_relative_error = desired_relative_error,
      )
    step_size = (1 - desired_relative_error) / maximum_singular_value
  else
    step_size = params.step_size
  end
  Printf.@printf("Step size %9g Primal weight %9g\n", step_size, primal_weight)

  solution_weighted_avg =
    FirstOrderLp.initialize_solution_weighted_average(primal_size, dual_size)
  # Difference between current_primal_solution and last primal iterate, i.e.
  # x_k - x_{k-1}.
  primal_delta = zeros(primal_size)
  # Difference between current_dual_solution and last dual iterate, i.e. y_k -
  # y_{k-1}
  dual_delta = zeros(dual_size)
  iterations_completed = 0
  iteration_limit = params.iteration_limit
  printing_frequency = params.printing_frequency
  first_infeasibility_detection = Dict{FirstOrderLp.PointType,Int64}()

  stats = PdhgStats(
    Vector{Float64}(undef, iteration_limit),
    Vector{Float64}(undef, iteration_limit),
    Vector{Float64}(undef, iteration_limit),
    Vector{Float64}(undef, iteration_limit),
    Vector{Float64}(undef, iteration_limit),
    Vector{Float64}(undef, iteration_limit),
    Vector{Float64}(undef, iteration_limit),
    Vector{Float64}(undef, iteration_limit),
    Vector{Float64}(undef, iteration_limit),
    Vector{Float64}(undef, iteration_limit),
    (
      isnothing(reference_point) ? nothing :
      Vector{Float64}(undef, iteration_limit)
    ), # distance_to_normalized_current
    (
      isnothing(reference_point) ? nothing :
      Vector{Float64}(undef, iteration_limit)
    ), # distance_to_normalized_average
    (
      isnothing(reference_point) ? nothing :
      Vector{Float64}(undef, iteration_limit)
    ), # distance_to_difference
    Vector{Float64}(undef, iteration_limit),
    Vector{Float64}(undef, iteration_limit),
    Vector{Float64}(undef, iteration_limit),
  )
  primal_iterates = nothing
  dual_iterates = nothing
  if params.store_iterates
    primal_iterates = Vector{Vector{Float64}}(undef, iteration_limit)
    dual_iterates = Vector{Vector{Float64}}(undef, iteration_limit)
  end

  current_primal_solution = initial_primal_solution
  current_dual_solution = initial_dual_solution
  last_active_set_change = 0
  iteration = 0
  while iteration < iteration_limit
    iteration += 1

    if params.store_iterates
      primal_iterates[iteration] = current_primal_solution
      dual_iterates[iteration] = current_dual_solution
    end
    time_spent_doing_basic_algorithm_checkpoint = time()

    next_primal, next_dual = take_pdhg_step(
      problem,
      current_primal_solution,
      current_dual_solution,
      primal_weight,
      step_size,
    )

    # Update deltas and iterates
    primal_delta = next_primal .- current_primal_solution
    dual_delta = next_dual .- current_dual_solution

    FirstOrderLp.add_to_solution_weighted_average(
      solution_weighted_avg,
      next_primal,
      next_dual,
      1.0,
    )
    # Compute stats and log.
    compute_stats(
      problem,
      iteration,
      params.two_dimensional_subspace,
      stats,
      current_primal_solution,
      current_dual_solution,
      primal_delta,
      dual_delta,
      solution_weighted_avg,
    )

    # Check infeasibility. This doesn't stop the run, just reports detection.
    scaled_infeasibility = check_infeasibility(
      problem,
      iteration,
      first_infeasibility_detection,
      current_primal_solution,
      primal_delta,
      current_dual_solution,
      dual_delta,
      solution_weighted_avg,
      params.eps_primal_infeasible,
      params.eps_dual_infeasible,
      params,
    )

    stats.max_scaled_primal_certificate_error_of_normalized_current[iteration] =
      get(scaled_infeasibility, FirstOrderLp.POINT_TYPE_CURRENT_ITERATE, Inf)
    stats.max_scaled_primal_certificate_error_of_normalized_average[iteration] =
      get(scaled_infeasibility, FirstOrderLp.POINT_TYPE_AVERAGE_ITERATE, Inf)
    stats.max_scaled_primal_certificate_error_of_difference[iteration] =
      get(scaled_infeasibility, FirstOrderLp.POINT_TYPE_ITERATE_DIFFERENCE, Inf)

    if !isnothing(reference_point)
      compute_infeasibility_distance_stats(
        iteration,
        stats,
        reference_point,
        current_primal_solution,
        current_dual_solution,
        solution_weighted_avg,
        primal_delta,
        dual_delta,
      )
    end

    if count_primal_active_constraint_changes(
      current_primal_solution,
      next_primal,
      problem,
    ) + count_dual_active_constraint_changes(
      current_dual_solution,
      next_dual,
      problem,
    ) > 0
      last_active_set_change = iteration
    end
    if params.verbosity && mod(iteration - 1, printing_frequency) == 0
      pdhg_log(problem, iteration, stats)
    end

    # update iterates
    current_primal_solution = next_primal
    current_dual_solution = next_dual

    iterations_completed += 1
  end
  return PdhgOutput(
    current_primal_solution,
    current_dual_solution,
    primal_delta,
    dual_delta,
    stats,
    first_infeasibility_detection,
    primal_iterates,
    dual_iterates,
    last_active_set_change,
  )
end

function optimize(
  params::PdhgParameters,
  problem::FirstOrderLp.QuadraticProgrammingProblem,
)
  current_primal_solution = zeros(length(problem.variable_lower_bound))
  current_dual_solution = zeros(length(problem.right_hand_side))

  return optimize(
    params,
    problem,
    current_primal_solution,
    current_dual_solution,
  )
end
