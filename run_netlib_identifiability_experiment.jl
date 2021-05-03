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
# Executes the identifiability experiement with some instances of the netlib
# database.
# To run this script execute
# $ julia --project run_netlib_identifiability_experiment.jl <path_where_to_store_results> <path_to_directory_containing_mps_files>
#
# By default the script runs a specific set of twelve instances. The script also
# optionally accepts a specific instance to run and whether to generate the
# iteration plot based on assuming differences are better than average and
# current iterates or not, for example,
# $ julia --project run_netlib_identifiability_experiment.jl <path_where_to_store_results> <path_to_directory_containing_mps_files> box1 true
# Note that you must specify either both the instance name and the boolean, or
# neither of them.

import FirstOrderLp

include("primal_dual_hybrid_gradient.jl")
include("utils.jl")

function run_netlib_instance(instance_name::String, difference_is_better::Bool)
  params = PdhgParameters(
    nothing, # step_size (forces the solver to use a provably correct step size)
    1.0, # primal_weight
    1000, # printing frequency
    false, # verbose
    1000000, # iteration limit
    nothing, # two_dimensional_subspace
    0.0, # eps_primal_infeasible
    0.0, # eps_dual_infeasible
    false, # store_iterates
    nothing, # reference_point
  )
  print("About to start solving " * instance_name * "\n")
  instance_path = joinpath(ARGS[2], (instance_name * ".mps"))
  lp = FirstOrderLp.qps_reader_to_standard_form(instance_path)
  output = optimize(params, lp)
  factor = difference_is_better ? 10 : 2
  params = PdhgParameters(
    nothing, # step_size (forces the solver to use a provably correct step size)
    1.0, # primal_weight
    1000, # printing frequency
    false, # verbose
    min(1000000, max(2000, factor * output.last_active_set_change)), # iteration limit
    nothing, # two_dimensional_subspace
    0.0, # eps_primal_infeasible
    0.0, # eps_dual_infeasible
    false, # store_iterates
    [output.primal_delta; output.dual_delta], # reference_point
  )
  output_with_distances = optimize(params, lp)
  generate_infeasibility_convergence_plot(
    output_with_distances,
    joinpath(ARGS[1], instance_name * ".pdf"),
  )
end

function main()
  if length(ARGS) == 4
    run_netlib_instance(ARGS[3], parse(Bool, ARGS[4]))
  else
    for difference_is_better in [false, true]
      if difference_is_better
        instance_names =
          ["box1", "galenet", "itest2", "itest6", "woodinfe", "ex72a", "ex73a"]
      else
        instance_names = ["bgdbg1", "bgindy", "cplex1", "chemcom", "mondou2"]
      end

      for instance_name in instance_names
        run_netlib_instance(instance_name, difference_is_better)
      end
    end
  end
end

main()
