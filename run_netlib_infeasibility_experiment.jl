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
# $ julia --project run_netlib_infeasibility_experiment.jl <path_where_to_store_results> <path_to_directory_containing_mps_files>
#
# By default the script runs a specific set of twelve instances. The script also
# optionally accepts a specific instance to run, for example,
# $ julia --project run_netlib_infeasibility_experiment.jl <path_where_to_store_results> <path_to_directory_containing_mps_files> box1

import FirstOrderLp

include("primal_dual_hybrid_gradient.jl")
include("utils.jl")

function run_netlib_instance(instance_name::String)
  params = PdhgParameters(
    nothing, # step_size (forces the solver to use a provably correct step size)
    1.0, # primal_weight
    1000, # printing frequency
    false, # verbose
    1000000, # iteration limit
    nothing, # two_dimensional_subspace
    1e-8, # eps_primal_infeasible
    1e-8, # eps_dual_infeasible
    false, # store_iterates
    nothing, # reference_point
  )
  print("About to start solving " * instance_name * "\n")
  instance_path = joinpath(ARGS[2], (instance_name * ".mps"))
  lp = FirstOrderLp.qps_reader_to_standard_form(instance_path)
  output = optimize(params, lp)
  generate_infeasibility_error_plot(
    output,
    joinpath(ARGS[1], instance_name * "-error.pdf"),
  )
end

function main()
  if length(ARGS) == 3
    run_netlib_instance(ARGS[3])
  else
    for instance_name in [
      "box1",
      "galenet",
      "itest2",
      "itest6",
      "woodinfe",
      "ex72a",
      "ex73a",
      "bgdbg1",
      "bgindy",
      "cplex1",
      "chemcom",
      "mondou2",
    ]
      run_netlib_instance(instance_name)
    end
  end
end

main()
