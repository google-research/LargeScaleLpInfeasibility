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
# Computes a performance profile for a set of instances from the netlib
# infeasibility database.
# To run this script execute
# $ julia --project performance_netlib_experiment.jl <path_where_to_store_results> <path_to_directory_containing_mps_files>

import FirstOrderLp

include("primal_dual_hybrid_gradient.jl")
include("utils.jl")

function run_identifiability_profile(
  instance_names::Vector{String},
  accuracy::Float64,
)
  outputs = Vector{PdhgOutput}()
  for instance_name in instance_names
    params = PdhgParameters(
      nothing, # step_size (forces the solver to use a provably correct step size)
      1.0, # primal_weight
      1000, # printing frequency
      false, # verbose
      1000000, # iteration limit
      nothing, # two_dimensional_subspace
      accuracy, # eps_primal_infasible
      accuracy, # eps_dual_infeasible
      false, # store_iterates
      nothing, # reference_point
    )
    print("About to start solving " * instance_name * "\n")
    instance_path = joinpath(ARGS[2], (instance_name * ".mps"))
    lp = FirstOrderLp.qps_reader_to_standard_form(instance_path)
    push!(outputs, optimize(params, lp))
  end
  instance_path =
    joinpath(ARGS[1], "per_instance_detection" * string(accuracy) * ".csv")
  path_csv = joinpath(ARGS[1], "detected_fraction" * string(accuracy) * ".csv")
  save_fraction_of_infeasible_instances_detected_to_csv(
    outputs,
    instance_names,
    instance_path,
    path_csv,
  )
  generate_plot_fraction_infeasible_instances_detected(
    path_csv,
    joinpath(ARGS[1], "performance_" * string(accuracy) * ".pdf"),
  )
end

function main()
  instance_names = [
    "bgdbg1",
    "bgetam",
    "bgindy",
    "bgprtr",
    "box1",
    "ceria3d",
    "chemcom",
    "cplex1",
    "cplex2",
    "ex72a",
    "ex73a",
    "forest6",
    "galenet",
    "gosh",
    "gran",
    "greenbea",
    "itest2",
    "itest6",
    "klein1",
    "klein2",
    "klein3",
    "mondou2",
    "pang",
    "qual",
    "reactor",
    "refinery",
    "vol1",
    "woodinfe",
  ]
  run_identifiability_profile(instance_names, 1e-4)
  run_identifiability_profile(instance_names, 1e-8)
end

main()
