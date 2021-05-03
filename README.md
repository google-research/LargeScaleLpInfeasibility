# Infeasibility detection with primal-dual hybrid gradient

## Introduction

This repository contains the code accompanying the paper "Infeasibility
detection with primal-dual hybrid gradient for large-scale linear programming"
\[1\]. It includes a simple implementation of PDHG and scripts to run all the
numerical experiments appearing in the paper.

## Obtaining the data

1.  Download and compile the EMPS program to uncompress compressed MPS files
    (from https://www.netlib.org/lp/data/emps.c)
2.  Download netlib LP infeasibility instances from
    https://www.netlib.org/lp/infeas/index.html
3.  Use emps to uncompress the compressed MPS files.

For example:

```shell
$ curl -O https://netlib.org/lp/data/emps.c
$ cc -O -o emps emps.c
$ curl -O "https://netlib.org/lp/infeas/{bgdbg1,bgetam,bgindy,bgprtr,box1,\
ceria3d,chemcom,cplex1,cplex2,ex72a,ex73a,forest6,galenet,gosh,gran,greenbea,\
itest2,itest6,klein1,klein2,klein3,mondou2,pang,pilot4i,qual,reactor,refinery,\
vol1,woodinfe}"
$ for f in bgdbg1 bgetam bgindy bgprtr box1 ceria3d chemcom cplex1 cplex2 \
   ex72a ex73a forest6 galenet gosh gran greenbea itest2 itest6 klein1 klein2 \
   klein3 mondou2 pang pilot4i qual reactor refinery vol1 woodinfe; do
   ./emps "${f}" > "${f}.mps"
done
```

## Running

Use the following scripts to run the experiments. The scripts use Julia 1.6.0.
All commands below assume that the current directory is the working directory.

A one-time step is required to set up the necessary packages on the local
machine:

```shell
$ julia --project -e 'import Pkg; Pkg.instantiate()'
```

This needs to be run again only if the dependencies change.

Note: The first time running the code on a machine will be unusually slow while
Julia compiles and caches all of the package dependencies.

### `run_low_dimensional_example.jl`

To generate the plots in Figure 1 of \[1\]:

```shell
$ julia --project run_low_dimensional_example.jl output_dir
```

This generates eight pdf files in output_dir:

*   `pfdfp.pdf`: Primal feasible, dual feasible, plotting primal variable
    values,
*   `pfdfd.pdf`: Primal feasible, dual feasible, plotting dual variable values,
*   `pfdip.pdf`: Primal feasible, dual infeasible, plotting primal variable
    values,
*   `pfdid.pdf`: Primal feasible, dual infeasible, plotting dual variable
    values,
*   `pidfp.pdf`: Primal infeasible, dual feasible, plotting primal variable
    values,
*   `pidfd.pdf`: Primal infeasible, dual feasible, plotting dual variable
    values,
*   `pidip.pdf`: Primal infeasible, dual infeasible, plotting primal variable
    values,
*   `pidid.pdf`: Primal infeasible, dual infeasible, plotting dual variable
    values.

### `run_netlib_identifiability_experiment.jl`

To generate the plots in Figure 2 of \[1\]:

```shell
$ julia --project run_netlib_identifiability_experiment.jl output_dir netlib_data_dir
```

### `run_netlib_infeasibility_experiment.jl`

To generate the plots in Figure 3 of \[1\]:

```shell
$ julia --project run_netlib_infeasibility_experiment.jl output_dir netlib_data_dir instance_name
```

`instance_name` is the name of the instance, without the trailing ".mps", for
example, `bgdbg1`.

## Auto-formatting Julia code

A one-time step is required to use the auto-formatter:

```shell
$ julia --project=formatter -e 'import Pkg; Pkg.instantiate()'
```

Run the following command to auto-format all Julia code in this directory before
submitting changes:

```shell
$ julia --project=formatter -e 'using JuliaFormatter; format(".")'
```

## References

\[1\]: David Applegate, Mateo Díaz, Haihao Lu, and Miles Lubin. "Infeasibility
detection with primal-dual hybrid gradient for large-scale linear programming".
[arXiv](https://arxiv.org/abs/2102.04592)

## Disclaimer

This is not an official Google product.
