# Tensorflow Quantum Benchmarks

## Testing instructions
Benchmarks are currently tested separately from the main repository. To run
a benchmark testcase, simply _run_ the benchmark file like with any other unit test:
```
bazel run <options> benchmarks/scripts:<benchmark file>
```

## Instructions to run
A benchmark can be run from the command line or a bash script by setting model
parameters via flags, separated from Bazel flags by a `--` delimiter. To run a benchmark with a set of specific parameters, use
the following command template:
```
bazel run <options> benchmarks/scripts:<benchmark file> -- <parameter values> --benchmarks=<method names>
```
Some notes on benchmark configuration:
  - "all" is a valid option for the `benchmarks` flag, and will result in all benchmarks methods associated with that file to run.
  - If a benchmark method runs twice with identical configurations, the most recent run will overwrite previous reports.
  - For information on valid parameter flags and their descriptions see `flags.py`


### Sample benchmark experiments

For example, to benchmark a dense depth-10 Clifford circuit over 5 qubits call:
```
bazel run -c opt --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=1" --cxxopt="-msse2" \
  --cxxopt="-msse3" --cxxopt="-msse4" \
  benchmarks/scripts:benchmark_clifford_circuit -- \
  --n_moments 5 --n_qubits 4 \
  --benchmarks=benchmark_clifford_circuit_eager
```
This will produce a proto benchmark report under `benchmarks/reports` corresponding to the chosen parameters:
```
benchmarks/scripts/reports/CliffordBenchmarks.benchmark_clifford_circuit_4_5_1
```


To benchmark the parameter shift differentiation method on a random depth-10 4-qubit circuit with 10 parameters call, where the circuit will be differentiated
over 50 trials, each time over a batch of 10 circuits.
```
bazel run -c opt --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=1" --cxxopt="-msse2" \
  --cxxopt="-msse3" --cxxopt="-msse4" \
  benchmarks/scripts:benchmark_op_gradients -- \
  --n_moments 10 --n_qubits 4 --n_symbols 10 \
  --n_runs 50 --batch_size 10 \
  --benchmarks=benchmark_parameter_shift
```

