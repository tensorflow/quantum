# Release 0.8.0
# Breaking Changes
- Build, compilation, and packaging:
  - The TensorFlow dependency has been upgraded from 2.7.0 to 2.11.0:
  - TensorFlow Quantum is now compiled with `_GLIBCXX_USE_CXX11_ABI=1`. Downstream projects that encounter `std::__cxx11` or `[abi:cxx11]` linker errors will need to adopt this compiler option. See [the GNU C++ Library docs on Dual ABI](https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html).
  - TensorFlow Quantum is now compiled with `-std=c++17`, see [install.md](/docs/install.md) for build instructions.

# Major Features and Improvements
- Significant performance improvements by introducing cuQuantum support for circuit execution on Nvidia GPUs:
  - TensorFlow Quantum Keras layers can now be executed on GPU by setting the optional arguement `use_cuquantum=True` at layer instantiation. Examples:
    - `tfq.layers.Expectation(use_cuquantum=True)`
    - `tfq.layers.SampledExpectation(use_cuquantum=True)` (note that cuQuantum runtime is unsupported for any noisy circuit operations
    - `tfq.layers.State(use_cuquantum=True)`
    - `tfq.layers.Sample(use_cuquantum=True)`
    - `tfq.layers.PQC(model_circuit, operators, use_cuquantum=True)`
    - `tfq.layers.ControlledPQC(model_circuit, operators, use_cuquantum=True)`
   - Important notes:
     - CuQuantum execution is currently only supported for source distributions meaning that the user must build TensorFlow Quantum & `tensorFlow-cpu` from source following the instructions in [install.md](/docs/install.md#build-from-source).
        - Ensure that the first entry is "N" in the `configure.sh` script at [this step](/docs/install.md#6-build-the-tensorflow-quantum-pip-package) of building. This ensures that you build upon `tensorflow-cpu` as `tensorflow-gpu` is unnecessary for CuQuantum support in TensorFlow Quantum.
     - The cuQuantum SDK must be installed locally. See [installation instructions](https://docs.nvidia.com/cuda/cuquantum/custatevec/getting_started.html) for details. As part of the installation process, ensure that the `CUQUANTUM_ROOT` environment variable is set (referred to in the installation instructions). If not set, bazel will attempt to automatically locate the folder containing the cuQuantum installation upon running `configure.sh` at [this step](/docs/install.md#6-build-the-tensorflow-quantum-pip-package).
       - Tested on Titan, Ampere and Volta Nvidia GPU architectures. Note that Pascal GPU architectures are not supported, see documentation to [check whether your GPU is compatible with cuQuantum](https://docs.nvidia.com/cuda/cuquantum/getting_started.html#custatevec)
     - Quantum concurrency (global context option) should be turned off when `use_cuquantum=True`. This can be done by running: `tfq.python.quantum_context.set_quantum_concurrent_op_mode(False)`



# Thanks to our Contributors
This release contains contributions from many people at Google, Nvidia, as well as:
