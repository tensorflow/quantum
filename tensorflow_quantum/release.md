# Release 0.7.0
# Breaking Changes
- Build, compilation, and packaging:
  - The TensorFlow dependency has been upgraded from 2.7.0 to 2.11.0:
  - TensorFlow Quantum is now compiled with `_GLIBCXX_USE_CXX11_ABI=1`. Downstream projects that encounter `std::__cxx11` or `[abi:cxx11]` linker errors will need to adopt this compiler option. See [the GNU C++ Library docs on Dual ABI](https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html).
  - TensorFlow Quantum is now compiled with `-std=c++17`, see [install.md](/docs/install.md) for build instructions.

# Major Features and Improvements
- Significant performance improvements by introducing cuQuantum support for circuit execution on GPU:
  - TensorFlow Quantum Keras layers can now be executed on GPU by setting `use_cuquantum=True` at layer instantiation. Examples:
    - `tfq.layers.Expectation(use_cuquantum=True)`
    - `tfq.layers.SampledExpectation(use_cuquantum=True)` (note that cuQuantum runtime is unsupported for any noisy circuit operations
    - `tfq.layers.State(use_cuquantum=True)`
    - `tfq.layers.Sample(use_cuquantum=True)`
    - `tfq.layers.SimulateSample(use_cuquantum=True)`

# Thanks to our Contributors
This release contains contributions from many people at Google, Nvidia, as well as:
