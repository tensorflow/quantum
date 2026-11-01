# TensorFlow Quantum design

TensorFlow Quantum (TFQ) is designed for the problems of NISQ-era quantum
machine learning. It brings quantum computing primitives—like building quantum
circuits—to the TensorFlow ecosystem. Models and operations built with
TensorFlow use these primitives to create powerful quantum-classical hybrid
systems.

Using TFQ, researchers can construct a TensorFlow graph using a quantum dataset,
a quantum model, and classical control parameters. These are all represented as
tensors in a single computational graph. The outcome of quantum
measurements—leading to classical probabilistic events—is obtained by TensorFlow
ops. Training is done with the standard
[Keras](https://www.tensorflow.org/guide/keras/overview) API. The `tfq.datasets`
module allows researchers to experiment with new and interesting quantum
datasets.


## Cirq

<a href="https://github.com/quantumlib/Cirq" class="external">Cirq</a> is a
quantum programming framework from Google. It provides all of the basic
operations—such as qubits, gates, circuits, and measurement—to create, modify
and invoke quantum circuits on a quantum computer, or a simulated quantum
computer. TensorFlow Quantum uses these Cirq primitives to extend TensorFlow for
batch computation, model building, and gradient computation. To be effective
with TensorFlow Quantum, it’s a good idea to be effective with Cirq.


## TensorFlow Quantum primitives

TensorFlow Quantum implements the components needed to integrate TensorFlow with
quantum computing hardware. To that end, TFQ introduces two datatype primitives:

- *Quantum circuit*: This represents
  <a href="https://github.com/quantumlib/Cirq" class="external">Cirq</a>-defined
  quantum circuits (`cirq.Circuit`) within TensorFlow. Create batches of
  circuits of varying size, similar to batches of different real-valued
  datapoints.
- *Pauli sum*: Represent linear combinations of tensor products of Pauli
  operators defined in Cirq (`cirq.PauliSum`). Like circuits, create batches of
  operators of varying size.

### Fundamental ops

Using the quantum circuit primitives within a `tf.Tensor`, TensorFlow Quantum
implements ops that process these circuits and produce meaningful outputs.

The TensorFlow ops are written in optimized C++. These ops sample from
circuits, calculate expectation values, and output the state produced by the
given circuits. Writing ops that are flexible and performant has some
challenges:

1. Circuits are not the same size. For simulated circuits, you are unable to
   create static operations (like `tf.matmul` or `tf.add`) and then substitute
   different numbers for circuits of different sizes. These ops must allow for
   dynamic sizes that the statically sized TensorFlow compute graph doesn't
   allow.
2. Quantum data can induce a different circuit structure altogether. This is
   another reason to support dynamic sizes in the TFQ ops. Quantum data can
   represent a structural change to the underlying quantum state that is
   represented by modifications to the original circuit. As new datapoints are
   swapped in and out at runtime, the TensorFlow compute graph can not be
   modified after it is built, so support for these varying structures is
   required.
3. `cirq.Circuits` are similar to compute graphs in that they are a series of
   operations—and some might contain symbols/placeholders. It is important to
   make this as compatible with TensorFlow as possible.

For performance reasons, Eigen (the C++ library used in many TensorFlow ops) is
not well suited for quantum circuit simulation. Instead, the circuit simulators
used in the
<a href="https://ai.googleblog.com/2019/10/quantum-supremacy-using-programmable.html" class="external">quantum beyond-classical experiment</a>
are used as verifiers and extended as the foundation of TFQ ops (all written
with AVX2 and SSE instructions). Ops with identical functional signatures were
created that use a physical quantum computer. Switching between a simulated and
physical quantum computer is as easy as changing a single line of code. These
ops are located in the
<a href="https://github.com/tensorflow/quantum/blob/master/tensorflow_quantum/core/ops/circuit_execution_ops.py" class="external"><code>circuit_execution_ops.py</code></a>.

### Layers

TensorFlow Quantum layers expose sampling, expectation, and state calculation to
developers using the `tf.keras.layers.Layer` interface. It's convenient to
create a circuit layer for classical control parameters or for readout
operations. Additionally, you can create a layer with a high degree of
complexity supporting batch circuit, batch control parameter value, and perform
batch readout operations. See `tfq.layers.Sample` for an example.

### Differentiators

Unlike many TensorFlow operations, observables in quantum circuits do not have
formulas for gradients that are relatively easy to calculate. This is because a
classical computer can only read samples from the circuits that are run on a
quantum computer.

To solve this problem, the `tfq.differentiators` module provides several
standard differentiation techniques. Users can also define their own method to
compute gradients—in both the “real world” setting of sample-based expectation
calculation, and the analytic exact world. Methods like finite difference are
often the fastest (wall clock time) in an analytic/exact environment. While
slower (wall clock time), more practical methods like
<a href="https://arxiv.org/abs/1811.11184" class="external">parameter shift</a> or
<a href="https://arxiv.org/abs/1901.05374" class="external">stochastic methods</a>
are often more effective. A `tfq.differentiators.Differentiator` is instantiated
and attached to an existing op with `generate_differentiable_op`, or passed to
the constructor of `tfq.layers.Expectation` or `tfq.layers.SampledExpectation`.
To implement a custom differentiator, inherit from the
`tfq.differentiators.Differentiator` class. To define a gradient operation for
sampling or state vector calculation, use `tf.custom_gradient`.

### Datasets

As the field of quantum computing grows, more quantum data and model
combinations will arise, making structured comparison more difficult. The
`tfq.datasets` module is used as the data source for quantum machine learning
tasks. It ensures structured comparisons for the model and performance.

It is hoped that with large community contributions, the `tfq.datasets` module
will grow to enable research that is more transparent and reproducible.
Carefully curated problems in: quantum control, fermionic simulation,
classification near phase transitions, quantum sensing, etc are all great
candidates for addition to `tfq.datasets`. To propose a new dataset open
a <a href="https://github.com/tensorflow/quantum/issues">GitHub issue</a>.

## Configuring execution backends

Every circuit-execution layer and low-level execution op in TensorFlow Quantum
accepts an optional `backend` argument that controls how quantum circuits are
evaluated. By default, TFQ uses its built-in, multi-threaded C++ simulators
(powered by
<a href="https://github.com/quantumlib/qsim" class="external">qsim</a>),
which execute directly inside the TensorFlow runtime. You can also configure
TFQ to use noisy C++ trajectory simulation, pure-Python Cirq simulators such as
`cirq.Simulator` or `cirq.DensityMatrixSimulator`, or any custom
`cirq.Sampler` (including noisy samplers and quantum hardware interfaces).

### Backend compatibility across layers and ops

Different TFQ layers and ops require different simulation capabilities
depending on whether they compute exact expectation values, sample-based
expectation values, raw bitstring samples, or full quantum states:

| Layer or op | Default `backend` | Built-in C++ options | Custom Cirq `backend` interface | Default differentiator |
| :--- | :--- | :--- | :--- | :--- |
| `tfq.layers.PQC` (`repetitions=None`) | `'noiseless'` | `'noiseless'`, `None` | `cirq.sim.simulator.SimulatesExpectationValues` | `Adjoint` (C++) / `ParameterShift` (Cirq) |
| `tfq.layers.PQC` (`repetitions=<int>`) | `'noiseless'` | `'noiseless'`, `None` | `cirq.Sampler` | `ParameterShift` |
| `tfq.layers.ControlledPQC` (`repetitions=None`) | `'noiseless'` | `'noiseless'`, `None` | `cirq.sim.simulator.SimulatesExpectationValues` | `Adjoint` (C++) / `ParameterShift` (Cirq) |
| `tfq.layers.ControlledPQC` (`repetitions=<int>`) | `'noiseless'` | `'noiseless'`, `None` | `cirq.Sampler` | `ParameterShift` |
| `tfq.layers.NoisyPQC` | `'noisy'` (fixed) | `'noisy'` | Not applicable | `ParameterShift` |
| `tfq.layers.NoisyControlledPQC` | `'noisy'` (fixed) | `'noisy'` | Not applicable | `ParameterShift` |
| `tfq.layers.Expectation` | `'noiseless'` | `'noiseless'`, `'noisy'`, `None` | `cirq.sim.simulator.SimulatesExpectationValues` | `Adjoint` (`'noiseless'`) / `ParameterShift` (other) |
| `tfq.layers.SampledExpectation` | `'noiseless'` | `'noiseless'`, `'noisy'`, `None` | `cirq.Sampler` | `ParameterShift` |
| `tfq.layers.Sample` | `'noiseless'` | `'noiseless'`, `'noisy'`, `None` | `cirq.Sampler` | Not differentiable |
| `tfq.layers.State` | `None` | `None` | `cirq.SimulatesFinalState` | Not differentiable |
| `tfq.get_expectation_op` | `None` | `None` | `cirq.sim.simulator.SimulatesExpectationValues`, `cirq.DensityMatrixSimulator` | Attach via `Differentiator` |
| `tfq.get_sampled_expectation_op` | `None` | `None` | `cirq.Sampler` | Attach via `Differentiator` |
| `tfq.get_sampling_op` | `None` | `None` | `cirq.Sampler` | Not differentiable |
| `tfq.get_state_op` | `None` | `None` | `cirq.SimulatesFinalState` | Not differentiable |

### Built-in C++ backends

TFQ provides two built-in C++ simulation backends that avoid Python callback
overhead during TensorFlow graph execution:

*   *Noiseless state-vector simulator (`'noiseless'` or `None`)*: The default
    backend for all layers (`backend='noiseless'`) and low-level ops
    (`backend=None`). It evaluates pure quantum states in C++ using vectorized
    SIMD instructions and OpenMP multithreading.
*   *Noisy Monte-Carlo trajectory simulator (`'noisy'`)*: Simulates circuits
    containing noise channels (such as depolarizing, amplitude-damping, or
    phase-damping channels listed in `tfq.util.get_supported_channels()`) by
    averaging over quantum trajectories in C++.
    *   In mid-level layers (`tfq.layers.Expectation`,
        `tfq.layers.SampledExpectation`, and `tfq.layers.Sample`), pass
        `backend='noisy'` at layer construction and provide an integer
        `repetitions` argument when calling the layer.
    *   In high-level layers, `tfq.layers.PQC` and `tfq.layers.ControlledPQC`
        do not accept `backend='noisy'`; instead, use `tfq.layers.NoisyPQC` or
        `tfq.layers.NoisyControlledPQC`, which configure the `'noisy'` C++
        backend automatically and accept `repetitions` and `sample_based`
        arguments.

```python
import cirq
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq

q = cirq.GridQubit(0, 0)
theta = sympy.Symbol('theta')
noisy_circuit = cirq.Circuit(
    cirq.rx(theta)(q),
    cirq.depolarize(p=0.05)(q)
)

# Mid-level Expectation layer using the built-in C++ noisy trajectory backend.
noisy_expectation = tfq.layers.Expectation(backend='noisy')
exp_val = noisy_expectation(
    noisy_circuit,
    symbol_names=[theta],
    symbol_values=[[0.5]],
    operators=[cirq.Z(q)],
    repetitions=1000
)

# High-level NoisyPQC layer (uses the C++ 'noisy' backend internally).
noisy_pqc = tfq.layers.NoisyPQC(
    noisy_circuit,
    operators=cirq.Z(q),
    repetitions=1000,
    sample_based=False
)
pqc_val = noisy_pqc(tfq.convert_to_tensor([cirq.Circuit()]))
```

### Custom Cirq simulators and samplers

To use a Cirq simulator or a custom sampler, pass an instantiated Cirq object
to the `backend` parameter of a layer or `tfq.get_*_op` factory function:

1.  *Analytic expectation (`SimulatesExpectationValues`)*: When computing exact
    expectation values without shot noise (`tfq.layers.Expectation`,
    `tfq.get_expectation_op`, or `PQC` / `ControlledPQC` with
    `repetitions=None`), pass an instance of
    `cirq.sim.simulator.SimulatesExpectationValues` such as `cirq.Simulator()`
    or `cirq.DensityMatrixSimulator()`. Using `cirq.DensityMatrixSimulator`
    computes exact expectation values of noisy circuits from the full density
    matrix without trajectory sampling variance:

    ```python
    q = cirq.GridQubit(0, 0)
    theta = sympy.Symbol('theta')
    circuit = cirq.Circuit(cirq.rx(theta)(q), cirq.depolarize(0.02)(q))

    dm_simulator = cirq.DensityMatrixSimulator()
    pqc_exact_noisy = tfq.layers.PQC(
        circuit,
        operators=cirq.Z(q),
        repetitions=None,
        backend=dm_simulator
    )
    ```

2.  *Sampled expectation and bitstring sampling (`cirq.Sampler`)*: When
    estimating expectation values from finite measurement shots
    (`tfq.layers.SampledExpectation`, `tfq.get_sampled_expectation_op`, or
    `PQC` / `ControlledPQC` with `repetitions=<int>`) or drawing bitstring
    samples (`tfq.layers.Sample`, `tfq.get_sampling_op`), pass any object that
    inherits from `cirq.Sampler`. This includes `cirq.Simulator()`,
    `cirq.DensityMatrixSimulator(noise=...)`, `cirq.ZerosSampler()`, hardware
    samplers, or custom `cirq.Sampler` subclasses:

    ```python
    class CustomNoisySampler(cirq.Sampler):
        def __init__(self, depolarizing_prob=0.01):
            self._sim = cirq.DensityMatrixSimulator(
                noise=cirq.ConstantQubitNoiseModel(
                    cirq.depolarize(depolarizing_prob)
                )
            )

        def run_sweep(self, program, params, repetitions=1):
            return self._sim.run_sweep(program, params, repetitions)

    q = cirq.GridQubit(0, 0)
    theta = sympy.Symbol('theta')
    circuit = cirq.Circuit(cirq.ry(theta)(q))
    sampler = CustomNoisySampler(depolarizing_prob=0.02)

    # Sample-based expectation with a custom cirq.Sampler.
    sampled_pqc = tfq.layers.PQC(
        circuit,
        operators=[cirq.Z(q), cirq.X(q)],
        repetitions=1000,
        backend=sampler
    )

    # Bitstring sampling with a custom cirq.Sampler.
    # TFQ automatically appends terminal measurements across all circuit qubits.
    sample_layer = tfq.layers.Sample(backend=sampler)
    bitstrings = sample_layer(
        circuit,
        symbol_names=[theta],
        symbol_values=[[1.57]],
        repetitions=10
    )
    ```

3.  *Quantum state simulation (`cirq.SimulatesFinalState`)*: `tfq.layers.State`
    and `tfq.get_state_op` accept any `cirq.SimulatesFinalState` instance.
    While the default C++ backend (`backend=None`) and `cirq.Simulator()`
    output state vectors of shape `[batch_size, 2 ** n_qubits]`, passing
    `cirq.DensityMatrixSimulator()` outputs full density matrices of shape
    `[batch_size, 2 ** n_qubits, 2 ** n_qubits]`:

    ```python
    q0, q1 = cirq.GridQubit.rect(1, 2)
    bell = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1))

    dm_state_layer = tfq.layers.State(backend=cirq.DensityMatrixSimulator())
    density_matrices = dm_state_layer([bell])
    ```

### Circuit serialization and concurrency control

When working with custom Cirq backends or hardware samplers, keep two execution
details in mind:

*   *Serialization constraints*: All circuits entering TFQ layers or ops pass
    through `tfq.convert_to_tensor`, which serializes circuits into protocol
    buffers. Consequently, input circuits must use `cirq.GridQubit` or
    `cirq.LineQubit`, must contain only gates and channels listed in
    `tfq.util.get_supported_gates()` and `tfq.util.get_supported_channels()`,
    and must not contain pre-existing terminal measurements (TFQ appends the
    required basis rotations and terminal measurements automatically when
    calling a `cirq.Sampler`). If your target backend requires device-specific
    gates, qubit routing, or compilation outside TFQ's supported gate set,
    perform that compilation inside your custom `cirq.Sampler.run_sweep` or
    `run_batch` method after TFQ deserializes the circuit.
*   *Controlling graph-level concurrency*: By default, TensorFlow may execute
    multiple quantum ops concurrently across independent graph branches. When
    dispatching jobs to a physical quantum processor or a resource-constrained
    backend that cannot handle concurrent requests, serialize quantum op
    execution either globally with `tfq.set_quantum_concurrent_op_mode(False)`
    or per op via `quantum_concurrent=False`:

    ```python
    # Disable concurrent execution globally across all TFQ ops.
    tfq.set_quantum_concurrent_op_mode(False)

    # Or disable concurrency for a single low-level op.
    serial_op = tfq.get_sampled_expectation_op(
        backend=sampler,
        quantum_concurrent=False
    )
    ```

### Performance and differentiation considerations

Using a custom Python `backend` changes both execution performance and default
gradient calculation compared to the built-in `'noiseless'` (`None`) backend:

1.  *Built-in C++ `qsim` vs. Python callback overhead*: The default
    `backend='noiseless'` (or `backend=None`) already runs TFQ's native,
    multi-threaded C++ `qsim` simulator inside the TensorFlow C++ runtime.
    Passing a Python simulator object to `backend`—even `cirq.Simulator()` or
    `qsimcirq.QSimSimulator()`—wraps circuit execution in a `tf.py_function`
    callback, which deserializes every `tf.string` tensor back into a Python
    `cirq.Circuit` on every forward and backward pass. In addition, invoking an
    externally multithreaded simulator such as `qsimcirq.QSimSimulator` inside
    TensorFlow's worker thread pool can cause OpenMP thread contention and
    significantly degrade performance. For noiseless state-vector simulation,
    prefer the default `backend='noiseless'` (`backend=None`).
2.  *Automatic switch from `Adjoint` to `ParameterShift` differentiation*: When
    `backend='noiseless'` (or `None`) and `differentiator=None`, analytic
    expectation layers (`tfq.layers.Expectation`, `tfq.layers.PQC` with
    `repetitions=None`, and `tfq.layers.ControlledPQC` with `repetitions=None`)
    default to `tfq.differentiators.Adjoint()`, which computes exact gradients
    in C++ in a single pass. Because `Adjoint` is only supported by the
    built-in C++ simulator, supplying any custom Cirq `backend` automatically
    switches the default differentiator to
    `tfq.differentiators.ParameterShift()`, which evaluates two shifted
    circuits per parameter and scales linearly with the number of circuit
    symbols. If you use a custom `SimulatesExpectationValues` simulator and
    want finite-difference analytic gradients instead of parameter-shift, pass
    an explicit differentiator such as
    `differentiator=tfq.differentiators.CentralDifference()` or
    `tfq.differentiators.ForwardDifference()`.
