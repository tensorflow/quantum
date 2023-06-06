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
