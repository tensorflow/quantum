# TensorFlow Quantum design and concepts

In October 2019,
<a href="https://www.blog.google/perspectives/sundar-pichai/what-our-quantum-computing-milestone-means/" class="external">Google announced</a>
they achieved
<a href="https://www.nature.com/articles/s41586-019-1666-5" class="external">quantum supremacy</a>.
Using 53&nbsp;*noisy*&nbsp;qubits, this demonstration was a critical first step to unlock
the full potential of quantum computing and marks the beginning of the
<a href="https://quantum-journal.org/papers/q-2018-08-06-79/" class="external">Noisy Intermediate-Scale Quantum</a>&nbsp;(NISQ)
computing era. In the coming years, quantum devices with tens-to-hundreds of
noisy qubits are expected to become a reality. So what is possible with these
devices?

There are many ideas for leveraging NISQ quantum computing including
optimization, quantum simulation, cryptography, and machine learning.
TensorFlow&nbsp;Quantum&nbsp;(TFQ) is designed to help researchers experiment
with these ideas. Researchers create and run *quantum circuits*. It integrates
with TensorFlow, an established machine learning framework used for research and
production. TensorFlow Quantum provides flexible and performant tools and
constructs used by quantum machine learning researchers. TensorFlow Quantum
hopes to bridge the quantum and classical machine learning communities—and
enrich both with new perspectives and ideas.

## NISQ quantum machine learning

During the NISQ-era, quantum algorithms with known speedups over classical
algorithms—like
<a href="https://arxiv.org/abs/quant-ph/9508027" class="external">Shor's factoring algorithm</a> or
<a href="https://arxiv.org/abs/quant-ph/9605043" class="external">Grover's search algorithm</a>—are
not yet possible at a meaningful scale.

A goal of TensorFlow Quantum is to help discover algorithms for the
NISQ-era, with particular interest in:

1. *Use classical machine learning to enhance NISQ algorithms.* The hope is that
   techniques from classical machine learning can enhance our understanding of
   quantum computing. For example,
   <a href="https://arxiv.org/abs/1907.05415" class="external">this paper</a>
   shows a recurrent neural network (RNN) used to discover that optimization of
   the control parameters for algorithms like the QAOA and VQE are more efficient
   than simple off the shelf optimizers. And
   <a href="https://www.nature.com/articles/s41534-019-0141-3" class="external">this paper</a>
   uses reinforcement learning to help mitigate errors and produce higher
   quality quantum gates.
2. *Model quantum data with quantum circuits.* Classically modeling quantum data
   is possible if you have an exact description of the datasource—but sometimes
   this isn’t possible. To solve this problem, you can try modeling on the
   quantum computer itself and measure/observe the important statistics.
   <a href="https://www.nature.com/articles/s41567-019-0648-8" class="external">This paper</a>
   shows a quantum circuit designed with a structure analogous to a
   convolutional neural network (CNN) to detect different topological phases of
   matter. The quantum computer holds the data and the model. The classical
   processor sees only measurement samples from the model output and never the
   data itself. In
   <a href="https://arxiv.org/pdf/1711.07500.pdf" class="external">this paper</a>
   the authors learn to compress information about quantum many-body systems
   using a DMERA model.

Other areas of interest in quantum machine learning include:

1. Modeling purely classical data on quantum computers.
2. Quantum-inspired classical algorithms. TFQ does not contain any purely
   classical algorithms that are quantum-inspired.

While these last two areas did not inform the design of TensorFlow Quantum,
you can still use TFQ for research here. For example, in
<a href="https://arxiv.org/abs/1802.06002" class="external">this paper</a>
the authors use a quantum computer to solve some purely classical data problems—
which could be implemented in TFQ.


## Software components

TensorFlow Quantum is designed for the problems of NISQ-era quantum machine
learning. Integration with [TensorFlow](https://www.tensorflow.org/overview) and
[Keras](https://www.tensorflow.org/guide/keras/overview) is seamless and
performant. And the `tfq.datasets` module allows researchers to experiment and
converse about new and interesting quantum datasets.

### Primitives

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

With these primitives, TFQ can build the functionality to merge quantum
computing with TensorFlow.

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
used in the quantum supremacy experiment were used as verifiers and extended for
the foundation of TFQ ops (all written with AVX2 and SSE instructions). Ops with
identical functional signatures were created that use a physical quantum
computer. Switching between a simulated and physical quantum computer is as easy
as changing a single line of code. These ops are located in the
`circuit_execution_ops.py` in `tensorflow_quantum/core/ops/`.

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

As the field of quantum computing grows, more and more quantum data and model
combinations will arise, making structured comparison more difficult. The
`tfq.datasets` module is used as the data source for quantum machine learning
tasks. It ensures structured comparisons for the model and performance.

It is hoped that with large community contributions, the `tfq.datasets` module
will grow to enable research that is more transparent and reproducible.
Carefully curated problems in: quantum control, fermionic simulation,
classification near phase transitions, quantum sensing, etc are all great
candidates for addition to `tfq.datasets`. To propose a new dataset open
a <a href="https://github.com/tensorflow/quantum/issues">GitHub issue</a>.
