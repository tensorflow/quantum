# TensorFlow Quantum

TensorFlow Quantum (TFQ) is a Python framework for
[quantum machine learning](concepts.md). As an application framework, TFQ allows
quantum algorithm researchers and ML application researchers to leverage
Google’s quantum computing frameworks, all from within TensorFlow.

TensorFlow Quantum focuses on *quantum data* and building *hybrid
quantum-classical models*. It provides tools to interleave quantum algorithms
and logic designed in
<a href="https://github.com/quantumlib/Cirq" class="external">Cirq</a> with
TensorFlow. A basic understanding of quantum computing is required to
effectively use TensorFlow Quantum.

To get started with TensorFlow Quantum, see the [install guide](install.md) and
read through some of the runnable
[notebook tutorials](./tutorials/hello_many_worlds.ipynb).


## Design

TensorFlow Quantum implements the components needed to integrate TensorFlow with
quantum computing hardware. To that end, TensorFlow Quantum introduces two
datatype primitives:

- *Quantum circuit* —This represents a Cirq-defined quantum circuit within
  TensorFlow. Create batches of circuits of varying size, similar to batches of
  different real-valued datapoints.
- *Pauli sum* —Represent linear combinations of tensor products of Pauli
  operators defined in Cirq. Like circuits, create batches of operators of
  varying size.

Using these primitives to represent quantum circuits, TensorFlow Quantum
provides the following operations:

- Sample from output distributions of batches of circuits.
- Calculate the expectation value of batches of Pauli sums on batches of
  circuits. TFQ implements backpropagation-compatible gradient calculation.
- Simulate batches of circuits and states. While inspecting all quantum state
  amplitudes directly throughout a quantum circuit is inefficient at scale in
  the real world, state simulation can help researchers understand how a quantum
  circuit maps states to a near exact level of precision.

Read more about the TensorFlow Quantum implementation in the
[design guide](design.md).


## Report issues

Report bugs or feature requests using the
<a href="https://github.com/tensorflow/quantum/issues" class="external">TensorFlow Quantum issue tracker</a>.
