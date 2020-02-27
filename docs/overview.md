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


## Usage

To provide some intuition for using quantum data, consider a supervised
classification of quantum states using a quantum neural network. Just like
classical ML, a challenge of quantum ML is to classify "noisy data". To
build and train such a model, the researcher can do the following:

1. *Prepare a quantum dataset* —Quantum data is loaded as tensors (a
   multi-dimensional array of numbers), which becomes an edge within
   TensorFlow’s computational graph. Each quantum data tensor is specified as a
   quantum circuit written in Cirq that generates quantum data on the fly. This
   avoids the need for quantum memory which has not yet been manufactured.
2. *Evaluate a quantum neural network model* —The researcher prototypes a
   quantum neural network using Cirq that they will embed inside of a TensorFlow
   compute graph. Parameterized quantum models can be selected from several
   broad categories based on the structure of the quantum data. The goal of the
   model is to perform quantum processing in order to extract information hidden
   in a typically entangled state. In other words, the quantum model
   disentangles the input quantum data, leaving the hidden information encoded
   in classical correlations—making it accessible to local measurements and
   classical post-processing.
3. *Sample or average* —Measurement of quantum states extracts classical
   information in the form of samples from a classical random variable. The
   distribution of values from this random variable generally depends on the
   quantum state itself and on the measured observable. As many variational
   algorithms depend on mean values of measurements, also known as expectation
   values, TFQ provides methods for averaging over several runs involving steps
   (1) and (2).
4. *Evaluate a classical neural networks model* —Once classical information is
   extracted, it is in a format ready for further classical post-processing.
   Since the extracted information may still be encoded in classical
   correlations between measured expectations, classical deep neural networks
   can be applied to distill such correlations.
5. *Evaluate cost function* —A cost is calculated using the results of classical
   post-processing. This could be based on how accurate the model performs the
   classification task if the quantum data was labeled, or other criteria if the
   task is unsupervised.
6. *Evaluate gradients and update parameters* —After evaluating the cost
   function, the free parameters in the pipeline are updated in a direction
   expected to decrease the cost. This is most commonly performed using gradient
   descent.


## Report issues

Report bugs or feature requests using the
<a href="https://github.com/tensorflow/quantum/issues" class="external">TensorFlow Quantum issue tracker</a>.
