<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfq.util" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfq.util

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/quantum/tree/master/tensorflow_quantum/python/util.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



A collection of helper functions that are useful several places in tfq.



## Functions

[`check_commutability(...)`](../tfq/util/check_commutability.md): Return False if at least one pair of terms in pauli_sum is not

[`convert_to_tensor(...)`](../tfq/util/convert_to_tensor.md): Convert lists of tfq supported primitives to tensor representations.

[`exp_identity(...)`](../tfq/util/exp_identity.md): Return a circuit for exponentiating an identity gate.

[`exponential(...)`](../tfq/util/exponential.md): Return a Cirq circuit with exponential forms of operators.

[`from_tensor(...)`](../tfq/util/from_tensor.md): Convert a tensor of tfq primitives back to Python objects.

[`get_circuit_symbols(...)`](../tfq/util/get_circuit_symbols.md): Returns a list of the sympy.Symbols that are present in `circuit`.

[`get_supported_gates(...)`](../tfq/util/get_supported_gates.md): A helper to get the gates supported by tfq.

[`kwargs_cartesian_product(...)`](../tfq/util/kwargs_cartesian_product.md): Compute the cartesian product of inputs yielding Python `dict`s.

[`random_circuit_resolver_batch(...)`](../tfq/util/random_circuit_resolver_batch.md): Generate a batch of random circuits and symbolless resolvers.

[`random_pauli_sums(...)`](../tfq/util/random_pauli_sums.md): Generate a list of random cirq pauli sums of length |n_sums|.

[`random_symbol_circuit(...)`](../tfq/util/random_symbol_circuit.md): Generate a random circuit including some parameterized gates.

[`random_symbol_circuit_resolver_batch(...)`](../tfq/util/random_symbol_circuit_resolver_batch.md): Generate a batch of random circuits and resolvers.

