<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfq.get_expectation_op" />
<meta itemprop="path" content="Stable" />
</div>

# tfq.get_expectation_op

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/quantum/tree/master/tensorflow_quantum/core/ops/circuit_execution_ops.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Get a TensorFlow op that will calculate batches of expectation values.

```python
tfq.get_expectation_op(
    backend=None
)
```



<!-- Placeholder for "Used in" -->

This function produces a non-differentiable TF op that will calculate
batches of expectation values given tensor batches of `cirq.Circuit`s,
parameter values, and `cirq.PauliSum` operators to measure.


```

>>> # Simulate circuits with C++.
>>> my_op = tfq.get_expectation_op()
>>> # Prepare some inputs.
>>> qubit = cirq.GridQubit(0, 0)
>>> my_symbol = sympy.Symbol('alpha')
>>> my_circuit_tensor = tfq.convert_to_tensor([
...     cirq.Circuit(cirq.H(qubit) ** my_symbol)
... ])
>>> my_values = np.array([[0.123]])
>>> my_paulis = tfq.convert_to_tensor([[
...     3.5 * cirq.X(qubit) - 2.2 * cirq.Y(qubit)
... ]])
>>> # This op can now be run with:
>>> output = my_op(
...     my_circuit_tensor, ['alpha'], my_values, my_paulis)
>>> output
tf.Tensor([[0.71530885]], shape=(1, 1), dtype=float32)

```


In order to make the op differentiable, a `tfq.differentiator` object is
needed. see <a href="../tfq/differentiators.md"><code>tfq.differentiators</code></a> for more details. Below is a simple
example of how to make my_op from the above code block differentiable:

```
>>> diff = tfq.differentiators.ForwardDifference()
>>> my_differentiable_op = diff.generate_differentiable_op(
...     analytic_op=my_op
... )
```


#### Args:


* <b>`backend`</b>: Optional Python `object` that specifies what backend this op
should use when evaluating circuits. Can be any
`cirq.SimulatesFinalState`. If not provided the default C++ analytical
expectation calculation op is returned.


#### Returns:

A `callable` with the following signature:

```op(programs, symbol_names, symbol_values, pauli_sums)```


* <b>`programs`</b>: `tf.Tensor` of strings with shape [batch_size] containing
    the string representations of the circuits to be executed.
* <b>`symbol_names`</b>: `tf.Tensor` of strings with shape [n_params], which
    is used to specify the order in which the values in
    `symbol_values` should be placed inside of the circuits in
    `programs`.
* <b>`symbol_values`</b>: `tf.Tensor` of real numbers with shape
    [batch_size, n_params] specifying parameter values to resolve
    into the circuits specified by programs, following the ordering
    dictated by `symbol_names`.
* <b>`pauli_sums`</b>: `tf.Tensor` of strings with shape [batch_size, n_ops]
    containing the string representation of the operators that will
    be used on all of the circuits in the expectation calculations.

* <b>`Returns`</b>:     `tf.Tensor` with shape [batch_size, n_ops] that holds the
        expectation value for each circuit with each op applied to it
        (after resolving the corresponding parameters in).