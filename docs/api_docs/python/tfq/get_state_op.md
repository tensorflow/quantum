<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfq.get_state_op" />
<meta itemprop="path" content="Stable" />
</div>

# tfq.get_state_op

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/quantum/tree/master/tensorflow_quantum/core/ops/circuit_execution_ops.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Get a TensorFlow op that produces states from given quantum circuits.

```python
tfq.get_state_op(
    backend=None
)
```



<!-- Placeholder for "Used in" -->

This function produces a non-differentiable op that will calculate
batches of state tensors given tensor batches of `cirq.Circuit`s and
parameter values.


```

>>> # Simulate circuits with cirq.
>>> my_op = tfq.get_state_op(backend=cirq.DensityMatrixSimulator())
>>> # Simulate circuits with C++.
>>> my_second_op = tfq.get_state_op()
>>> # Prepare some inputs.
>>> qubit = cirq.GridQubit(0, 0)
>>> my_symbol = sympy.Symbol('alpha')
>>> my_circuit_tensor = tfq.convert_to_tensor([
...     cirq.Circuit(cirq.Y(qubit) ** my_symbol)
... ])
>>> my_values = np.array([[0.5]])
>>> # This op can now be run to calculate the state.
>>> output = my_second_op(my_circuit_tensor, ['alpha'], my_values)
>>> output
<tf.RaggedTensor [[(0.5+0.5j), (0.5+0.5j)]]>

```


#### Args:


* <b>`backend`</b>: Optional Python `object` that specifies what backend this op
    should use when evaluating circuits. Can be any
    `cirq.SimulatesFinalState`. If not provided, the default C++
    wavefunction simulator will be used.


#### Returns:

A `callable` with the following signature:

```op(programs, symbol_names, symbol_values)```


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

* <b>`Returns`</b>:     `tf.Tensor` with shape [batch_size, <ragged> size of state] that
    contains the state information of the circuit.