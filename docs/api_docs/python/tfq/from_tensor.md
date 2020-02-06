<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfq.from_tensor" />
<meta itemprop="path" content="Stable" />
</div>

# tfq.from_tensor

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/TFQuantum/tree/master/tensorflow_quantum/python/util.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Convert a tensor of tfq primitives back to Python objects.

``` python
tfq.from_tensor(tensor_to_convert)
```



<!-- Placeholder for "Used in" -->

Convert a tensor representing `cirq.PauliSum` or `cirq.Circuit`
objects back to Python objects.


```

>>> my_qubits = cirq.GridQubit.rect(1, 2)
>>> my_circuits = [cirq.Circuit(cirq.X(my_qubits[0])),
...                cirq.Circuit(cirq.Z(my_qubits[0]))
... ]
>>> tensor_input = tfq.convert_to_tensor(my_circuits)
>>> # Now tensor_input can be used as model input etc.
>>> same_circuits = tfq.from_tensor(tensor_input)
>>> # same_circuits now holds cirq.Circuit objects once more.
>>> same_circuits
[cirq.Circuit([
    cirq.Moment(operations=[
        cirq.X.on(cirq.GridQubit(0, 0)),
    ]),
])
 cirq.Circuit([
    cirq.Moment(operations=[
        cirq.Z.on(cirq.GridQubit(0, 0)),
    ]),
])]

```

#### Args:


* <b>`tensor_to_convert`</b>: `tf.Tensor` or `np.ndarray` representation to
    convert back into python objects.


#### Returns:

Python `list` of items converted to their python representation stored
    in a (potentially nested) `list`.
