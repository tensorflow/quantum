<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfq.util.convert_to_tensor" />
<meta itemprop="path" content="Stable" />
</div>

# tfq.util.convert_to_tensor

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/quantum/tree/master/tensorflow_quantum/python/util.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Convert lists of tfq supported primitives to tensor representations.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tfq.convert_to_tensor`</p>
</p>
</section>

```python
tfq.util.convert_to_tensor(
    items_to_convert
)
```



<!-- Placeholder for "Used in" -->

Recursively convert a nested lists of `cirq.PauliSum` or `cirq.Circuit`
objects to a `tf.Tensor` representation. Note that cirq serialization only
supports `cirq.GridQubit`s so we also require that input circuits and
pauli sums are defined only on `cirq.GridQubit`s.


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


* <b>`items_to_convert`</b>: Python `list` or nested `list` of `cirq.Circuit`
    or `cirq.Paulisum` objects. Should be rectangular, or this function
    will error.


#### Returns:

`tf.Tensor` that represents the input items.
