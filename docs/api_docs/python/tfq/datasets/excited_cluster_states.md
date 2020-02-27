<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfq.datasets.excited_cluster_states" />
<meta itemprop="path" content="Stable" />
</div>

# tfq.datasets.excited_cluster_states

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/quantum/tree/master/tensorflow_quantum/datasets/cluster_state.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Return a tuple of potentially excited cluster states and their labels.

```python
tfq.datasets.excited_cluster_states(
    qubits
)
```



<!-- Placeholder for "Used in" -->

For every qubit in `qubits` this method will create a cluster state circuit
on `qubits`, apply a `cirq.X` on that qubit along with a label of 1 and add
it to the return dataset. Finally a cluster state circuit on `qubits` that
doesn't contain any `cirq.X` gates with a label of -1 will be added to the
returned dataset.

Note: This is a toy dataset that can serve as guidance for the community
to contribute new datasets to TensorFlow Quantum.


```

>>> circuits, labels = tfq.datasets.excited_cluster_states(
...     cirq.GridQubit.rect(1, 3)
... )
>>> print(circuits[0])
(0, 0): ───H───@───────@───X───
               │       │
(0, 1): ───H───@───@───┼───────
                   │   │
(0, 2): ───H───────@───@───────
>>> labels[0]
1
>>> print(circuits[-1])
(0, 0): ───H───@───────@───
               │       │
(0, 1): ───H───@───@───┼───
                   │   │
(0, 2): ───H───────@───@───
>>> labels[-1]
-1

```


Circuits that feature a `cirq.X` gate on one of the qubits are labeled 1,
while the circuit that doesn't feature a `cirq.X` anywhere has the label -1.


#### Args:


* <b>`qubits`</b>: Python `list` of `cirq.GridQubit`s on which the excited cluster
    state dataset will be created.


#### Returns:

A `tuple` of `cirq.Circuit`s and Python `int` labels.
