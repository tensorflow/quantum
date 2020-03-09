<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfq.util.get_circuit_symbols" />
<meta itemprop="path" content="Stable" />
</div>

# tfq.util.get_circuit_symbols

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/quantum/tree/master/tensorflow_quantum/python/util.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Returns a list of the sympy.Symbols that are present in `circuit`.

```python
tfq.util.get_circuit_symbols(
    circuit
)
```



<!-- Placeholder for "Used in" -->


#### Args:


* <b>`circuit`</b>: A `cirq.Circuit` object.


#### Returns:

Python `list` containing the symbols found in the circuit.
