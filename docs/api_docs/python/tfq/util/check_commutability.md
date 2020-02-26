<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfq.util.check_commutability" />
<meta itemprop="path" content="Stable" />
</div>

# tfq.util.check_commutability

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/quantum/tree/master/tensorflow_quantum/python/util.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Return False if at least one pair of terms in pauli_sum is not

```python
tfq.util.check_commutability(
    pauli_sum
)
```



<!-- Placeholder for "Used in" -->
commutable.

#### Args:


* <b>`pauli_sum`</b>: `cirq.PauliSum` object to be checked if all of terms inside
    are commutable each other.