<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfq.util.exponential" />
<meta itemprop="path" content="Stable" />
</div>

# tfq.util.exponential

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/quantum/tree/master/tensorflow_quantum/python/util.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Return a Cirq circuit with exponential forms of operators.

```python
tfq.util.exponential(
    operators, coefficients=None
)
```



<!-- Placeholder for "Used in" -->

Construct an exponential form of given `operators` and `coefficients`.
Operators to be exponentiated are specified in `operators` as
`cirq.PauliSum` or `cirq.PauliString`. Parameters are given by
`coefficients`.

Note that only operators whose standard representations consist of terms
which all commute can be exponentiated.  This allows use of the identity
exp(A+B+...) = exp(A)exp(B)... else there would need to be automatic
handling of Trotterization and convergence, which is not supported yet.

#### Args:


* <b>`operators`</b>: Python `list` of `cirq.PauliSum` or `cirq.PauliString` object
    to be exponentiated. Here are simple examples.
    Let q = cirq.GridQubit(0, 0)
    E.g. operator = 0.5 * X(q) -> exp(-i * 0.5 * X(q))
         operator = 0.5 * cirq.PauliString({q: cirq.I})
                   -> exp(-i * 0.5)*np.eye(2)
    Be careful of the negation and the PauliString of the identity gate.
* <b>`coefficients`</b>: (Optional) Python `list` of Python `str`, `float` or
    `sympy.Symbol` object of parameters. Defaults to None, then all
    coefficients of `operators` are set to 1.0.

#### Returns:

A `cirq.Circuit` containing exponential form of given `operators`
    and `coefficients`.
