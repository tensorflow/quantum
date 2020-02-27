<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfq.differentiators.LinearCombination" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="differentiate_analytic"/>
<meta itemprop="property" content="differentiate_sampled"/>
<meta itemprop="property" content="generate_differentiable_op"/>
<meta itemprop="property" content="refresh"/>
</div>

# tfq.differentiators.LinearCombination

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/quantum/tree/master/tensorflow_quantum/python/differentiators/linear_combination.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Differentiate a circuit with respect to its inputs by

Inherits From: [`Differentiator`](../../tfq/differentiators/Differentiator.md)

```python
tfq.differentiators.LinearCombination(
    weights, perturbations
)
```



<!-- Placeholder for "Used in" -->
linearly combining values obtained by evaluating the op using parameter
values perturbed about their forward-pass values.


```

>>> my_op = tfq.get_expectation_op()
>>> weights = [5, 6, 7]
>>> perturbations = [0, 0.5, 0.25]
>>> linear_differentiator = tfq.differentiators.LinearCombination(
...    weights, perturbations
... )
>>> # Get an expectation op, with this differentiator attached.
>>> op = linear_differentiator.generate_differentiable_op(
...     analytic_op=my_op
... )
>>> qubit = cirq.GridQubit(0, 0)
>>> circuit = tfq.convert_to_tensor([
...     cirq.Circuit(cirq.X(qubit) ** sympy.Symbol('alpha'))
... ])
>>> psums = tfq.convert_to_tensor([[cirq.Z(qubit)]])
>>> symbol_values_array = np.array([[0.123]], dtype=np.float32)
>>> # Calculate tfq gradient.
>>> symbol_values_tensor = tf.convert_to_tensor(symbol_values_array)
>>> with tf.GradientTape() as g:
...     g.watch(symbol_values_tensor)
...     expectations = op(circuit, ['alpha'], symbol_values_tensor, psums
... )
>>> # Gradient would be: 5 * f(x+0) + 6 * f(x+0.5) + 7 * f(x+0.25)
>>> grads = g.gradient(expectations, symbol_values_tensor)
>>> # Note: this gradient visn't correct in value, but showcases
>>> # the principle of how gradients can be defined in a very flexible
>>> # fashion.
>>> grads
tf.Tensor([[5.089467]], shape=(1, 1), dtype=float32)

```

#### Args:


* <b>`weights`</b>: Python `list` of real numbers representing linear
    combination coeffecients for each perturbed function
    evaluation.
* <b>`perturbations`</b>: Python `list` of real numbers representing
    perturbation values.

## Methods

<h3 id="differentiate_analytic"><code>differentiate_analytic</code></h3>

<a target="_blank" href="https://github.com/tensorflow/quantum/tree/master/tensorflow_quantum/python/differentiators/linear_combination.py">View source</a>

```python
differentiate_analytic(
    programs, symbol_names, symbol_values, pauli_sums, forward_pass_vals, grad
)
```




<h3 id="differentiate_sampled"><code>differentiate_sampled</code></h3>

<a target="_blank" href="https://github.com/tensorflow/quantum/tree/master/tensorflow_quantum/python/differentiators/linear_combination.py">View source</a>

```python
differentiate_sampled(
    programs, symbol_names, symbol_values, pauli_sums, num_samples,
    forward_pass_vals, grad
)
```




<h3 id="generate_differentiable_op"><code>generate_differentiable_op</code></h3>

<a target="_blank" href="https://github.com/tensorflow/quantum/tree/master/tensorflow_quantum/python/differentiators/differentiator.py">View source</a>

```python
generate_differentiable_op()
```

Generate a differentiable op by attaching self to an op.

This function returns a `tf.function` that passes values through to
`forward_op` during the forward pass and this differentiator (`self`) to
backpropagate through the op during the backward pass. If sampled_op
is provided the differentiators `differentiate_sampled` method will
be invoked (which requires sampled_op to be a sample based expectation
op with num_samples input tensor). If analytic_op is provided the
differentiators `differentiate_analytic` method will be invoked (which
requires analytic_op to be an analytic based expectation op that does
NOT have num_samples as an input). If both sampled_op and analytic_op
are provided an exception will be raised.

***CAUTION***

This `generate_differentiable_op()` can be called only ONCE because
of the `one differentiator per op` policy. You need to call `refresh()`
to reuse this differentiator with another op.

#### Args:


* <b>`sampled_op`</b>: A `callable` op that you want to make differentiable
    using this differentiator's `differentiate_sampled` method.
* <b>`analytic_op`</b>: A `callable` op that you want to make differentiable
    using this differentiators `differentiate_analytic` method.


#### Returns:

A `callable` op that who's gradients are now registered to be
a call to this differentiators `differentiate_*` function.


<h3 id="refresh"><code>refresh</code></h3>

<a target="_blank" href="https://github.com/tensorflow/quantum/tree/master/tensorflow_quantum/python/differentiators/differentiator.py">View source</a>

```python
refresh()
```

Refresh this differentiator in order to use it with other ops.




