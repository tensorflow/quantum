<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfq.differentiators.ForwardDifference" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="differentiate_analytic"/>
<meta itemprop="property" content="differentiate_sampled"/>
<meta itemprop="property" content="generate_differentiable_op"/>
<meta itemprop="property" content="refresh"/>
</div>

# tfq.differentiators.ForwardDifference

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/TFQuantum/tree/master/tensorflow_quantum/python/differentiators/linear_combination.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



## Class `ForwardDifference`

Differentiate a circuit using forward differencing.

Inherits From: [`LinearCombination`](../../tfq/differentiators/LinearCombination.md)

<!-- Placeholder for "Used in" -->

Forward differencing computes a derivative at a point x using only
points larger than x (in this way, it is 'one sided'). A closed form for
the coefficients of this derivative for an arbitrary positive error order
is used here, which is described in the following article:
https://www.sciencedirect.com/science/article/pii/S0377042799000886.


```

>>> my_op = tfq.get_expectation_op()
>>> linear_differentiator = tfq.differentiators.ForwardDifference(2, 0.01)
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
...     expectations = op(circuit, ['alpha'], symbol_values_tensor, psums)
>>> # Gradient would be: -50 * f(x + 0.02) +  200 * f(x + 0.01) - 150 * f(x)
>>> grads = g.gradient(expectations, symbol_values_tensor)
>>> grads
tf.Tensor([[-1.184372]], shape=(1, 1), dtype=float32)

```

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/quantumlib/TFQuantum/tree/master/tensorflow_quantum/python/differentiators/linear_combination.py">View source</a>

``` python
__init__(
    error_order=1,
    grid_spacing=0.001
)
```

Instantiate a ForwardDifference.

Create a ForwardDifference differentiator, passing along an error order
and grid spacing to be used to contstruct differentiator coeffecients.

#### Args:


* <b>`error_order`</b>: A positive `int` specifying the error order of this
    differentiator. This corresponds to the smallest power
    of `grid_spacing` remaining in the series that was truncated
    to generate this finite differencing expression.
* <b>`grid_spacing`</b>: A positive `float` specifying how large of a
    grid to use in calculating this finite difference.



## Methods

<h3 id="differentiate_analytic"><code>differentiate_analytic</code></h3>

<a target="_blank" href="https://github.com/quantumlib/TFQuantum/tree/master/tensorflow_quantum/python/differentiators/linear_combination.py">View source</a>

``` python
differentiate_analytic(
    programs,
    symbol_names,
    symbol_values,
    pauli_sums,
    forward_pass_vals,
    grad
)
```




<h3 id="differentiate_sampled"><code>differentiate_sampled</code></h3>

<a target="_blank" href="https://github.com/quantumlib/TFQuantum/tree/master/tensorflow_quantum/python/differentiators/linear_combination.py">View source</a>

``` python
differentiate_sampled(
    programs,
    symbol_names,
    symbol_values,
    pauli_sums,
    num_samples,
    forward_pass_vals,
    grad
)
```




<h3 id="generate_differentiable_op"><code>generate_differentiable_op</code></h3>

<a target="_blank" href="https://github.com/quantumlib/TFQuantum/tree/master/tensorflow_quantum/python/differentiators/differentiator.py">View source</a>

``` python
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

<a target="_blank" href="https://github.com/quantumlib/TFQuantum/tree/master/tensorflow_quantum/python/differentiators/differentiator.py">View source</a>

``` python
refresh()
```

Refresh this differentiator in order to use it with other ops.




