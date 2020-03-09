<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfq.differentiators.ParameterShift" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="differentiate_analytic"/>
<meta itemprop="property" content="differentiate_sampled"/>
<meta itemprop="property" content="generate_differentiable_op"/>
<meta itemprop="property" content="refresh"/>
</div>

# tfq.differentiators.ParameterShift

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/quantum/tree/master/tensorflow_quantum/python/differentiators/parameter_shift.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Calculate the general version of parameter-shift rule based gradients.

Inherits From: [`Differentiator`](../../tfq/differentiators/Differentiator.md)

<!-- Placeholder for "Used in" -->

This ParameterShift is the gradient estimator of the following paper:

[arXiv:1905.13311](https://arxiv.org/abs/1905.13311), Gavin E. Crooks.

This ParameterShift is used for any programs with parameterized gates.
It internally decomposes any programs into array of gates with at most
two distinct eigenvalues.

```
>>> non_diff_op = tfq.get_expectation_op()
>>> linear_differentiator = tfq.differentiators.ParameterShift()
>>> # Get an expectation op, with this differentiator attached.
>>> op = linear_differentiator.generate_differentiable_op(
...     analytic_op=non_diff_op
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
>>> # This value is now computed via the ParameterShift rule.
>>> # https://arxiv.org/abs/1905.13311
>>> grads = g.gradient(expectations, symbol_values_tensor)
>>> grads
tf.Tensor([[-1.1839752]], shape=(1, 1), dtype=float32)
```

## Methods

<h3 id="differentiate_analytic"><code>differentiate_analytic</code></h3>

<a target="_blank" href="https://github.com/tensorflow/quantum/tree/master/tensorflow_quantum/python/differentiators/parameter_shift.py">View source</a>

```python
differentiate_analytic(
    programs, symbol_names, symbol_values, pauli_sums, forward_pass_vals, grad
)
```

Calculate the gradient.

The gradient calculations follows the following steps:

1. Compute the decomposition of the incoming circuits so that we have
    their generator information (done using cirq in a tf.py_function)
2. Use formula (31) from paper inside of TensorFlow to calculate
    gradients from all the decomposed circuits.
3. Sum up terms and reshape for the total gradient that is compatible
    with TensorFlow.

**CAUTION**
Analytic gradient measurements based on this ParameterShift generally
run at least K(=2) times SLOWER than the original circuit.
On top of it, since all parameters of gates are shifted individually,
the time complexity is linear in the number of parameterized gates L.
So, you will see O(KL) slower time & space complexity than the original
forward pass measurements.

#### Args:


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
* <b>`forward_pass_vals`</b>: `tf.Tensor` of real numbers with shape
    [batch_size, n_ops] containing the output of the forward pass
    through the op you are differentiating.
* <b>`grad`</b>: `tf.Tensor` of real numbers with shape [batch_size, n_ops]
    representing the gradient backpropagated to the output of the
    op you are differentiating through.


#### Returns:

Backward gradient values for each program & each pauli sum. It has
the shape of [batch_size, n_symbols].


<h3 id="differentiate_sampled"><code>differentiate_sampled</code></h3>

<a target="_blank" href="https://github.com/tensorflow/quantum/tree/master/tensorflow_quantum/python/differentiators/parameter_shift.py">View source</a>

```python
differentiate_sampled(
    programs, symbol_names, symbol_values, pauli_sums, num_samples,
    forward_pass_vals, grad
)
```

Calculate the gradient.

The gradient calculations follows the following steps:

1. Compute the decomposition of the incoming circuits so that we have
    their generator information (done using cirq in a tf.py_function)
2. Use formula (31) from paper inside of TensorFlow to calculate
    gradients from all the decomposed circuits.
3. Sum up terms and reshape for the total gradient that is compatible
    with TensorFlow.

**CAUTION**
Analytic gradient measurements based on this ParameterShift generally
run at least K(=2) times SLOW than the original circuit.
On top of it, since all parameters of gates are shifted individually,
the time complexity is linear in the number of parameterized gates L.
So, you will see O(KL) slower time & space complexity than the original
forward pass measurements.

#### Args:


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
* <b>`num_samples`</b>: `tf.Tensor` of positiver integers indicating the number
    of samples used per term to calculate the expectation value
    in the forward pass.
* <b>`forward_pass_vals`</b>: `tf.Tensor` of real numbers with shape
    [batch_size, n_ops] containing the output of the forward pass
    through the op you are differentiating.
* <b>`grad`</b>: `tf.Tensor` of real numbers with shape [batch_size, n_ops]
    representing the gradient backpropagated to the output of the
    op you are differentiating through.


#### Returns:

Backward gradient values for each program & each pauli sum. It has
the shape of [batch_size, n_symbols].


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




