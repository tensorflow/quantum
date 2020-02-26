<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfq.differentiators.Differentiator" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="differentiate_analytic"/>
<meta itemprop="property" content="differentiate_sampled"/>
<meta itemprop="property" content="generate_differentiable_op"/>
<meta itemprop="property" content="refresh"/>
</div>

# tfq.differentiators.Differentiator

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/quantum/tree/master/tensorflow_quantum/python/differentiators/differentiator.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Interface that defines how to specify gradients for a quantum circuit.

<!-- Placeholder for "Used in" -->

This abstract class allows for the creation of gradient calculation
procedures for (expectation values from) quantum circuits, with
respect to a set of input parameter values. This allows one
to backpropagate through a quantum circuit.

## Methods

<h3 id="differentiate_analytic"><code>differentiate_analytic</code></h3>

<a target="_blank" href="https://github.com/tensorflow/quantum/tree/master/tensorflow_quantum/python/differentiators/differentiator.py">View source</a>

```python
differentiate_analytic(
    programs, symbol_names, symbol_values, pauli_sums, forward_pass_vals, grad
)
```

Specify how to differentiate a circuit with analytical expectation.

This is called at graph runtime by TensorFlow. `differentiate_analytic`
should calculate the gradient of a batch of circuits and return it
formatted as indicated below. See
<a href="../../tfq/differentiators/ForwardDifference.md"><code>tfq.differentiators.ForwardDifference</code></a> for an example.

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

A `tf.Tensor` with the same shape as `symbol_values` representing
the gradient backpropageted to the `symbol_values` input of the op
you are differentiating through.


<h3 id="differentiate_sampled"><code>differentiate_sampled</code></h3>

<a target="_blank" href="https://github.com/tensorflow/quantum/tree/master/tensorflow_quantum/python/differentiators/differentiator.py">View source</a>

```python
differentiate_sampled(
    programs, symbol_names, symbol_values, pauli_sums, num_samples,
    forward_pass_vals, grad
)
```

Specify how to differentiate a circuit with sampled expectation.

This is called at graph runtime by TensorFlow. `differentiate_sampled`
should calculate the gradient of a batch of circuits and return it
formatted as indicated below. See
<a href="../../tfq/differentiators/ForwardDifference.md"><code>tfq.differentiators.ForwardDifference</code></a> for an example.

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
* <b>`num_samples`</b>: `tf.Tensor` of positive integers representing the
    number of samples per term in each term of pauli_sums used
    during the forward pass.
* <b>`forward_pass_vals`</b>: `tf.Tensor` of real numbers with shape
    [batch_size, n_ops] containing the output of the forward pass
    through the op you are differentiating.
* <b>`grad`</b>: `tf.Tensor` of real numbers with shape [batch_size, n_ops]
    representing the gradient backpropagated to the output of the
    op you are differentiating through.


#### Returns:

A `tf.Tensor` with the same shape as `symbol_values` representing
the gradient backpropageted to the `symbol_values` input of the op
you are differentiating through.


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




