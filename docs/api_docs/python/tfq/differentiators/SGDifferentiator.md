<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfq.differentiators.SGDifferentiator" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="differentiate_analytic"/>
<meta itemprop="property" content="differentiate_sampled"/>
<meta itemprop="property" content="generate_differentiable_op"/>
<meta itemprop="property" content="refresh"/>
</div>

# tfq.differentiators.SGDifferentiator

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/quantum/tree/master/tensorflow_quantum/python/differentiators/stochastic_differentiator.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



## Class `SGDifferentiator`

Stochastic generator based differentiator class.

Inherits From: [`Differentiator`](../../tfq/differentiators/Differentiator.md)

<!-- Placeholder for "Used in" -->
SGDifferentiator allows you to get the sampled gradient value from three
different stochastic processes:
- parameter coordinate sampling
    Choose one of the symbols of the given programs and perform coordinate
    descent optimization.
    e.g. if a program has parameters ['a','b','c'], choose 'a' w.r.t given
        probability and get the partial derivative of the direction 'a' only
- parameter-shift rule generators sampling
    e.g. Given symbols, there could be many operators sharing the same
        symbol, X**'a', Y**'a', Z**'a'. Choose Y**'a' w.r.t given
        probability and get the partial derivative of the generator.
- cost Hamiltonian sampling
    e.g. if there are cost Hamiltonians such as ['Z1',Z2',Z3'], then choose
        'Z2' w.r.t given probability and get the partial derivative of the
        Hamiltonian observable only.
and the expectation value of the sampled gradient value converges into
the true ground truth gradient value.
This Stochastic Generator Differentiator is the modified gradient estimator
of the following two papers:
- [arXiv:1901.05374](https://arxiv.org/abs/1901.05374), Harrow et al.
- [arXiv:1910.01155](https://arxiv.org/abs/1910.01155), Sweke et al.

```
>>> # Get an expectation op.
>>> my_op = tfq.get_expectation_op()
>>> # Attach a differentiator.
>>> my_dif = tfq.differentiators.SGDifferentiator()
>>> op = my_dif.generate_differentiable_op(
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
>>> # This value is now computed via the stochastic processes described in:
>>> # https://arxiv.org/abs/1901.05374
>>> # https://arxiv.org/abs/1910.01155
>>> grads = g.gradient(expectations, symbol_values_tensor)
>>> # the result is non-deterministic in general, but in this special case,
>>> # it has only one result.
>>> grads
<tf.Tensor: shape=(1, 1), dtype=float32, numpy=[[-1.1839752]]>
```

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/quantum/tree/master/tensorflow_quantum/python/differentiators/stochastic_differentiator.py">View source</a>

``` python
__init__(
    stochastic_coordinate=True,
    stochastic_generator=True,
    stochastic_cost=True,
    uniform_sampling=False
)
```

Instantiate this differentiator.
Create a SGDifferentiator.
Args:
    stochastic_coordinate: Python `bool` to determine if
        sampling on coordinate is performed or not. Default to True.
    stochastic_generator: Python `bool` to determine if
        sampling on generator is performed or not. Default to True.
    stochastic_cost: Python `bool` to determine if sampling on
        cost Hamiltonian is performed or not. Default to True.
    uniform_sampling: Python `bool` to determine the
        probabilistic distributions on the sampling targets.
        Default to False.



## Methods

<h3 id="differentiate_analytic"><code>differentiate_analytic</code></h3>

<a target="_blank" href="https://github.com/tensorflow/quantum/tree/master/tensorflow_quantum/python/differentiators/stochastic_differentiator.py">View source</a>

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

Compute the sampled gradient with cascaded stochastic processes.
The gradient calculations follows the following steps:
1. Compute the decomposition of the incoming circuits so that we have
    their generator information (done using cirq in a tf.py_function)
2. Construct probability distributions & perform stochastic processes
    to select parameter-shift terms.
    - Stochastic generator : sampling on parameter-shifted gates.
    - Stochastic coordinate : sampling on symbols.
    - Stochastic cost : sampling on pauli sums
3. Sum up terms and reshape for the total gradient that is compatible
    with tensorflow differentiation.
Args:
    programs: `tf.Tensor` of strings with shape [n_programs] containing
        the string representations of the circuits to be executed.
    symbol_names: `tf.Tensor` of strings with shape [n_symbols], which
        is used to specify the order in which the values in
        `symbol_values` should be placed inside of the circuits in
        `programs`.
    symbol_values: `tf.Tensor` of real numbers with shape
        [n_programs, n_symbols] specifying parameter values to resolve
        into the circuits specified by programs, following the ordering
        dictated by `symbol_names`.
    pauli_sums : `tf.Tensor` of strings with shape [n_programs, n_ops]
        representing output observables for each program.
    forward_pass_vals : `tf.Tensor` of real numbers for forward pass
        values with the shape of [n_programs, n_ops]
    grad : `tf.Tensor` of real numbers for backpropagated gradient
        values from the upper layer with the shape of
        [n_programs, n_ops]
Returns:
    A `tf.Tensor` of real numbers for sampled gradients from the above
    samplers with the shape of [n_programs, n_symbols]

<h3 id="differentiate_sampled"><code>differentiate_sampled</code></h3>

<a target="_blank" href="https://github.com/tensorflow/quantum/tree/master/tensorflow_quantum/python/differentiators/stochastic_differentiator.py">View source</a>

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

Compute the sampled gradient with cascaded stochastic processes.
The gradient calculations follows the following steps:
1. Compute the decomposition of the incoming circuits so that we have
    their generator information (done using cirq in a tf.py_function)
2. Construct probability distributions & perform stochastic processes
    to select parameter-shift terms.
    - Stochastic generator : sampling on parameter-shifted gates.
    - Stochastic coordinate : sampling on symbols.
    - Stochastic cost : sampling on pauli sums
3. Sum up terms and reshape for the total gradient that is compatible
    with tensorflow differentiation.
Args:
    programs: `tf.Tensor` of strings with shape [n_programs] containing
        the string representations of the circuits to be executed.
    symbol_names: `tf.Tensor` of strings with shape [n_symbols], which
        is used to specify the order in which the values in
        `symbol_values` should be placed inside of the circuits in
        `programs`.
    symbol_values: `tf.Tensor` of real numbers with shape
        [n_programs, n_symbols] specifying parameter values to resolve
        into the circuits specified by programs, following the ordering
        dictated by `symbol_names`.
    num_samples: `tf.Tensor` of positive integers representing the
        number of samples per term in each term of pauli_sums used
        during the forward pass.
    pauli_sums : `tf.Tensor` of strings with shape [n_programs, n_ops]
        representing output observables for each program.
    forward_pass_vals : `tf.Tensor` of real numbers for forward pass
        values with the shape of [n_programs, n_ops]
    grad : `tf.Tensor` of real numbers for backpropagated gradient
        values from the upper layer with the shape of
        [n_programs, n_ops]
Returns:
    A `tf.Tensor` of real numbers for sampled gradients from the above
    samplers with the shape of [n_programs, n_symbols]

<h3 id="generate_differentiable_op"><code>generate_differentiable_op</code></h3>

<a target="_blank" href="https://github.com/tensorflow/quantum/tree/master/tensorflow_quantum/python/differentiators/differentiator.py">View source</a>

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

<a target="_blank" href="https://github.com/tensorflow/quantum/tree/master/tensorflow_quantum/python/differentiators/differentiator.py">View source</a>

``` python
refresh()
```

Refresh this differentiator in order to use it with other ops.




