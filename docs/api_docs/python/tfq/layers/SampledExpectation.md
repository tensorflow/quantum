<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfq.layers.SampledExpectation" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="activity_regularizer"/>
<meta itemprop="property" content="dtype"/>
<meta itemprop="property" content="dynamic"/>
<meta itemprop="property" content="input"/>
<meta itemprop="property" content="input_mask"/>
<meta itemprop="property" content="input_shape"/>
<meta itemprop="property" content="input_spec"/>
<meta itemprop="property" content="losses"/>
<meta itemprop="property" content="metrics"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="name_scope"/>
<meta itemprop="property" content="non_trainable_variables"/>
<meta itemprop="property" content="non_trainable_weights"/>
<meta itemprop="property" content="output"/>
<meta itemprop="property" content="output_mask"/>
<meta itemprop="property" content="output_shape"/>
<meta itemprop="property" content="submodules"/>
<meta itemprop="property" content="trainable"/>
<meta itemprop="property" content="trainable_variables"/>
<meta itemprop="property" content="trainable_weights"/>
<meta itemprop="property" content="updates"/>
<meta itemprop="property" content="variables"/>
<meta itemprop="property" content="weights"/>
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="build"/>
<meta itemprop="property" content="compute_mask"/>
<meta itemprop="property" content="compute_output_shape"/>
<meta itemprop="property" content="count_params"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
<meta itemprop="property" content="get_input_at"/>
<meta itemprop="property" content="get_input_mask_at"/>
<meta itemprop="property" content="get_input_shape_at"/>
<meta itemprop="property" content="get_losses_for"/>
<meta itemprop="property" content="get_output_at"/>
<meta itemprop="property" content="get_output_mask_at"/>
<meta itemprop="property" content="get_output_shape_at"/>
<meta itemprop="property" content="get_updates_for"/>
<meta itemprop="property" content="get_weights"/>
<meta itemprop="property" content="set_weights"/>
<meta itemprop="property" content="with_name_scope"/>
</div>

# tfq.layers.SampledExpectation

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/quantum/tree/master/tensorflow_quantum/python/layers/circuit_executors/sampled_expectation.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



## Class `SampledExpectation`

A layer that calculates a sampled expectation value.



<!-- Placeholder for "Used in" -->

Given an input circuit and set of parameter values, output expectation
values of observables computed using measurement results sampled from
the input circuit.


First define a simple helper function for generating a parametrized
quantum circuit that we will use throughout:

```
>>> def _gen_single_bit_rotation_problem(bit, symbols):
...     """Generate a toy problem on 1 qubit."""
...     starting_state = [0.123, 0.456, 0.789]
...     circuit = cirq.Circuit(
...         cirq.Rx(starting_state[0])(bit),
...         cirq.Ry(starting_state[1])(bit),
...         cirq.Rz(starting_state[2])(bit),
...         cirq.Rz(symbols[2])(bit),
...         cirq.Ry(symbols[1])(bit),
...         cirq.Rx(symbols[0])(bit)
...     )
...     return circuit
```


In quantum machine learning there are two very common use cases that
align with keras layer constructs. The first is where the circuits
represent the input data points:


```

>>> bit = cirq.GridQubit(0, 0)
>>> symbols = sympy.symbols('x y z')
>>> ops = [-1.0 * cirq.Z(bit), cirq.X(bit) + 2.0 * cirq.Z(bit)]
>>> num_samples = [100, 200]
>>> circuit_list = [
...     _gen_single_bit_rotation_problem(bit, symbols),
...     cirq.Circuit(
...         cirq.Z(bit) ** symbols[0],
...         cirq.X(bit) ** symbols[1],
...         cirq.Z(bit) ** symbols[2]
...     ),
...     cirq.Circuit(
...         cirq.X(bit) ** symbols[0],
...         cirq.Z(bit) ** symbols[1],
...         cirq.X(bit) ** symbols[2]
...     )
... ]
>>> sampled_expectation_layer = tfq.layers.SampledExpectation()
>>> output = sampled_expectation_layer(
...     circuit_list,
...     symbol_names=symbols,
...     operators=ops,
...     repetitions=num_samples)
>>> # Here output[i][j] corresponds to the sampled expectation
>>> # of all the ops in ops w.r.t circuits[i] where Keras managed
>>> # variables are placed in the symbols 'x', 'y', 'z'.
>>> tf.shape(output)
tf.Tensor([3 2], shape=(2,), dtype=int32)

```


Here, different `cirq.Circuit` instances sharing the common symbols 'x',
'y' and 'z' are used as input. Keras uses the `symbol_names`
argument to map Keras managed variables to these circuits constructed
with `sympy.Symbol`s. The shape of `num_samples` is equal to that of `ops`.


The second most common use case is where there is a fixed circuit and
the expectation operators vary:


```

>>> bit = cirq.GridQubit(0, 0)
>>> symbols = sympy.symbols('x, y, z')
>>> ops = [-1.0 * cirq.Z(bit), cirq.X(bit) + 2.0 * cirq.Z(bit)]
>>> fixed_circuit = _gen_single_bit_rotation_problem(bit, symbols)
>>> expectation_layer = tfq.layers.SampledExpectation()
>>> output = expectation_layer(
...     fixed_circuit,
...     symbol_names=symbols,
...     operators=ops,
...     repetitions=5000,
...     initializer=tf.keras.initializers.RandomUniform(0, 2 * np.pi))
>>> # Here output[i][j] corresponds to
>>> # the sampled expectation of operators[i][j] using 5000 samples w.r.t
>>> # the circuit where variable values are managed by keras and store
>>> # numbers in the symbols 'x', 'y', 'z'.
>>> tf.shape(output)
tf.Tensor([1 2], shape=(2,), dtype=int32)

```


Here different `cirq.PauliSum` or `cirq.PauliString` instances can be
used as input to calculate the expectation on the fixed circuit that
the layer was initially constructed with.


There are also some more complex use cases that provide greater flexibility.
Notably these configurations all make use of the `symbol_values` parameter
that causes the `SampledExpectation` layer to stop managing the
`sympy.Symbol`s in the quantum circuits and instead requires the user to
supply inputs themselves. Lets look at the case where there
is a single fixed circuit, some fixed operators and symbols that must be
common to all circuits:


```

>>> bit = cirq.GridQubit(0, 0)
>>> symbols = sympy.symbols('x y z')
>>> ops = [cirq.Z(bit), cirq.X(bit)]
>>> num_samples = [100, 200]
>>> circuit = _gen_single_bit_rotation_problem(bit, symbols)
>>> values = [[1,1,1], [2,2,2], [3,3,3]]
>>> sampled_expectation_layer = tfq.layers.SampledExpectation()
>>> output = sampled_expectation_layer(
...     circuit,
...     symbol_names=symbols,
...     symbol_values=values,
...     operators=ops,
...     repetitions=num_samples)
>>> # output[i][j] = The sampled expectation of ops[j] with
>>> # values_tensor[i] placed into the symbols of the circuit
>>> # with the order specified by feed_in_params.
>>> # so output[1][2] = The sampled expectation of a circuit with parameter
>>> # values [2,2,2] w.r.t Pauli X, estimated using 200 samples per term.
>>> output  # Non-deterministic result. It can vary every time.
tf.Tensor(
[[0.52, 0.72],
 [0.34, 1.  ],
 [0.78, 0.48]], shape=(3, 2), dtype=float32)

```


Tip: you can compare the above result with that of `Expectation`:
tf.Tensor(
[[0.63005245 0.76338404]
 [0.25707167 0.9632684 ]
 [0.79086655 0.5441111 ]], shape=(3, 2), dtype=float32)


Here is a simple model that uses this particular input signature of
<a href="../../tfq/layers/SampledExpectation.md"><code>tfq.layers.SampledExpectation</code></a>, that learns to undo the random rotation
of the qubit:


```

>>> bit = cirq.GridQubit(0, 0)
>>> symbols = sympy.symbols('x, y, z')
>>> circuit = _gen_single_bit_rotation_problem(bit, symbols)
>>> control_input = tf.keras.Input(shape=(1,))
>>> circuit_inputs = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
>>> d1 = tf.keras.layers.Dense(10)(control_input)
>>> d2 = tf.keras.layers.Dense(3)(d1)
>>> expectation = tfq.layers.SampledExpectation()(
...     circuit_inputs, # See note below!
...     symbol_names=symbols,
...     symbol_values=d2,
...     operators=cirq.Z(bit),
...     repetitions=5000)
>>> data_in = np.array([[1], [0]], dtype=np.float32)
>>> data_out = np.array([[1], [-1]], dtype=np.float32)
>>> model = tf.keras.Model(
...     inputs=[circuit_inputs, control_input], outputs=expectation)
>>> model.compile(
...     optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
...     loss=tf.keras.losses.mean_squared_error)
>>> history = model.fit(
...     x=[tfq.convert_to_tensor([circuit] * 2), data_in],
...     y=data_out,
...     epochs=100)

```


For an example featuring this layer, please check out `Taking gradients`
in our dev website http://www.tensorflow.org/quantum/tutorials.

Lastly `symbol_values`, `operators` and circuit `inputs` can all be fed
Python `list` objects. In addition to this they can also be fed `tf.Tensor`
inputs, meaning that you can input all of these things from other Tensor
objects (like `tf.keras.Dense` layer outputs or `tf.keras.Input`s etc).

Note: When specifying a new layer for a *compiled* `tf.keras.Model` using
something like `tfq.layers.SampledExpectation()(cirq.Circuit(...), ...)`
please be sure to instead use
`tfq.layers.SampledExpectation()(circuit_input_tensor, ...)` where
`circuit_input_tensor` is filled with
`tfq.conver_to_tensor([cirq.Circuit(..)] * batch_size)` at runtime. This
is because compiled keras models require layer `call` inputs to be
traceable back to a `tf.keras.Input`.

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/quantum/tree/master/tensorflow_quantum/python/layers/circuit_executors/sampled_expectation.py">View source</a>

``` python
__init__(
    backend=None,
    differentiator=None,
    **kwargs
)
```

Instantiate this Layer.

Create a layer that will output expectation values gained from
simulating a quantum circuit.

#### Args:


* <b>`backend`</b>: Optional Backend to use to simulate states. Defaults to
    the native TensorFlow simulator (None), however users may also
    specify a preconfigured cirq simulation object to use instead,
    which must inherit `cirq.SimulatesFinalState`.
* <b>`differentiator`</b>: Optional Differentiator to use to calculate analytic
    derivative values of given operators_to_measure and circuit,
    which must inherit <a href="../../tfq/differentiators/Differentiator.md"><code>tfq.differentiators.Differentiator</code></a>.
    Defaults to None, which uses `parameter_shift.ParameterShift()`.



## Properties

<h3 id="activity_regularizer"><code>activity_regularizer</code></h3>

Optional regularizer function for the output of this layer.


<h3 id="dtype"><code>dtype</code></h3>




<h3 id="dynamic"><code>dynamic</code></h3>




<h3 id="input"><code>input</code></h3>

Retrieves the input tensor(s) of a layer.

Only applicable if the layer has exactly one input,
i.e. if it is connected to one incoming layer.

#### Returns:

Input tensor or list of input tensors.



#### Raises:


* <b>`RuntimeError`</b>: If called in Eager mode.
* <b>`AttributeError`</b>: If no inbound nodes are found.

<h3 id="input_mask"><code>input_mask</code></h3>

Retrieves the input mask tensor(s) of a layer.

Only applicable if the layer has exactly one inbound node,
i.e. if it is connected to one incoming layer.

#### Returns:

Input mask tensor (potentially None) or list of input
mask tensors.



#### Raises:


* <b>`AttributeError`</b>: if the layer is connected to
more than one incoming layers.

<h3 id="input_shape"><code>input_shape</code></h3>

Retrieves the input shape(s) of a layer.

Only applicable if the layer has exactly one input,
i.e. if it is connected to one incoming layer, or if all inputs
have the same shape.

#### Returns:

Input shape, as an integer shape tuple
(or list of shape tuples, one tuple per input tensor).



#### Raises:


* <b>`AttributeError`</b>: if the layer has no defined input_shape.
* <b>`RuntimeError`</b>: if called in Eager mode.

<h3 id="input_spec"><code>input_spec</code></h3>




<h3 id="losses"><code>losses</code></h3>

Losses which are associated with this `Layer`.

Variable regularization tensors are created when this property is accessed,
so it is eager safe: accessing `losses` under a `tf.GradientTape` will
propagate gradients back to the corresponding variables.

#### Returns:

A list of tensors.


<h3 id="metrics"><code>metrics</code></h3>




<h3 id="name"><code>name</code></h3>

Returns the name of this module as passed or determined in the ctor.

NOTE: This is not the same as the `self.name_scope.name` which includes
parent module names.

<h3 id="name_scope"><code>name_scope</code></h3>

Returns a `tf.name_scope` instance for this class.


<h3 id="non_trainable_variables"><code>non_trainable_variables</code></h3>




<h3 id="non_trainable_weights"><code>non_trainable_weights</code></h3>




<h3 id="output"><code>output</code></h3>

Retrieves the output tensor(s) of a layer.

Only applicable if the layer has exactly one output,
i.e. if it is connected to one incoming layer.

#### Returns:

Output tensor or list of output tensors.



#### Raises:


* <b>`AttributeError`</b>: if the layer is connected to more than one incoming
  layers.
* <b>`RuntimeError`</b>: if called in Eager mode.

<h3 id="output_mask"><code>output_mask</code></h3>

Retrieves the output mask tensor(s) of a layer.

Only applicable if the layer has exactly one inbound node,
i.e. if it is connected to one incoming layer.

#### Returns:

Output mask tensor (potentially None) or list of output
mask tensors.



#### Raises:


* <b>`AttributeError`</b>: if the layer is connected to
more than one incoming layers.

<h3 id="output_shape"><code>output_shape</code></h3>

Retrieves the output shape(s) of a layer.

Only applicable if the layer has one output,
or if all outputs have the same shape.

#### Returns:

Output shape, as an integer shape tuple
(or list of shape tuples, one tuple per output tensor).



#### Raises:


* <b>`AttributeError`</b>: if the layer has no defined output shape.
* <b>`RuntimeError`</b>: if called in Eager mode.

<h3 id="submodules"><code>submodules</code></h3>

Sequence of all sub-modules.

Submodules are modules which are properties of this module, or found as
properties of modules which are properties of this module (and so on).

```
a = tf.Module()
b = tf.Module()
c = tf.Module()
a.b = b
b.c = c
assert list(a.submodules) == [b, c]
assert list(b.submodules) == [c]
assert list(c.submodules) == []
```

#### Returns:

A sequence of all submodules.


<h3 id="trainable"><code>trainable</code></h3>




<h3 id="trainable_variables"><code>trainable_variables</code></h3>

Sequence of variables owned by this module and it's submodules.

Note: this method uses reflection to find variables on the current instance
and submodules. For performance reasons you may wish to cache the result
of calling this method if you don't expect the return value to change.

#### Returns:

A sequence of variables for the current module (sorted by attribute
name) followed by variables from all submodules recursively (breadth
first).


<h3 id="trainable_weights"><code>trainable_weights</code></h3>




<h3 id="updates"><code>updates</code></h3>




<h3 id="variables"><code>variables</code></h3>

Returns the list of all layer variables/weights.

Alias of `self.weights`.

#### Returns:

A list of variables.


<h3 id="weights"><code>weights</code></h3>

Returns the list of all layer variables/weights.


#### Returns:

A list of variables.




## Methods

<h3 id="__call__"><code>__call__</code></h3>

``` python
__call__(
    inputs,
    *args,
    **kwargs
)
```

Wraps `call`, applying pre- and post-processing steps.


#### Arguments:


* <b>`inputs`</b>: input tensor(s).
* <b>`*args`</b>: additional positional arguments to be passed to `self.call`.
* <b>`**kwargs`</b>: additional keyword arguments to be passed to `self.call`.


#### Returns:

Output tensor(s).



#### Note:

- The following optional keyword arguments are reserved for specific uses:
  * `training`: Boolean scalar tensor of Python boolean indicating
    whether the `call` is meant for training or inference.
  * `mask`: Boolean input mask.
- If the layer's `call` method takes a `mask` argument (as some Keras
  layers do), its default value will be set to the mask generated
  for `inputs` by the previous layer (if `input` did come from
  a layer that generated a corresponding mask, i.e. if it came from
  a Keras layer with masking support.



#### Raises:


* <b>`ValueError`</b>: if the layer's `call` method returns None (an invalid value).

<h3 id="build"><code>build</code></h3>

``` python
build(input_shape)
```

Creates the variables of the layer (optional, for subclass implementers).

This is a method that implementers of subclasses of `Layer` or `Model`
can override if they need a state-creation step in-between
layer instantiation and layer call.

This is typically used to create the weights of `Layer` subclasses.

#### Arguments:


* <b>`input_shape`</b>: Instance of `TensorShape`, or list of instances of
  `TensorShape` if the layer expects a list of inputs
  (one instance per input).

<h3 id="compute_mask"><code>compute_mask</code></h3>

``` python
compute_mask(
    inputs,
    mask=None
)
```

Computes an output mask tensor.


#### Arguments:


* <b>`inputs`</b>: Tensor or list of tensors.
* <b>`mask`</b>: Tensor or list of tensors.


#### Returns:

None or a tensor (or list of tensors,
    one per output tensor of the layer).


<h3 id="compute_output_shape"><code>compute_output_shape</code></h3>

``` python
compute_output_shape(input_shape)
```

Computes the output shape of the layer.

If the layer has not been built, this method will call `build` on the
layer. This assumes that the layer will later be used with inputs that
match the input shape provided here.

#### Arguments:


* <b>`input_shape`</b>: Shape tuple (tuple of integers)
    or list of shape tuples (one per output tensor of the layer).
    Shape tuples can include None for free dimensions,
    instead of an integer.


#### Returns:

An input shape tuple.


<h3 id="count_params"><code>count_params</code></h3>

``` python
count_params()
```

Count the total number of scalars composing the weights.


#### Returns:

An integer count.



#### Raises:


* <b>`ValueError`</b>: if the layer isn't yet built
  (in which case its weights aren't yet defined).

<h3 id="from_config"><code>from_config</code></h3>

``` python
@classmethod
from_config(
    cls,
    config
)
```

Creates a layer from its config.

This method is the reverse of `get_config`,
capable of instantiating the same layer from the config
dictionary. It does not handle layer connectivity
(handled by Network), nor weights (handled by `set_weights`).

#### Arguments:


* <b>`config`</b>: A Python dictionary, typically the
    output of get_config.


#### Returns:

A layer instance.


<h3 id="get_config"><code>get_config</code></h3>

``` python
get_config()
```

Returns the config of the layer.

A layer config is a Python dictionary (serializable)
containing the configuration of a layer.
The same layer can be reinstantiated later
(without its trained weights) from this configuration.

The config of a layer does not include connectivity
information, nor the layer class name. These are handled
by `Network` (one layer of abstraction above).

#### Returns:

Python dictionary.


<h3 id="get_input_at"><code>get_input_at</code></h3>

``` python
get_input_at(node_index)
```

Retrieves the input tensor(s) of a layer at a given node.


#### Arguments:


* <b>`node_index`</b>: Integer, index of the node
    from which to retrieve the attribute.
    E.g. `node_index=0` will correspond to the
    first time the layer was called.


#### Returns:

A tensor (or list of tensors if the layer has multiple inputs).



#### Raises:


* <b>`RuntimeError`</b>: If called in Eager mode.

<h3 id="get_input_mask_at"><code>get_input_mask_at</code></h3>

``` python
get_input_mask_at(node_index)
```

Retrieves the input mask tensor(s) of a layer at a given node.


#### Arguments:


* <b>`node_index`</b>: Integer, index of the node
    from which to retrieve the attribute.
    E.g. `node_index=0` will correspond to the
    first time the layer was called.


#### Returns:

A mask tensor
(or list of tensors if the layer has multiple inputs).


<h3 id="get_input_shape_at"><code>get_input_shape_at</code></h3>

``` python
get_input_shape_at(node_index)
```

Retrieves the input shape(s) of a layer at a given node.


#### Arguments:


* <b>`node_index`</b>: Integer, index of the node
    from which to retrieve the attribute.
    E.g. `node_index=0` will correspond to the
    first time the layer was called.


#### Returns:

A shape tuple
(or list of shape tuples if the layer has multiple inputs).



#### Raises:


* <b>`RuntimeError`</b>: If called in Eager mode.

<h3 id="get_losses_for"><code>get_losses_for</code></h3>

``` python
get_losses_for(inputs)
```

Retrieves losses relevant to a specific set of inputs.


#### Arguments:


* <b>`inputs`</b>: Input tensor or list/tuple of input tensors.


#### Returns:

List of loss tensors of the layer that depend on `inputs`.


<h3 id="get_output_at"><code>get_output_at</code></h3>

``` python
get_output_at(node_index)
```

Retrieves the output tensor(s) of a layer at a given node.


#### Arguments:


* <b>`node_index`</b>: Integer, index of the node
    from which to retrieve the attribute.
    E.g. `node_index=0` will correspond to the
    first time the layer was called.


#### Returns:

A tensor (or list of tensors if the layer has multiple outputs).



#### Raises:


* <b>`RuntimeError`</b>: If called in Eager mode.

<h3 id="get_output_mask_at"><code>get_output_mask_at</code></h3>

``` python
get_output_mask_at(node_index)
```

Retrieves the output mask tensor(s) of a layer at a given node.


#### Arguments:


* <b>`node_index`</b>: Integer, index of the node
    from which to retrieve the attribute.
    E.g. `node_index=0` will correspond to the
    first time the layer was called.


#### Returns:

A mask tensor
(or list of tensors if the layer has multiple outputs).


<h3 id="get_output_shape_at"><code>get_output_shape_at</code></h3>

``` python
get_output_shape_at(node_index)
```

Retrieves the output shape(s) of a layer at a given node.


#### Arguments:


* <b>`node_index`</b>: Integer, index of the node
    from which to retrieve the attribute.
    E.g. `node_index=0` will correspond to the
    first time the layer was called.


#### Returns:

A shape tuple
(or list of shape tuples if the layer has multiple outputs).



#### Raises:


* <b>`RuntimeError`</b>: If called in Eager mode.

<h3 id="get_updates_for"><code>get_updates_for</code></h3>

``` python
get_updates_for(inputs)
```

Retrieves updates relevant to a specific set of inputs.


#### Arguments:


* <b>`inputs`</b>: Input tensor or list/tuple of input tensors.


#### Returns:

List of update ops of the layer that depend on `inputs`.


<h3 id="get_weights"><code>get_weights</code></h3>

``` python
get_weights()
```

Returns the current weights of the layer.


#### Returns:

Weights values as a list of numpy arrays.


<h3 id="set_weights"><code>set_weights</code></h3>

``` python
set_weights(weights)
```

Sets the weights of the layer, from Numpy arrays.


#### Arguments:


* <b>`weights`</b>: a list of Numpy arrays. The number
    of arrays and their shape must match
    number of the dimensions of the weights
    of the layer (i.e. it should match the
    output of `get_weights`).


#### Raises:


* <b>`ValueError`</b>: If the provided weights list does not match the
    layer's specifications.

<h3 id="with_name_scope"><code>with_name_scope</code></h3>

``` python
@classmethod
with_name_scope(
    cls,
    method
)
```

Decorator to automatically enter the module name scope.

```
class MyModule(tf.Module):
  @tf.Module.with_name_scope
  def __call__(self, x):
    if not hasattr(self, 'w'):
      self.w = tf.Variable(tf.random.normal([x.shape[1], 64]))
    return tf.matmul(x, self.w)
```

Using the above module would produce `tf.Variable`s and `tf.Tensor`s whose
names included the module name:

```
mod = MyModule()
mod(tf.ones([8, 32]))
# ==> <tf.Tensor: ...>
mod.w
# ==> <tf.Variable ...'my_module/w:0'>
```

#### Args:


* <b>`method`</b>: The method to wrap.


#### Returns:

The original method wrapped such that it enters the module's name scope.




