<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfq" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__version__"/>
</div>

# Module: tfq

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/quantum/tree/master/tensorflow_quantum/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Module functions for tensorflow_quantum.*



## Modules

[`datasets`](./tfq/datasets.md) module: Experimental location for interesting quantum datasets.

[`differentiators`](./tfq/differentiators.md) module: Module functions for tfq.differentiators.*

[`layers`](./tfq/layers.md) module: Module definitions for tensorflow_quantum.python.layers.*

[`util`](./tfq/util.md) module: A collection of helper functions that are useful several places in tfq.

## Functions

[`convert_to_tensor(...)`](./tfq/convert_to_tensor.md): Convert lists of tfq supported primitives to tensor representations.

[`from_tensor(...)`](./tfq/from_tensor.md): Convert a tensor of tfq primitives back to Python objects.

[`get_expectation_op(...)`](./tfq/get_expectation_op.md): Get a TensorFlow op that will calculate batches of expectation values.

[`get_sampled_expectation_op(...)`](./tfq/get_sampled_expectation_op.md): Get a TensorFlow op that will calculate sampled expectation values.

[`get_sampling_op(...)`](./tfq/get_sampling_op.md): Get a Tensorflow op that produces samples from given quantum circuits.

[`get_state_op(...)`](./tfq/get_state_op.md): Get a TensorFlow op that produces states from given quantum circuits.

[`padded_to_ragged(...)`](./tfq/padded_to_ragged.md): Utility `tf.function` that converts a padded tensor to ragged.

## Other Members

* `__version__ = '0.2.0'` <a id="__version__"></a>
