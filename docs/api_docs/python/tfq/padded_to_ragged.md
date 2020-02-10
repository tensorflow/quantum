<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfq.padded_to_ragged" />
<meta itemprop="path" content="Stable" />
</div>

# tfq.padded_to_ragged

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/quantum/tree/master/tensorflow_quantum/core/ops/tfq_utility_ops.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Utility `tf.function` that converts a padded tensor to ragged.

``` python
tfq.padded_to_ragged(masked_state)
```



<!-- Placeholder for "Used in" -->

Convert a state `tf.Tensor` padded with the value -2 to a `tf.RaggedTensor`
using efficient boolean masking.

#### Args:


* <b>`masked_state`</b>: `tf.State` tensor with -2 padding.

#### Returns:


* <b>`state_ragged`</b>: State tensor without padding as a `tf.RaggedTensor`.