<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfq.util.kwargs_cartesian_product" />
<meta itemprop="path" content="Stable" />
</div>

# tfq.util.kwargs_cartesian_product

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/quantum/tree/master/tensorflow_quantum/python/util.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Compute the cartesian product of inputs yielding Python `dict`s.

```python
tfq.util.kwargs_cartesian_product(
    **kwargs
)
```



<!-- Placeholder for "Used in" -->

Note that all kwargs must provide `iterable` values. Useful for testing
purposes.

```python
a = {'one': [1,2,3], 'two': [4,5]}
result = list(kwargs_cartesian_product(**a))

# Result now contains:
# [{'one': 1, 'two': 4},
#  {'one': 1, 'two': 5},
#  {'one': 2, 'two': 4},
#  {'one': 2, 'two': 5},
#  {'one': 3, 'two': 4},
#  {'one': 3, 'two': 5}]
```

#### Returns:

Python `generator` of the cartesian product of the inputs `kwargs`.
