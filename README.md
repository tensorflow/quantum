<!-- H1 title omitted because our logo acts as the title. -->
<div align="center">

<img width="450px" alt="TensorFlow Quantum logo"
src="docs/images/logo/tf_quantum1.svg">

High-performance Python framework for hybrid quantum-classical machine learning

[![Licensed under the Apache 2.0
license](https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square)](https://github.com/tensorflow/quantum/blob/master/LICENSE)
[![Compatible with Python versions 3.10 and
higher](https://img.shields.io/badge/Python-3.10+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![TensorFlow Quantum project on
PyPI](https://img.shields.io/pypi/v/TensorFlow_Quantum.svg?logo=python&logoColor=white&label=PyPI&style=flat-square&color=fcbc2c)](https://pypi.org/project/tensorflow-quantum)

[Features](#features) &ndash;
[Installation](#installation) &ndash;
[Quick Start](#quick-start) &ndash;
[Documentation](#documentation) &ndash;
[Getting help](#getting-help) &ndash;
[Citing TFQ](#citing-tensorflow-quantum) &ndash;
[Contact](#contact)

</div>

## Features

[TensorFlow Quantum](https://www.tensorflow.org/quantum) (TFQ) is a Python
framework for hybrid quantum-classical machine learning focused on modeling
quantum data. It enables quantum algorithms researchers and machine learning
applications researchers to explore computing workflows that leverage Google’s
quantum computing offerings – all from within the powerful
[TensorFlow](https://tensorflow.org) ecosystem.

*   Integrates with [Cirq](https://github.com/quantumlib/Cirq) for writing
    quantum circuit definitions
*   Integrates with [qsim](https://github.com/quantumlib/qsim) for running
    quantum circuit simulations
*   Uses [Keras](https://keras.io) to provide high-level abstractions for
    quantum machine learning constructs
*   Provides an extensible system for automatic differentiation of quantum
    circuits
*   Offers many methods for computing gradients, including parameter shift and
    adjoint methods
*   Implements operations as C++ TensorFlow Ops, making them 1<sup>st</sup>-class
    citizens in the TF compute graph
*   Harnesses TensorFlow’s computational machinery to provide exceptional
    performance and scalability

## Motivation

TensorFlow Quantum provides users with the tools they need to interleave quantum
algorithms and logic designed in Cirq with the powerful and performant ML tools
from TensorFlow. With this connection, we hope to unlock new and exciting paths
for quantum computing research that would not have otherwise been possible.

Thanks to its power and scalability, TensorFlow Quantum has already been
instrumental in enabling ground-breaking research in QML. It empowers
researchers to pursue questions whose answers can only be obtained through fast
simulation of many millions of moderately-sized circuits.

## Installation

Please see the [installation
instructions](https://www.tensorflow.org/quantum/install) in the documentation.

## Quick start

[Guides and tutorials for TensorFlow
Quantum](https://tensorflow.org/quantum/overview) are available online at the
TensorFlow.org web site.

## Documentation

[Documentation for TensorFlow Quantum](https://tensorflow.org/quantum),
including tutorials and API documentation, can be found online at the
TensorFlow.org web site.

All of the examples can be found in GitHub in the form of [Python notebook
tutorials](https://github.com/tensorflow/quantum/tree/master/docs/tutorials)

## Getting help

Please report bugs or feature requests using the [TensorFlow Quantum issue
tracker](https://github.com/tensorflow/quantum/issues) on GitHub.

There is also a [Stack Overflow tag for TensorFlow
Quantum](https://stackoverflow.com/questions/tagged/tensorflow-quantum) that you
can use for more general TFQ-related discussions.

## Citing TensorFlow Quantum<a name="how-to-cite-tfq"></a><a name="how-to-cite"></a>

When publishing articles or otherwise writing about TensorFlow Quantum, please
cite the paper ["TensorFlow Quantum: A Software Framework for Quantum Machine
Learning" (2020)](https://arxiv.org/abs/2003.02989) and include information
about the version of TFQ you are using.

```bibtex
@misc{broughton2021tensorflowquantum,
      title={TensorFlow Quantum: A Software Framework for Quantum Machine Learning},
      author={Michael Broughton and Guillaume Verdon and Trevor McCourt
      and Antonio J. Martinez and Jae Hyeon Yoo and Sergei V. Isakov
      and Philip Massey and Ramin Halavati and Murphy Yuezhen Niu
      and Alexander Zlokapa and Evan Peters and Owen Lockwood and Andrea Skolik
      and Sofiene Jerbi and Vedran Dunjko and Martin Leib and Michael Streif
      and David Von Dollen and Hongxiang Chen and Shuxiang Cao and Roeland Wiersema
      and Hsin-Yuan Huang and Jarrod R. McClean and Ryan Babbush and Sergio Boixo
      and Dave Bacon and Alan K. Ho and Hartmut Neven and Masoud Mohseni},
      year={2021},
      eprint={2003.02989},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      doi={10.48550/arXiv.2003.02989},
      url={https://arxiv.org/abs/2003.02989},
}
```

## Contact

For any questions or concerns not addressed here, please email
quantum-oss-maintainers@google.com.

## Disclaimer

This is not an officially supported Google product. This project is not eligible
for the [Google Open Source Software Vulnerability Rewards
Program](https://bughunters.google.com/open-source-security).

Copyright 2020 Google LLC.

<div align="center">
  <a href="https://quantumai.google">
    <img width="15%" alt="Google Quantum AI"
         src="docs/images/quantum-ai-vertical.svg">
  </a>
</div>
