# Tools for building releases of TensorFlow Quantum

This directory contains configurations and scripts that the TensorFlow Quantum
maintainers use to create Python packages for software releases. The process of
making a TFQ release is unfortunately complex, and has not been fully automated.
The scripts in this directory help automate some steps and are a way of
capturing the process more precisely, but there are still manual steps involved.

## Background: how TensorFlow Quantum is linked with TensorFlow

TFQ is a TensorFlow addon: it is an optional component that extends TensorFlow
with additional functionality. TFQ is implemented as a Python library that uses
static C++ objects linked with TensorFlow static objects when both TFQ and
TensorFlow are installed on your system.

TensorFlow does not provide ABI stability guarantees between versions of
TensorFlow. In order to avoid the need for users to compile the TFQ source code
themselves when they want to install TFQ, each release of TFQ must be pinned to
a specific version of TensorFlow. As a consequence, TFQ releases will not work
with any other version of TensorFlow than the one they are pinned to.

Wheels for TFQ are produced by compiling them locally with a toolchain that
matches that used by the version of TensorFlow being targetted by a given
version of TFQ. A number of elements affect whether the whole process succeeds
and the resulting wheel is portable to environments other than the specific
computer TFQ is built on, including:

*   The version of Python and the local Python environment
*   The version of TensorFlow
*   The TensorFlow build container used
*   The Crosstool used
*   Whether CUDA is being used, and its version
*   The combined dependency requirements between Cirq, NumPy, Protobuf, and
    other Python packages

## Procedure

The procedure below builds a wheel file and uses `auditwheel` to "repair" the
resulting wheel. Its primary function is to ensure that a Python wheel
containing compiled extensions (like the C/C++ code in TFQ) can run on a wide
range of Linux distributions, even if those distributions have different
versions of system libraries.

1.  Git clone the TensorFlow Quantum repo to a directory on your computer.

1.  `cd` into the local clone directory.

1.  Create a Python virtual environment.

1.  Run `pip install -r requirements.txt`

1.  Verify the major.minor version of Python you are using. The rest of these
    instructions use 3.11 as an example.

1.  Run `./release/build_wheel.sh -p 3.11`

1.  If the above succeeds, it will leave the wheel in `/tmp/tensorflow_quantum/`
    on your system. Take note of the name of the wheel file that
    `build_wheel.sh` prints when it finishes.

1.  Run `./release/run_auditwheel.sh /tmp/tensorflow_quantum/WHEEL_FILE`, where
    `WHEEL_FILE` is the file noted in the previous step. If this works, it will
    create a new wheel file in `../wheelhouse`. If an error occurs, it will
    hopefully report the problem. If the error is a platform tag mismatch, run
    `./release/run_auditwheel.sh -s /tmp/tensorflow_quantum/WHEEL_FILE`; this
    will run auditwheel's `show` command on the wheel file to indicate what
    version of `manylinux` this wheel can be made to run on if you use
    `auditwheel` to repair it. With that information, you may be able to edit
    the `build_wheel.sh` script to experiment with different values for the
    Crosstool and/or the Docker images used.

1.  Once the previous step is successful, the next step is to test the wheel.
    Go to a remotely hosted Colab (or any other linux platform that is
    distinctly difference from yours) upload this new generated wheel file to
    the colab, and test if it works. (E.g., you can try to run the tutorial at
    `https://www.tensorflow.org/quantum/tutorials/hello_many_worlds` with the new
    TensorFlow and TFQ wheels to verify things are working smoothly).

1.  Repeat the `build_wheel.sh` and `run_auditwheel.sh` steps for different
    versions of Python.

An essential part of creating TFQ wheels that will work on other systems is that
the platform tags of the TFQ wheel must match the tags of the current TensorFlow
version you are targeting. (Visit `https://pypi.org/project/tensorflow/<target
TF version>/#files` to determine what the tags are).

## More information

Check the SIG readme for information:

* https://github.com/tensorflow/build/tree/ff4320fee2cf48568ebd2f476d7714438bfa0bee/tf_sig_build_dockerfiles#readme
