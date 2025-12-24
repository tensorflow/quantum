# Tools for building releases of TensorFlow Quantum

This directory contains configurations and scripts that the TensorFlow Quantum
maintainers use to create Python packages for software releases. The process of
making a TFQ release is complex and has not been fully automated. The scripts
in this directory help automate some steps and are a way of capturing the
process more precisely, but there are still manual steps involved.

## Background: how TensorFlow Quantum is linked with TensorFlow

TFQ is implemented as a Python library that integrates static C++ objects. Those
C++ objects are linked with TensorFlow static objects when both TFQ and
TensorFlow are installed on your system. Unlike a pure Python library, the
result is platform-dependent: the Python code itself remains portable, but the
underlying C++ objects need to be compiled specifically for each target
environment (operating system and CPU architecture).

TensorFlow does not provide ABI stability guarantees between versions of
TensorFlow. In order to avoid the need for users to compile the TFQ source code
themselves when they want to install TFQ, each release of TFQ must be pinned to
a specific version of TensorFlow. As a consequence, TFQ releases will not work
with any other version of TensorFlow than the one they are pinned to.

Python wheels for TFQ are produced by compiling them locally with a toolchain
that matches that used by the version of TensorFlow being targeted by a given
version of TFQ. A number of elements affect whether the whole process succeeds
and the resulting wheel is portable to environments other than the specific
computer TFQ is built on, including:

*   The version of Python and the local Python environment
*   The version of TensorFlow
*   The TensorFlow build container used
*   The Crosstool configuration used
*   Whether CUDA is being used, and its version
*   The dependency requirements implied by Cirq, TF-Keras, NumPy, Protobuf, and
    other Python packages

## Procedure

Building a TensorFlow Quantum release for Linux involves some additional steps
beyond just building TFQ and producing an initial Python wheel. The procedure
uses `auditwheel` to "repair" the resulting wheel; this improves the
compatibility of the wheel so that it can run on a wider range of Linux
distributions, even if those distributions have different versions of system
libraries.

### Preliminary steps

1.  Make sure you have `pyenv`, `pip`, and `jq` installed on your system.

2.  (Optional) Preinstall Python versions 3.9, 3.10, 3.11, and 3.12 into `pyenv`
    so that `build_release.sh` can create virtual environments with those Python
    versions without having to install the requested version(s) itself.

3.  Git clone the TensorFlow Quantum repo to a directory on your computer.

4.  `cd` into this local clone directory.

### Build the release

1.  Run `./release/build_release.sh X.Y.Z`, where _X.Y.Z_ is a Python version
    for which you want to build a TFQ release.

2.  If the previous step completes successfully, proceed to the next section
    below and test the wheel.

3.  Repeat steps 1&ndash;2 for other Python versions.

### Testing the release

Testing is currently not automated to the degree that building a release is.
Assuming that one of the procedures above was used to create one or more wheels
for a TFQ release, here are the steps for testing each one.

1.  First, perform a quick local test.

    1.  `cd` out of the TFQ source directory. This is a critical step, because
        importing TFQ into a Python interpreter when the current directory is
        the TFQ source tree will result in baffling errors (usually something
        about `pauli_sum_pb2` not found).

    1.  Create a fresh Python virtual environment.

    1.  Run `pip install /path/to/whl/file`, where `/path/to/whl/file` is the
        path to the wheel file corresponding to the version of Python you are
        running.

    1.  Run the following snippet. If this results in an error, stop and debug
        the problem.

        ```python
        import tensorflow_quantum as tfq
        `print(tfq.__version__)
        ```

    1.  If the previous snippet ran without error, next try running some
        more elaborate TFQ example code.

2.  Second, test in Colab.

    1.  Go to a remotely hosted Colab and make a copy of the Hello Many Worlds
        [tutorial notebook](
        https://www.tensorflow.org/quantum/tutorials/hello_many_worlds).

    1.  Using the Colab file explorer, upload a TFQ wheel you created matching
        the version of Python running in Colab. (At the time of this writing,
        this is Python 3.12.)

    1.  When the upload finishes, right-click on the file name in the Colab file
        explorer and copy the path to the file in Colab.

    1.  Find the notebook cell that contains the `!pip install` command for
        TensorFlow Quantum. **Replace that command** with the following, pasting
        in the path that you copied in the previous step:

        ```python
        !pip install /here/paste/the/path/to/the/wheel/file
        ```

    1.  Run the notebook step by step. If Colab asks you to restart the session,
        do so, and after it finishes restarting, continue with the remaining
        cells in the notebook.

    1.  If the notebook executes all the way through without error,
        congratulations! If something fails, proceed to debug the problem.

## Alternative procedure

As mentioned above, `build_release.sh` relies on other scripts to do the main
work. Those steps can be run manually, and sometimes that's useful to do that
when debugging problems. The steps in this more manual approach are:

1.  Create a Python virtual environment. (The maintainers currently use `pyenv`
    but Python's built-in `venv` should work too.)

2.  Run `pip install -r requirements.txt`

3.  Run `./release/build_distribution.sh`

4.  If the above succeeds, it will leave the wheel in `/tmp/tensorflow_quantum/`
    on your system. Take note of the name of the wheel file that
    `build_distribution.sh` prints when it finishes.

5.  Run `./release/clean_distribution.sh /tmp/tensorflow_quantum/WHEEL_FILE`,
    where `WHEEL_FILE` is the file noted in the previous step. If this works, it
    will create a new wheel file in `../wheelhouse`. If an error occurs, it will
    hopefully report the problem. If the error is a platform tag mismatch, run
    `./release/clean_distribution.sh -s /tmp/tensorflow_quantum/WHEEL_FILE`;
    this will run auditwheel's `show` command on the wheel file to indicate what
    version of `manylinux` this wheel can be made to run on if you use
    `auditwheel` to repair it. With that information, you may be able to edit
    the `build_distribution.sh` script to experiment with different values for
    the Crosstool and/or the Docker images used.

6.  If the previous step succeeded, go to the next section (Testing the
    release files) and do preliminary testing on the wheel.

7.  If the tests succeed, repeat the `build_distribution.sh` and
    `clean_distribution.sh` steps for different versions of Python. If the
    preliminary tests fail, proceed to debugging the reason.

## More information

"TensorFlow SIG Build" is a community group dedicated to the TensorFlow build
process. This repository is a showcase of resources, guides, tools, and builds
contributed by the community, for the community. The following resources may be
useful when trying to figure out how to make this all work.

*   The "TF SIG Build Dockerfiles" document:
    https://github.com/tensorflow/build/tree/ff4320fee2cf48568ebd2f476d7714438bfa0bee/tf_sig_build_dockerfiles#readme

*   Other info in the SIG Build repo: https://github.com/tensorflow/build
