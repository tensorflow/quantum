# Making releases of TensorFlow Quantum

[TensorFlow Quantum](https://tensorflow.org/quantum) (TFQ) is implemented as a
Python library that integrates static C++ objects. Those C++ objects for TFQ are
linked with the TensorFlow [TensorFlow](https://tensorflow.org) library. The
process of building and linking C++ objects is complex, so TFQ is distributed in
binary Python [wheel](https://peps.python.org/pep-0427/) format for certain
combinations of operating systems and hardware architectures. This page
documents the process used by the TensorFlow Quantum maintainers to build and
release the packages for public TensorFlow Quantum distributions.

At present, binary distributions in the form of Python wheel files are made
available only for Linux systems.

## Version numbering policy

TensorFlow Quantum's version numbering follows the _major.minor.patch_ syntax
from [semver](https://semver.org/) (e.g., `0.7.6`). The version number for
TensorFlow Quantum is not tied to TensorFlow's version numbering.

## Building a binary distribution of TFQ

TensorFlow does not provide ABI stability guarantees between versions of
TensorFlow. Consequently, each release of TFQ must be pinned to a specific
version of TensorFlow, which also means that TFQ releases will not work with
any other version of TensorFlow than the one they are pinned to.

Python wheels for TFQ are produced by compiling them with a toolchain that
matches the toolchain used to build the TensorFlow version being targeted. A
number of factors affect whether the whole process succeeds and the resulting
wheel is portable to environments other than the specific computer TFQ is built
on, including:

*   The version of Python and the local Python environment.
*   The version of TensorFlow.
*   The TensorFlow build container used.
*   The Crosstool configuration used.
*   The dependency requirements implied by Cirq, TF-Keras, NumPy, Protobuf, and
    other Python packages that TFQ and TF depend on.

The following subsections describe the steps necessary to build binary
distributions in Python wheel format using scripts in TFQ's `scripts/`
and `release/` subdirectories.

### Preliminary steps

1.  Make sure you have `docker`, `pyenv`, `pip`, and `jq` installed on your
    system.

2.  Git clone the TensorFlow Quantum repo to a directory on your computer.

3.  `cd` into that directory in a Bash shell.

### Rebuild the `requirements.txt` file

1.  Create a Python virtual environment using the lowest version of Python
    supported by TFQ. (Currently this is Python 3.10.)

2.  Run these commands in your shell:

    ```shell
    pip install pip-tools
    ./scripts/generate_requirements.sh
    ```

    This will update the dependency versions in the file `requirements.txt` to
    the latest versions based on `requirements.in`. If this process fails, you
    may have to iterate on adjustments to the constraints in `requirements.in`
    followed by running `generate_requirements.sh` again until it succeeds.

3.  If any changes to `requirements.in` are needed, check `release/setup.py`
    and also make changes there if appropriate.

If all went well, proceed to the next subsection.

### Build the Python wheels

For each Python version _X.Y.Z_ that will be supported in the TFQ release, do
the following:

1.  Run `./release/build_release.sh X.Y.Z`.

2.  If the previous step fails, resolve the error.

    *   Important: if solving the problem requires changes to the versions of
        dependency packages, make sure to update both `requirements.in` and
        `release/setup.py` (if applicable), and then **go back to the previous
        subsection on rebuilding `requirements.txt`** and do it over.

3.  If `build_release.sh` succeeds without error, it should leave a new `.whl`
    file in the subdirectory `wheelhouse/`. Verify that it is there.

If all went well, proceed to test the wheels.

### Test the wheels

The tests below are ordered from simplest/quickest to more complex/slower,
so that basic problems (if they exist) can be discovered sooner.

#### Simple test in a local Python virtual environment

Iterate on the following steps for each Python version used for the builds
above:

1.  Create a fresh Python virtual environment for the version of Python
    corresponding to the wheel to be tested.

2.  Run `pip install wheelhouse/WHEELFILE`, where `WHEELFILE` is the file
    name of the wheel for the version of Python being tested.

3.  Run `./scripts/run_example.sh`. Ignore any warnings or errors related to
    CUDA or warnings related to optimizations, such as AVX/FMA instruction
    sets. (These originate from TensorFlow.) This test will not print any
    output if it succeeds, but if it fails, debug whatever error is shown in
    the Python traceback.

#### Docker tests

The scripts and instructions in `release/docker/README.md` describe how to
build Docker images for different combinations of Ubuntu and Python versions.
Build those images if you have not done so already. (They do not need to be
rebuilt if you made them for previous releases.)

1.  Start a Docker container that has the version of Python matching the
    version of Python needed by the TFQ wheel being tested. This example
    assumes Python 3.12:

    ```shell
    docker run -it --rm --network host -v .:/tfq ubuntu24-cp312
    ```

    The command above starts a Bash shell prompt inside a basic Ubuntu 24.04
    environment with Python 3.12 preinstalled and your local TensorFlow Quantum
    source directory accessible at `/tfq` from inside the container.

2.  Run `pip install /tfq/wheelhouse/WHEELFILE`, where `WHEELFILE` is the file
    name of the wheel being tested.

3.  Run `/tfq/scripts/run_example.sh`. Ignore any warnings or errors related to
    TensorFlow's use of CUDA or optimizations such as AVX/FMA instruction sets.
    This test will not print any output if it succeeds, but if it fails, debug
    whatever error is shown in the Python traceback.

#### Colab tests

This test can only be done for one of the wheels, namely the wheel built for
the version of Python running in Colab. Before proceeding, determine the
version of Python running in your Colab environment. (Currently for public
Colab it is Python 3.12.)

1.  Open a copy of a TFQ tutorial notebook in Colab. One way to do this is as
    follows:

    1.  Navigate to TFQ's `docs/tutorials` directory in your browser (i.e.,
        https://github.com/tensorflow/quantum/tree/master/docs/tutorials).

    2.  Click on a tutorial file, e.g., `hello_many_worlds.ipynb`.

    3.  Look for a link titled _Run in Google Colab_ near the top of the
        tutorial document, and click the link. This should open a browser tab
        or window in Google Colab with the tutorial notebook file loaded and
        ready to use.

2.  Using Colab's file explorer, upload a TFQ wheel that matches the version of
    Python running in Colab.

3.  When the upload finishes, right-click on the file name in the Colab
    file explorer and copy the path to the file.

4.  Find the notebook cell that contains the `!pip install` command for
    TensorFlow Quantum. **Replace that command** with the following,
    pasting in the path that you copied in the previous step:

    ```python
    !pip install /paste/the/path/to/the/wheel/file/here
    ```

5.  Run the notebook.

If the notebook executes all the way through without error, congratulations! If
it fails, analyze the cause, adjust TFQ or the TFQ build accordingly, then run
the tests again.

Repeat steps 1-5 for the other TFQ tutorials.

## Releasing a new version of TFQ

After all changes planned for the release have been merged into the working
version of TensorFlow Quantum on GitHub, all the builds succeed, and tutorials
run in all versions of Python and platforms supported, a release can be made.
Follow the steps in the order given below, even if you are accustomed to using
a different process for other projects, because certain components in TFQ
depend on other components being in place.

### 1. Release wheels on test.pypi.org

After successfully building and testing the wheels for a new version, the next
step is to create a test release.

1.  Upload the wheels to test.pypi.org using `twine`. (TFQ maintainers have
    access to the necessary account credentials.)

2.  Repeat a subset of the tests in the section _Test the wheels_, but instead
    of installing TFQ from a wheel file, tell pip to install from test.pypi.org:

    ```shell
    pip install --index-url https://test.pypi.org/simple/ \
        --extra-index-url https://pypi.org/simple/ \
        tensorflow-quantum==X.Y.Z
    ```

    where _X.Y.Z_ is the version number of the new TFQ release. Also test the
    tutorials in Colab, this time modifying the `!pip install` command to
    reference test.pypi.org to use the `--index-url` and `--extra-index-url`
    arguments shown above.

### 2. Release wheels on pypi.org

1.  If the test.pypi.org release and subsequent tests do not reveal problems,
    upload the wheels to PyPI. Note: **this step is irrevocable**, so be
    absolutely certain that everything is working and you are really ready to
    do the release. If a problem is found after this step and wheels need to
    be rebuilt, you will have to bump the TFQ version number and start over
    because PyPI by design does not allow releases to be modified.

2.  Once the wheels are on PyPI, repeat the Colab tests, this time modifying
    the `!pip install` command to be simply the following:

    ```shell
    !pip install tensorflow-quantum==X.Y.Z
    ```

### 3. Update tutorials and other documentation on GitHub

If all went well so far and the wheels are on PyPI:

1.  Update all the tutorials to refer to the new version of TFQ, update the
    top-level `README.md` file, and update any other documentation files that
    mention the TFQ version number, the Python version, and any Python package
    versions (such as the Cirq version supported).

2.  Open a PR for these changes and ask for a review.

3.  Merge the PR after it has been reviewed and accepted.

Note that this step is purposefully done _after_ the release is uploaded to
PyPI. When a PR is merged on GitHub, Google's documentation pipeline for
https://tensorflow.org/quantum runs automatically to update the tutorial pages,
and since the tutorials contain a `pip install` command with the current TFQ
version number, the version on PyPI needs to exist so that the documentation
builds successfully. (Documentation and tutorial files are not part of the
binary wheel distribution; updates to documentation will not change the
wheels.)

### 4. Verify the documentation has been updated on tensorflow.org

Verify that the TensorFlow Quantum API and tutorial pages on tensorflow.org
have been updated by the TensorFlow documentation pipeline. Particularly check
that all the tutorials ran without errors. If any errors occurred, debug and
resolve them. It is possible that this may require changes to TFQ (such as
changes to versions of TFQ dependencies); in extreme cases, the changes may
require bumping the TFQ version number and redoing the build and release steps
up to this point, before the release is finalized on GitHub.

### 5. Release the new version on GitHub

Once you have verified that the documentation has been updated on
https://tensorflow.org/quantum:

1.  Write release notes and create a draft release on GitHub.

2.  Attach to this draft release the wheel files that were uploaded to PyPI.

3.  Have someone review the release notes and draft release.

4.  Publish the release on GitHub.

The GitHub release system will automatically tag the files in the repository
with the release tag you specify (e.g., `v0.7.6`), as well as create an archive
of all the source files and attach it to the release.

### 6. Bump the TFQ version number in GitHub

1.  Update the version number in `tensorflow_quantum/__init__.py` and
    `release/setup.py` in the repository on GitHub

2.  Open a PR for these changes and ask for a review.

3.  Merge the PR after it has been reviewed and accepted.

### 7. Announce the release

Announce the release on the TFQ developer's mailing list, as well as via a
banner on GitHub, the next Cirq Cynq, and any other venues where TFQ releases
are normally announced.
