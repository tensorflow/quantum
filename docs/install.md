# Install TensorFlow Quantum

There are a few ways to set up your environment to use TensorFlow Quantum (TFQ):

* The easiest way to learn and use TFQ requires no installation—run the
  [TensorFlow Quantum tutorials](./tutorials/hello_many_worlds.ipynb) directly
  in your browser using
  [Google Colab](https://colab.research.google.com/github/tensorflow/quantum/blob/master/docs/tutorials/hello_many_worlds.ipynb).
* To use TensorFlow Quantum on a local machine, install the TFQ package using
  Python's pip package manager.
* Or build TensorFlow Quantum from source.

TensorFlow Quantum is supported on Python 3.10, 3.11, and 3.12 and depends directly on [Cirq](https://github.com/quantumlib/Cirq).

## Pip package

### Requirements

* pip 19.0 or later (requires `manylinux2014` support)
* [TensorFlow == 2.16.2](https://www.tensorflow.org/install/pip)

See the [TensorFlow install guide](https://www.tensorflow.org/install/pip) to
set up your Python development environment and an (optional) virtual environment.

Upgrade `pip` and install TensorFlow
<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">pip3 install --upgrade pip</code>
  <code class="devsite-terminal">pip3 install tensorflow==2.16.2</code>
</pre>
<!-- common_typos_enable -->

### Install the package

Install the latest stable release of TensorFlow Quantum:

<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">pip3 install -U tensorflow-quantum</code>
</pre>
<!-- common_typos_enable -->

Success: TensorFlow Quantum is now installed.

<!-- Nightly builds which might depend on newer version of TensorFlow can be installed with: -->

<!-- <\!-- common_typos_disable -\-> -->
<!-- <pre class="devsite-click-to-copy"> -->
<!--   <code class="devsite-terminal">pip3 install -U tfq-nightly</code> -->
<!-- </pre> -->
<!-- <\!-- common_typos_enable -\-> -->

## Build from source

The following steps are tested for Ubuntu-like systems.

### 1. Set up a Python 3 development environment

First we need the Python 3.10 development tools.
<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">sudo apt update</code>
  <code class="devsite-terminal">sudo apt-get install pkg-config zip g++ zlib1g-dev unzip python3.10</code>
  <code class="devsite-terminal">sudo apt install python3.10 python3.10-dev python3.10-venv python3-pip</code>
  <code class="devsite-terminal">python3.10 -m pip install --upgrade pip</code>
</pre>
<!-- common_typos_enable -->

### 2. Create a virtual environment

Go to your workspace directory and make a virtual environment for TFQ development.
<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">python3.10 -m venv quantum_env</code>
  <code class="devsite-terminal">source quantum_env/bin/activate</code>
</pre>
<!-- common_typos_enable -->

Make sure that the virtual environment is activated for the rest of the steps
below, and every time you want to use TFQ in the future.

### 3. Install Bazel

As noted in the TensorFlow
[build from source](https://www.tensorflow.org/install/source#install_bazel)
guide, the <a href="https://bazel.build/" class="external">Bazel</a>
build system will be required.

Our latest source builds use TensorFlow 2.16.2. To ensure compatibility we use
`bazel` version 6.5.0. To remove any existing version of Bazel:
<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">sudo apt-get remove bazel</code>
</pre>
<!-- common_typos_enable -->

Download and install `bazel` version 6.5.0:

<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">wget https://github.com/bazelbuild/bazel/releases/download/6.5.0/bazel_6.5.0-linux-x86_64.deb
</code>
  <code class="devsite-terminal">sudo dpkg -i bazel_6.5.0-linux-x86_64.deb</code>
</pre>
<!-- common_typos_enable -->

To prevent automatic updating of `bazel` to an incompatible version, run the following:

<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">sudo apt-mark hold bazel</code>
</pre>
<!-- common_typos_enable -->

Finally, confirm installation of the correct `bazel` version:

<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">bazel --version</code>
</pre>
<!-- common_typos_enable -->


### 4. Build TensorFlow from source

Here we adapt instructions from the TensorFlow [build from source](https://www.tensorflow.org/install/source)
guide, see the link for further details. TensorFlow Quantum is compatible with TensorFlow version&nbsp;2.16.2.

Download the
<a href="https://github.com/tensorflow/tensorflow" class="external">TensorFlow source code</a>:

<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">git clone https://github.com/tensorflow/tensorflow.git</code>
  <code class="devsite-terminal">cd tensorflow</code>
  <code class="devsite-terminal">git checkout v2.16.2</code>
</pre>

Be sure the virtual environment you created in step 2 is activated. Then,
install the TensorFlow dependencies using the command below, substituting your
actual version of Python in place of 3_10 (e.g., if your Python is 3.11, use
file `requirements_lock_3_11.txt`):

<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">pip install -r requirements_lock_3_10.txt</code>
</pre>
<!-- common_typos_enable -->

Configure the TensorFlow build. When asked for the Python interpreter and library locations, be sure to specify locations inside your virtual environment folder.  The remaining options can be left at default values.

<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">./configure</code>
</pre>
<!-- common_typos_enable -->

Build the TensorFlow package:

<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">bazel build -c opt --cxxopt="-O3" --cxxopt="-march=native" --cxxopt="-std=c++17" --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=1" //tensorflow/tools/pip_package:build_pip_package</code>
</pre>
<!-- common_typos_enable -->

Note: It may take over an hour to build the package.

After the build is complete, install the package and leave the TensorFlow directory:

<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg</code>
  <code class="devsite-terminal">pip install /tmp/tensorflow_pkg/<var>name_of_generated_wheel</var>.whl</code>
  <code class="devsite-terminal">cd ..</code>
</pre>
<!-- common_typos_enable -->

### 5. Download TensorFlow Quantum

We use the standard [fork and pull request workflow](https://guides.github.com/activities/forking/) for contributions.  After forking from the [TensorFlow Quantum](https://github.com/tensorflow/quantum) GitHub page, download the source of your fork and install the requirements:

<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">git clone https://github.com/<var>username</var>/quantum.git</code>
  <code class="devsite-terminal">cd quantum</code>
  <code class="devsite-terminal">pip install -r requirements.txt</code>
</pre>
<!-- common_typos_enable -->

### 6. Build and install TensorFlow Quantum

Be sure the virtual environment you created in step 2 is activated. Then, run
the command below to install the TensorFlow Quantum dependencies:

<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">pip install -r requirements.txt</code>
</pre>
<!-- common_typos_enable -->

Next, use TensorFlow Quantum's `configure.sh` script to configure the TFQ
build:

<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">./configure.sh</code>
</pre>
<!-- common_typos_enable -->

Now build TensorFlow Quantum:

<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">bazel build -c opt --cxxopt="-O3" --cxxopt="-march=native" release:build_pip_package</code>
</pre>
<!-- common_typos_enable -->

After the build is complete, run the next command to create a Python package
for TensorFlow Quantum and write it to a temporary directory (we use
`/tmp/tfquantum/` in this example):

<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">bazel-bin/release/build_pip_package /tmp/tfquantum/</code>
  <code class="devsite-terminal">pip install /tmp/tfquantum/<var>name_of_generated_wheel</var>.whl</code>
</pre>
<!-- common_typos_enable -->

To confirm that TensorFlow Quantum has successfully been installed, you can run the tests:
<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">./scripts/test_all.sh</code>
</pre>
<!-- common_typos_enable -->


Success: TensorFlow Quantum is now installed.
