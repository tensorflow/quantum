# Install TensorFlow Quantum

There are a few ways to set up your environment to use TensorFlow Quantum (TFQ):

* The easiest way to learn and use TFQ requires no installation—run the
  [TensorFlow Quantum tutorials](./tutorials/hello_many_worlds.ipynb) directly
  in your browser using
  [Google Colab](https://colab.research.google.com/github/tensorflow/quantum/blob/master/docs/tutorials/hello_many_worlds.ipynb).
* To use TensorFlow Quantum on a local machine, install the TFQ package using
  Python's pip package manager.
* Or build TensorFlow Quantum from source.

TensorFlow Quantum is supported on Python 3.6 and 3.7.

## Pip package

### Requirements

* pip 19.0 or later (requires `manylinux2010` support)
* [TensorFlow == 2.1](https://www.tensorflow.org/install/pip)
* [Cirq 0.7](https://cirq.readthedocs.io/en/stable/install.html)

See the [TensorFlow install guide](https://www.tensorflow.org/install/pip) to
set up your Python development environment and an (optional) virtual environment.

Upgrade `pip` and install TensorFlow and Cirq (these are not included as
dependencies):

<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">pip3 install --upgrade pip</code>
  <code class="devsite-terminal">pip3 install tensorflow==2.1.0</code>
  <code class="devsite-terminal">pip3 install cirq==0.7.0</code>
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

Install the latest nightly version of TensorFlow Quantum:

<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">pip3 install -U tfq-nightly</code>
</pre>
<!-- common_typos_enable -->

## Build from source

The following steps are tested for Ubuntu-like systems.

### 1. Set up a Python 3 development environment

<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">sudo apt update</code>
  <code class="devsite-terminal">sudo apt-get install pkg-config zip g++ zlib1g-dev unzip python3</code>
  <code class="devsite-terminal">sudo apt install python3 python3-dev python3-venv python3-pip</code>
  <code class="devsite-terminal">python3 -m pip install --upgrade pip</code>
</pre>
<!-- common_typos_enable -->

### 2. Create a virtual environment

<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">python3 -m venv tfq_env</code>
  <code class="devsite-terminal">source tfq_env/bin/activate</code>
</pre>
<!-- common_typos_enable -->

### 3. Install Bazel

See the TensorFlow
[build from source](https://www.tensorflow.org/install/source#install_bazel)
guide to install the <a href="https://bazel.build/" class="external">Bazel</a>
build system.

To ensure compatibility with TensorFlow, `bazel` version 0.26.1 or lower is
required. To remove any existing version of Bazel:

<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">sudo apt-get remove bazel</code>
</pre>
<!-- common_typos_enable -->

Then install Bazel version 0.26.0:

<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">wget https://github.com/bazelbuild/bazel/releases/download/0.26.0/bazel_0.26.0-linux-x86_64.deb</code>
  <code class="devsite-terminal">sudo dpkg -i bazel_0.26.0-linux-x86_64.deb</code>
</pre>
<!-- common_typos_enable -->


### 4. Build TensorFlow from source

Read the TensorFlow [build from source](https://www.tensorflow.org/install/source)
guide for details. TensorFlow Quantum is compatible with TensorFlow version&nbsp;2.1.

Download the
<a href="https://github.com/tensorflow/tensorflow" class="external">TensorFlow source code</a>:

<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">git clone https://github.com/tensorflow/tensorflow.git</code>
  <code class="devsite-terminal">cd tensorflow</code>
  <code class="devsite-terminal">git checkout v2.1.0</code>
</pre>

Install the TensorFlow dependencies:

<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">python3 -m pip install -U pip six numpy wheel setuptools mock 'future>=0.17.1'</code>
  <code class="devsite-terminal">python3 -m pip install -U keras_applications --no-deps</code>
  <code class="devsite-terminal">python3 -m pip install -U keras_preprocessing --no-deps</code>
</pre>
<!-- common_typos_enable -->

Configure the TensorFlow build. The default Python location and Python library
paths should point inside the virtual environment. The default options are
recommended:

<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">./configure</code>
</pre>
<!-- common_typos_enable -->

Verify that your Bazel version is correct:

<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">bazel version</code>
</pre>
<!-- common_typos_enable -->

Build the TensorFlow package:

<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">bazel build -c opt --cxxopt="-O3" --cxxopt="-march=native" --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" //tensorflow/tools/pip_package:build_pip_package</code>
</pre>
<!-- common_typos_enable -->

Note: It may take over an hour to build the package.

After the build is complete, install the package:

<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg</code>
  <code class="devsite-terminal">pip install /tmp/tensorflow_pkg/<var>name_of_generated_wheel</var>.whl</code>
</pre>
<!-- common_typos_enable -->

### 5. Download TensorFlow Quantum

Download the TensorFlow Quantum source code and install the requirements:

<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">cd ..</code>
  <code class="devsite-terminal">git clone https://github.com/tensorflow/quantum.git</code>
  <code class="devsite-terminal">cd quantum</code>
  <code class="devsite-terminal">python3 -m pip install -r requirements.txt</code>
</pre>
<!-- common_typos_enable -->

Verify your Bazel version (since it can auto-update):

<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">bazel version</code>
</pre>
<!-- common_typos_enable -->

### 6. Build the TensorFlow Quantum pip package

Build the TensorFlow Quantum pip package and install:

<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">./configure.sh</code>
  <code class="devsite-terminal">bazel build -c opt --cxxopt="-O3" --cxxopt="-march=native" --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" release:build_pip_package</code>
  <code class="devsite-terminal">bazel-bin/release/build_pip_package /tmp/tfquantum/</code>
  <code class="devsite-terminal">python3 -m pip install /tmp/tfquantum/<var>name_of_generated_wheel</var>.whl</code>
</pre>
<!-- common_typos_enable -->

Success: TensorFlow Quantum is now installed.
