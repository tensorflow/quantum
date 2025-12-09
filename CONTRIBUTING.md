# Guidelines for contributing to this project

Welcome, and thank you for your interest in contributing to our project! We're
excited to work with you. To ensure a smooth and effective collaboration, please
follow these guidelines.

Table of contents quick links:

*   [Getting started](#getting-started)
*   [Development process](#development-process)
*   [Coding conventions](#coding-conventions)
*   [Contribution workflow](#contribution-workflow)

## Getting started

Before you write any code, please complete the following steps. This will set
you up for success and ensure your contributions can be merged smoothly.

*   Review our [community guidelines](#learn-our-community-guidelines) to
    understand our commitment to a welcoming and harassment-free environment.

*   Sign the [Contributor License Agreement (CLA)](#cla) to grant us permission
    to use your work.

*   Set up a [GitHub account](https://github.com) if you do not already have
    one. All contributions in this project (bug fixes, new development,
    documentation, etc.) follow a GitHub-centered workflow.

*   Learn how to find [opportunities for contributions](#opportunities).

*   Understand our [development process](#development-process) and [contribution
    workflow](#contribution-workflow).

### Learn our community guidelines

We strive to create an open and welcoming community where everyone is treated
with respect. All contributors and maintainers pledge to make participation a
harassment-free experience for everyone, regardless of background or experience.

Our key principles include:

*   Using welcoming and inclusive language.
*   Being respectful of different viewpoints and experiences.
*   Gracefully accepting constructive criticism.
*   Focusing on what is best for the project and the community.

Please read our full [Code of Conduct](CODE_OF_CONDUCT.md) for more details.

### <a name="cla"></a>Sign the Contributor License Agreement

All contributions to this project must be accompanied by a signed [Contributor
License Agreement (CLA)](https://cla.developers.google.com/).

*   _What it does_: The CLA gives us permission to use and redistribute your
    contributions as part of the project. You (or your employer) retain full
    copyright to your work.

*   _How to sign_: Visit
    [cla.developers.google.com](https://cla.developers.google.com/) to sign the
    agreement or check your status. You typically only need to sign a Google CLA
    once, and it covers contributions to all of Google's open-source projects.

*   _Originality_: By signing the CLA, you confirm that your contributions are
    original works of authorship. Although coding assistance tools are
    permitted, the final contribution must be your own creation.

Only original works from you and other people who signed the CLA can be merged
into this project.

### <a name="opportunities"></a>Finding opportunities to make contributions

The best way to get started is by tackling an existing issue.

1.  Navigate to the [Issues tab](https://github.com/tensorflow/quantum/issues)
    on GitHub.

2.  Filter the issues list for those labeled ["good first
    issue"](https://github.com/tensorflow/quantum/labels/good%20first%20issue).
    These are specifically chosen as good entry points for new contributors.

3.  Once you pick an issue, leave a comment to let others know you are working
    on it. This helps prevent duplicated effort.

## Development process

This section describes the development process:

*   Preliminary software requirements
*   How to set up a local development environment
*   How to build TensorFlow Quantum locally
*   How to run tests locally
*   How to add new modules to TensorFlow Quantum

### Preliminary software requirements

Install the tools described in the [install TensorFlow Quantum](
docs/install.md) guide's section on how to build from source. The required tools
include Python version 3.10 or greater, as well as [git](https://git-scm.com/),
[pip](https://pip.pypa.io/en/stable/), a working Clang or GCC compiler toolchain
that supports C++17, and [Bazel](https://bazel.build/) version 6.5.0.

### <a name="dev-env"></a>Setting up a local development environment

1.  Fork the TFQ repository to your account on GitHub. (Go to the TFQ repo page
    on GitHub and use the Fork button.) For more information on forks, consult
    [GitHub's guide](https://help.github.com/articles/fork-a-repo/).

2.  Clone your fork to somewhere on your computer. In the following command,
    substitute your GitHub account name in place of _YOURACCOUNT_:

    ```shell
    git clone git@github.com:YOURACCOUNT/quantum.git
    cd quantum
    ```

3.  Set up a Python virtual environment. Here is an example of doing that with
    Python's built-in [venv](https://docs.python.org/3/library/venv.html), but
    you can use other methods and tools if you are more comfortable with them:

    ```shell
    python3 -m venv quantum_env
    source quantum_env/bin/activate
    ```

4.  Install the TFQ Python requirements into the virtual environment:

    ```shell
    pip install -r requirements.txt
    ```

### Building TensorFlow Quantum from source

Once the TFQ dependencies have been installed as described in the previous
section, you should be able to build TFQ with these steps:

1.  Run the configuration script:

    ```shell
    ./configure.sh
    ```

2.  Build a pip package for TFQ:

    ```shell
    bazel build -c opt release:build_pip_package
    bazel-bin/release/build_pip_package /tmp/tfquantum/
    ```

3.  The previous step will write a file with a name ending in `.whl` into the
    directory `/tmp/tfquantum/`. Install this package file into your Python
    virtual environment:

    ```shell
    # Find out the name of the Python package (wheel) file:
    ls -l /tmp/tfquantum/
    # Once you know the file name, install it:
    pip install /tmp/tfquantum/NAME-OF-FILE.whl
    ```

4.  Run the TFQ tests:

    ```shell
    bazel test -c opt --test_output=errors //tensorflow_quantum/...
    ```

More options (such as building with optimization turned on) and other ways of
installing and building TensorFlow Quantum can be found in the [Build TensorFlow
Quantum from source](docs/install.md) section of the TFQ installation guide.

### Running tests locally

While working on changes, and especially before doing git commits, use the
following scripts to check that TFQ continues to run correctly and that code
conforms to this project's [coding conventions](#coding-conventions):

1.  `./scripts/test_all.sh`. We use TensorFlow's testing suite for our
    testing; be sure to follow the structure they have
    [outlined](https://www.tensorflow.org/api_docs/python/tf/test).

2.  `./scripts/lint_all.sh`. We use [pylint](https://www.pylint.org/) to
    ensure that code has proper formatting and is free of lint.

3.  `./scripts/format_check.sh`. We use
    [yapf](https://github.com/google/yapf) along with
    [clang-format](https://clang.llvm.org/docs/ClangFormat.html) to ensure we
    have consistent formatting everywhere.

4.  `./scripts/format_ipynb.sh`. We use [`nbformat`](
    https://pypi.org/project/nbformat/) to ensure the Jupyter notebooks have
    consistent style.

5.  `./scripts/format_all.sh`. This script runs the formatters (`yapf`,
    `clang-format`, etc.) and automatically applies any recommended changes.

You can also run the linters and formatters on individual files and
subdirectories when you don't need to check everything. Configuration files for
these programs are located at the top level of the TensorFlow Quantum
repository. (Also see the [section on configuration](#dot-files) below.)

*   For C++ code:

    *   Run `clang-format --dry-run /path/to/file` to check the format of
        the C++ file _/path/to/file_.

*   For Python code:

    *   Run `pylint /path/to/file` to check _/path/to/file_ for lint.
    *   Run `yapf -d /path/to/file` to check the format of the file.

### Adding new modules

If you are adding new modules, be sure to properly expose them to the user
using `__init__.py` files and update the `/scripts/import_test.py` file
to ensure that you are exposing them properly.

## Coding conventions

### Coding Style

TensorFlow Quantum follows the [TensorFlow code
style](https://www.tensorflow.org/community/contribute/code_style). Briefly:

*   For Python, we follow the [PEP 8 Python style
    guide](https://www.python.org/dev/peps/pep-0008/) and the [Google Python
    Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md),
    except that TensorFlow uses 2 spaces instead of 4.

*   For C++ code and Protobuf definitions, we follow the [Google C++ Style
    Guide](https://google.github.io/styleguide/cppguide.html) and
    [TensorFlow-specific specific style
    details](https://github.com/tensorflow/community/blob/master/governance/cpp-style.md).

*   For shell scripts, we follow the [Google Shell Style Guide](
    https://google.github.io/styleguide/shellguide.html).

*   For Markdown files, we follow the [Google Markdown Style Guide](
    https://google.github.io/styleguide/docguide/style.html).

### <a name="dot-files"></a>Editor, linter, and format checker configurations

Configuration files for a number of programs we use can be found at the top
level of the TensorFlow Quantum repository.

*   [`.clang-format`](.clang-format) for linting and formatting C++ code and
    Protobufs.

*   [`.editorconfig`](.editorconfig) for basic code editor configuration
    (e.g., indentation and line length) specified using the
    [EditorConfig](https://editorconfig.org/) format.

*   [`.pylintrc`](.pylintrc) for linting Python code.

*   [`.style.yapf`](.style.yapf) for reformatting Python code.

*   [`.yamllint.yaml`](.yamllint.yaml) for linting YAML files.

### Comment conventions

Every source file must begin with a header comment containing a copyright
statement and license. License headers are necessary in Python, C++, Bash/shell,
and other programming language files.

For other kinds of comments, follow these guidelines:

*   _Write clear and concise comments_: Comments should explain the "why," not
    the "what." The comments should explain the intent, trade-offs, and
    reasoning behind the implementation.

*   _Comment sparingly_: Well-written code should be self-documenting where
    possible. Avoid adding comments for code that is self-explanatory.

*   _Use complete sentences_: Start comments with a capital letter and use
    proper punctuation and correct grammar.

### Git commit do's and don't's

We strive to follow the principles of granular commits. Making smaller, focused
commits has many benefits: changes are easier for reviewers to understand and
evaluate (leading to faster and more thorough PR reviews), they allow more
effective use of tools like `git bisect` for debugging, and they make managing
changes easier with tools like `git cherry-pick` and `git rebase`.

A commit should:

*   Represent a single, self-contained change, such as a specific bug fix or
    the addition of a specific feature. It should not combine unrelated changes,
    and reverting it should not affect unrelated parts of the overall code.

*   Have an easily understood, concise title written in the imperative: "Fix
    bug ABC," and not "Fixed bug ABC" or "Fixes bug ABC." This convention fits
    well with messages generated by commands like `git merge` and `git revert`.

*   Include a description, unless the change is exceptionally small or obvious.

A commit should not:

*   Be half-done. If you‘re tempted to commit just because you need a clean
    working copy (to check out a branch, pull in changes, etc.) consider using
    [`git stash`](https://git-scm.com/docs/git-stash) instead.

*   Be untested.

## Contribution workflow

All contributions, for both code and documentation, follow the GitHub
[fork-and-pull model](https://en.wikipedia.org/wiki/Fork_and_pull_model).

### Open an issue or comment on an existing one

Before you begin work, check the [GitHub Issue
Tracker](https://github.com/tensorflow/quantum/issues) (including closed issues)
to see if your idea or bug has been discussed before.

*   _For existing open issues_: If you find an unassigned issue you'd like to
    work on, leave a comment to claim it.

*   _For new ideas_: If your idea isn't covered by an existing issue, please
    **open a new issue to discuss your proposed changes before starting work**.
    This ensures we can agree on the approach before you invest more time.

### Develop your changes

Once an issue is assigned to you, follow these steps. (This assumes you have
already forked the repo and [set up your development environment](#dev-env).)

1.  Sync your fork. Keep your fork's `master` branch up-to-date with the main
    project.

    ```shell
    # Add the main repo as an "upstream" remote (only needs to be done once).
    git remote add upstream https://github.com/tensorflow/quantum.git

    # Fetch the latest changes and update your local master.
    git checkout master
    git pull upstream master
    git push origin master
    ```

2.  Create a new [git branch](https://docs.github.com/articles/about-branches)
    in your local clone of the TFQ git repository to do your work:

    ```shell
    git checkout -b SOME-BRANCH-NAME
    ```

3.  Make your changes. Remember to write and run tests for your changes. Commit
    changes often. Write small, focused commits with clear messages.

    ```shell
    git add -A
    git commit
    ```

### Submit your changes for review

1.  Once you are done with a particular piece of work, push your changes to your
    fork on GitHub:

    ```shell
    git push origin SOME-BRANCH-NAME
    ```

2.  Open a [**draft** pull request (PR)](
    https://help.github.com/articles/creating-a-pull-request-from-a-fork)
    on GitHub. This will trigger continuous integration (CI) checks and other
    automation. The results will appear on the PR page on GitHub.

3.  Iterate on feedback.

    *   Monitor the CI checks on your PR. If any tests fail, iterate on the
        develop-test-commit-push process until the problems are resolved.

    *   Once all the checks pass and you are ready to submit the PR for
        consideration, [mark the PR as ready for review](
        https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/changing-the-stage-of-a-pull-request).

    *   Maintainers and other contributors will review your PR. Participate in
        the discussion and address any comments or requested changes by pushing
        additional commits.
