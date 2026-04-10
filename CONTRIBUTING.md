# Contributing to TensorFlow Quantum

Thank you for your interest in contributing to this project! We look forward to
working with you. Here are some guidelines to get you started.

## Before you begin

### Summary

*   Read and sign the [Contributor License Agreement]
*   Read the [code of conduct].
*   Follow the [development process].

[Contributor License Agreement]: https://cla.developers.google.com/
[code of conduct]: ./CODE_OF_CONDUCT.md
[development process]: #development-process

### Sign our Contributor License Agreement

Contributions to this project must be accompanied by a [Contributor License
Agreement] (CLA). You (or your employer) retain the copyright to your
contribution; this simply gives us permission to use and redistribute your
contributions as part of the project. If you or your current employer have
already signed the Google CLA (even if it was for a different project), you
probably don't need to do it again. Visit <https://cla.developers.google.com/>
to see your current agreements or to sign a new one.

Only original work from you and other people who have signed the CLA can be
incorporated into the project. By signing the Contributor License Agreement, you
agree that your contributions are an original work of authorship.

### Review our community guidelines

In the interest of fostering an open and welcoming environment, contributors and
maintainers pledge to make participation in our project and our community a
harassment-free experience for everyone. Our community aspires to treat everyone
equally, and to value all contributions. Please review our [code of conduct] for
more information.

## Code base conventions

TensorFlow Quantum (TFQ) is a Python framework for quantum machine learning
(QML) implemented as an add-on to [TensorFlow]. User documentation for TFQ is
available on the [TensorFlow Quantum documentation site]. The TFQ project
generally follows TensorFlow development practices, and the [TensorFlow
contribution guide] is essential reading if you want to get involved with TFQ.

[TensorFlow]: https://tensorflow.org
[TensorFlow Quantum documentation site]: https://tensorflow.org/quantum
[TensorFlow contribution guide]: https://www.tensorflow.org/community/contribute

### Getting oriented

Here is a summary of the main subdirectories in the TFQ source tree:

*   `benchmarks/`: Code for performance benchmarking
*   `docs/`: Documentation source files
*   `release/`: Scripts and configurations for building TFQ releases
*   `scripts/`: Utility for running tests and doing other tasks
*   `tensorflow_quantum/`: The core source code for TensorFlow Quantum
*   `third_party/`: External dependencies and third-party integrations
*   *`github/`: GitHub-specific configurations and workflows

### Coding style

This project follows the [TensorFlow style], which in turn follows these Google
style guides:

*   [C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
*   [Python Style Guide](https://google.github.io/styleguide/pyguide.html)
*   [Markdown Style Guide](https://google.github.io/styleguide/docguide/style.html)
*   [Shell Style Guide](https://google.github.io/styleguide/shellguide.html)

Software tool configurations can be found in the following files at the top
level of the source tree:

*   `.editorconfig`: basic code editor configuration
*   `.pylintrc`: configuration for linting Python files using [Pylint]
*   `.style.yapf`: configuration for formatting Python files using [YAPF]
*   `.yamllint.yaml`: configuration for linting YAML files using [yamllint]

All new source code files longer than 2 lines must begin with a header comment
with the copyright and license.

[Pylint]: https://www.pylint.org/
[YAPF]: https://github.com/google/yapf
[yamllint]: https://github.com/adrienverge/yamllint
[TensorFlow style]: ttps://www.tensorflow.org/community/contribute/code_style

### Git conventions

Git commits should be small and focused. Granular commits make changes easier to
understand and evaluate (leading to faster and more thorough PR reviews), allow
more effective use of tools like `git bisect` for debugging, and allow easier
management of changes with tools like `git cherry-pick` and `git rebase`.

Each commit should:

*   Represent a single, self-contained change, such as a specific bug fix or the
    addition of a specific feature.

*   Not combine unrelated changes. Reverting a commit should not affect
    unrelated parts of the overall code.

*   Have an easily understood, concise title written in the imperative: "Fix bug
    ABC," and not "Fixed bug ABC" or "Fixes bug ABC."

*   Include a description, unless the change is exceptionally small or obvious.

## Development process

TFQ development takes place on GitHub using a GitHub-centric workflow.

### Past issues

First search the [issue tracker](https://github.com/tensorflow/quantum/issues)
to check if your idea or bug has been discussed before.

Before beginning on any substantial changes, we recommend opening a new issue on
GitHub (if one doesn't already exist for the topic) and discussing your proposed
changes. This will let us give you advice on the proposed changes.

### Repository forks and branches

The preferred approach to working on TensorFlow Quantum is to first create a
[fork](https://docs.github.com/articles/fork-a-repo) of the repository in your
GitHub account, then clone that fork to your local computing environment. Keep
your fork regularly synchronized with the upstream TFQ repository. Create a
separate [git branch](https://docs.github.com/articles/about-branches) for your
work on individual issues or topics.

### Set up your environment

Follow the instructions in [docs/install.md](docs/install.md) for setting up a
development environment. After doing that, you should end up with:

*   The correct version of Bazel (6.5.0)
*   A Python virtual environment
*   The TFQ Python requirements installed in that Python virtual environment
*   The TFQ build configured by running `./configure.sh`

### Adding modules

If you are adding new modules, be sure to properly expose them to the user using
`__init__.py` files and update the `scripts/import_test.py` file to ensure that
you are exposing them properly.

### Linting and formatting

Code should meet common style standards for Python and be free of error-prone
constructs. Use the following commands regularly to lint and reformat your code
according to project conventions:

```shell
scripts/lint_all.sh
scripts/format_check.sh
```

If the format check reports problems, you can correct them automatically using

```shell
scripts/format_all.sh
```

### Builds

For relatively "quick" builds of TFQ during development, you can use the
following command:

```shell
bazel build -c opt --cxxopt="-O3" --cxxopt="-march=native" release:build_pip_package
```

("Quick" here is relative: depending on the capabilities of your computer, a
build takes anywhere from a few minutes to tens of minutes.)

### Running tests

When new functions, classes, and files are introduced, they should also have
corresponding tests. Bug fixes also generally require new unit tests, because
the presence of bugs usually indicates insufficient test coverage. Existing
tests must continue to pass (or be updated) when changes are introduced.

We use TensorFlow's testing suite for our testing. Tests must follow the
[TensorFlow test guidelines](https://www.tensorflow.org/api_docs/python/tf/test)
in order to work correctly. To run the TFQ test suite, run this command:

```shell
scripts/test_all.sh
```

### Contributing code

All submissions require review. We use GitHub's tools for code reviews on
[pull requests](https://docs.github.com/articles/about-pull-requests).

#### Final checks

Before opening a pull request (PR) and requesting a code review, you should make
sure that the following tests are passing locally:

```shell
scripts/lint_all.sh
scripts/format_check.sh
scripts/test_all.sh
```

#### Draft pull requests

When getting ready to submit your work, first create a _draft_ pull request from
your branch on GitHub to the main project repository. (Consult GitHub's
[docs](https://help.github.com/articles/creating-a-pull-request-from-a-fork) for
help on creating pull requests.) The pull request will trigger continuous
integration (CI) checks and other automation on GitHub. Monitor the checks; if
any tests fail, continue development and testing to resolve the problems.

#### Code review

Once all the CI checks pass and you are ready to submit the PR for
consideration, [mark the PR as ready for review]. A reviewer from the TFQ team
will comment on your code and may ask for changes. You can perform the necessary
changes locally, commit them to your branch as usual, and then push changes to
your fork on GitHub following the same process as above. When you do that,
GitHub will update the code in the pull request automatically.

[mark the PR as ready for review]: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/changing-the-stage-of-a-pull-request
