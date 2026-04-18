# Contributing

## Contributor License Agreements

We'd love to accept your patches! Before we can take them, we have to jump a
couple of legal hurdles.

Please fill out either the individual or corporate Contributor License Agreement
(CLA).

*   If you are an individual writing original source code and you're sure you
    own the intellectual property, then you'll need to sign an
    [individual CLA](http://code.google.com/legal/individual-cla-v1.0.html).
*   If you work for a company that wants to allow you to contribute your work,
    then you'll need to sign a
    [corporate CLA](http://code.google.com/legal/corporate-cla-v1.0.html).

Follow either of the two links above to access the appropriate CLA and
instructions for how to sign and return it. Once we receive it, we'll be able to
accept your pull requests.

NOTE: Only original source code from you and other people that have signed the
CLA can be accepted into the main repository.

## Code Reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests and the
[TensorFlow Community Guidelines](https://www.tensorflow.org/community/contribute)
for more information on contributor best practices.

Before making any changes, we recommend opening an issue (if it doesn't already
exist) and discussing your proposed changes. This will let us give you advice
on the proposed changes. If the changes are minor, then feel free to make
them without discussion.

## Code Standards

We have some standards in place to ensure that incoming code is the highest
quality it can be. Before a code review can happen you should make sure that
the following tests are passing locally:

1. `./scripts/test_all.sh` passes. We use TensorFlow's testing suite for our
testing and be sure that any code you add follows the structure they have
[outlined](https://www.tensorflow.org/api_docs/python/tf/test).

2. `./scripts/lint_all.sh` passes. We use [pylint](https://www.pylint.org/)
to ensure that code has proper formatting and is lint free.

3. `./scripts/format_check.sh` passes. We use
[yapf](https://github.com/google/yapf) along with
[clang format](https://clang.llvm.org/docs/ClangFormat.html) to ensure we have
consistent formatting everywhere.

### Pre-commit Hooks

This repository ships a [pre-commit](https://pre-commit.com/) configuration
(`.pre-commit-config.yaml`) that automates all of the checks above so they run
automatically before every `git commit`. We strongly recommend installing the
hooks when you set up your local development environment:

```bash
# Install the pre-commit tool (already listed in requirements.txt).
pip install pre-commit

# Register the hooks with your local Git clone – only needed once.
pre-commit install
```

After installation the hooks run automatically on each `git commit`. You can
also run them manually at any time:

```bash
# Run all hooks against every file in the repository.
pre-commit run --all-files

# Run a single hook (e.g. yapf formatting) against all files.
pre-commit run yapf --all-files
```

The hooks cover the following checks:

| Hook | What it does |
|------|-------------|
| `trailing-whitespace` | Removes trailing whitespace from source files |
| `end-of-file-fixer` | Ensures files end with a single newline |
| `check-yaml` / `check-json` | Validates YAML and JSON syntax |
| `check-merge-conflict` | Rejects unresolved merge-conflict markers |
| `check-added-large-files` | Rejects files larger than 500 KB |
| `mixed-line-ending` | Normalises all line endings to LF |
| `no-commit-to-branch` | Prevents direct commits to `master` |
| `isort` | Sorts Python imports (Google profile) |
| `yapf` | Formats Python source with Google style |
| `pylint` | Lints Python source using `.pylintrc` |
| `yamllint` | Lints YAML files using `.yamllint.yaml` |
| `clang-format` | Formats C++ / C source with Google style |
| `format-ipynb` | Formats Jupyter notebook code cells with yapf |

### Adding Modules

If you are adding new modules, be sure to properly expose them to the user
using `__init__.py` files and update the `/scripts/import_test.py` file
to ensure that you are exposing them properly.

