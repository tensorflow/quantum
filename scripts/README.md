# For installation/testing scripts

TODO (mbbrough): once we have dependancies on non github version of cirq add a pip requirements file.

We have basic scripts to make doing local testin, code linting and formatting easier for you.

1. `./scripts/format_all.sh` will apply clang-format and yapf to all source files.
2. `./scripts/test_all.sh` will run all bazel py_test and cc_test rules.
3. `./scripts/lint_all.sh` will run pylint and (eventually) clang-tidy

Make sure you have run all of these checks before submitting a PR and are happy with the outputs.
