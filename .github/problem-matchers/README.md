# Problem Matchers

GitHub [Problem Matchers](https://github.com/actions/toolkit/blob/main/docs/problem-matchers.md) are a mechanism that enable workflow steps to scan the outputs of GitHub Actions for regex patterns and automatically write annotations in the workflow summary page. Using Problem Matchers allows information to be displayed more prominently in the GitHub user interface.

This directory contains Problem Matchers used by the GitHub Actions workflows in the [`workflows`](./workflows) subdirectory.

The following problem matcher JSON files found in this directory were copied from the [Home Assistant](https://github.com/home-assistant/core) project on GitHub. The Home Assistant project is licensed under the Apache 2.0 open-source license. The version of the files at the time they were copied was 2025.1.2.

*   [`pylint.json`](https://github.com/home-assistant/core/blob/dev/.github/workflows/matchers/pylint.json)
*   [`yamllint.json`](https://github.com/home-assistant/core/blob/dev/.github/workflows/matchers/yamllint.json)

The Pytest problem matcher originally came from the
[gh-problem-matcher-wrap](https://github.com/liskin/gh-problem-matcher-wrap/tree/master/problem-matchers)
repository (copied 2025-03-04, version 3.0.0), and was subsequently modified by
Michael Hucka. The original JSON file is Copyright © 2020 Tomáš Janoušek and
made available under the terms of the MIT license.

*   [`pytest.json`](https://github.com/liskin/gh-problem-matcher-wrap/blob/master/problem-matchers/pytest.json)

The following problem matcher for gcc came from the [Microsoft
vscode-cpptools](https://github.com/microsoft/vscode-cpptools) repository as it appeared on
2025-12-04. The last commit was by user michelleangela on 2020-01-09. The file is licensed under the
MIT license.

*   [`gcc.json`](https://github.com/microsoft/vscode-cpptools/blob/acdd5ca9d21cb1dda535594bee461beb9e8a5f06/Extension/package.json)

The following problem matcher for yapf came from the [Google
SCAAML](https://github.com/google/scaaml) repository as it appeared on 2025-12-08. The last commit
was by user jmichelp on 2022-07-03. The file is licensed under the Apache 2.0 license.

*   [`yapf.json`](https://github.com/google/scaaml/blob/6d9c3a42c527212ef77f2877419dd8f6e77eb442/.github/python_matcher.json)
