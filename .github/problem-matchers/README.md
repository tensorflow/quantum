# Problem Matchers

GitHub [Problem Matchers](https://github.com/actions/toolkit/blob/main/docs/problem-matchers.md) are a mechanism that enable workflow steps to scan the outputs of GitHub Actions for regex patterns and automatically write annotations in the workflow summary page. Using Problem Matchers allows information to be displayed more prominently in the GitHub user interface.

This directory contains Problem Matchers used by the GitHub Actions workflows in the [`workflows`](./workflows) subdirectory.

The following problem matcher JSON files found in this directory were copied from the [Home Assistant](https://github.com/home-assistant/core) project on GitHub. The Home Assistant project is licensed under the Apache 2.0 open-source license. The version of the files at the time they were copied was 2025.1.2.

- [`pylint.json`](https://github.com/home-assistant/core/blob/dev/.github/workflows/matchers/pylint.json)
- [`yamllint.json`](https://github.com/home-assistant/core/blob/dev/.github/workflows/matchers/yamllint.json)
