# Contributing and Development

<cite>
**Referenced Files in This Document**   
- [CONTRIBUTING.md](file://CONTRIBUTING.md)
- [docs/source/contribution.rst](file://docs/source/contribution.rst)
- [pyproject.toml](file://pyproject.toml)
- [CHANGELOG.md](file://CHANGELOG.md)
</cite>

## Table of Contents
1. [Introduction](#introduction)
2. [Contribution Guidelines](#contribution-guidelines)
3. [Development Setup](#development-setup)
4. [Testing Infrastructure and CI Requirements](#testing-infrastructure-and-ci-requirements)
5. [Release Process and Versioning Strategy](#release-process-and-versioning-strategy)
6. [Community Engagement](#community-engagement)
7. [Documentation Standards and Review Processes](#documentation-standards-and-review-processes)

## Introduction
This document provides comprehensive guidance for contributing to and developing the TyxonQ quantum computing framework. It outlines the processes, tools, and standards required for effective community participation in code contributions, testing, documentation, and release management. The goal is to ensure high-quality, maintainable, and collaborative development of this open-source quantum computing platform.

## Contribution Guidelines

TyxonQ welcomes contributions from the community through various channels including answering questions, reporting issues, improving documentation, and submitting code changes via pull requests. All contributions should follow the established workflow and quality standards to maintain code integrity and project consistency.

The contribution process begins with forking the repository and creating a dedicated feature branch for each proposed change. Contributors are encouraged to discuss major enhancements or API changes through GitHub issues before submitting large pull requests. Each pull request should represent a single, coherent change and be accompanied by appropriate tests and documentation updates.

**Section sources**
- [CONTRIBUTING.md](file://CONTRIBUTING.md)
- [docs/source/contribution.rst](file://docs/source/contribution.rst)

## Development Setup

Development environment configuration is managed through `pyproject.toml`, which specifies project metadata, dependencies, and build requirements. The framework requires Python 3.10 or higher and supports multiple backends including NumPy, PyTorch, and CuPy.

Core dependencies include scientific computing libraries such as NumPy, SciPy, SymPy, and quantum-specific packages like Qiskit and OpenFermion. Development dependencies are specified in the optional "dev" group and include tools for testing, building, and packaging such as pytest, build, and twine.

To set up a development environment, install the required packages using pip and configure the local installation in development mode to enable immediate testing of code changes.

**Section sources**
- [pyproject.toml](file://pyproject.toml)

## Testing Infrastructure and CI Requirements

The testing infrastructure is configured through pytest with specific options defined in `pyproject.toml`. The test suite is organized across multiple directories including `tests_core_module`, `tests_examples`, `tests_applications_chem`, and `tests_mol_valid`, covering different aspects of the framework.

Testing requirements include running code formatting checks with black, type checking with mypy (using numpy==1.21.5 as standard), linting with pylint, and executing the full test suite with pytest. The CI pipeline automatically runs these checks on pull requests to ensure code quality and compatibility.

Parallel testing is supported through pytest-xdist, and benchmark testing is available with pytest-benchmark. Various fixtures are provided for testing across different backends and data types, enabling comprehensive validation of functionality.

**Section sources**
- [pyproject.toml](file://pyproject.toml)
- [docs/source/contribution.rst](file://docs/source/contribution.rst)

## Release Process and Versioning Strategy

TyxonQ follows Semantic Versioning (SemVer) as specified in the changelog, with version numbers reflecting the nature of changes in each release. The changelog documents all notable changes, organized by version number and release date, following the Keep a Changelog format.

The release process involves several steps: updating version numbers in configuration files, creating git tags, and publishing releases to multiple platforms. GitHub releases are created from tags, PyPI packages are uploaded using twine, Docker images are pushed to DockerHub, and Binder environments are updated to reflect the new version.

Each release must be accompanied by appropriate updates to the CHANGELOG.md file, documenting all added features, changes, and fixes in a structured format that helps users understand the evolution of the framework.

**Section sources**
- [CHANGELOG.md](file://CHANGELOG.md)
- [docs/source/contribution.rst](file://docs/source/contribution.rst)

## Community Engagement

Community participation is facilitated through multiple channels including GitHub for code contributions and issue tracking, Discord for real-time discussions, and WeChat for developer community engagement. The project encourages users to report bugs, request features, and contribute improvements through the established processes.

For large-scale contributions or architectural changes, the project recommends opening a GitHub issue to discuss the proposal before implementation. This ensures alignment with project goals and allows for feedback from the core development team before significant effort is invested.

The community is also encouraged to participate in documentation improvements, translation efforts, and tutorial development to make the framework more accessible to users worldwide.

**Section sources**
- [CONTRIBUTING.md](file://CONTRIBUTING.md)
- [README_jp.md](file://README_jp.md)

## Documentation Standards and Review Processes

Documentation is managed using Sphinx, with source files in reStructuredText format located in the docs/source directory. The project maintains both English and Chinese documentation, with internationalization support for additional languages.

API documentation is automatically generated from the codebase using a dedicated script that creates RST files from docstrings. Special attention is required for mathematical formulas to ensure compatibility between Markdown and LaTeX rendering, particularly avoiding nested equation environments that cause PDF build failures.

The review process for contributions includes automated checks for code style, type safety, and test coverage, followed by manual review by core developers. All pull requests must pass the full test suite and include appropriate documentation updates before being merged into the main branch.

**Section sources**
- [docs/source/contribution.rst](file://docs/source/contribution.rst)