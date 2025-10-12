============
Contributing
============

Thank you for your interest in contributing to TyxonQ! This guide will help you get started with contributing
to our quantum computing framework.

.. contents:: Contents
   :depth: 3
   :local:

Getting Started
===============

Types of Contributions
----------------------

We welcome all types of contributions:

🐛 **Bug Reports**: Help us identify and fix issues  
✨ **Feature Requests**: Suggest new functionality  
📝 **Documentation**: Improve guides, examples, and API docs  
🔧 **Code Contributions**: Implement features and fix bugs  
🧪 **Testing**: Add tests and improve coverage  
📊 **Performance**: Optimize algorithms and implementations  
🎨 **Examples**: Create tutorials and demonstrations  

Where to Start
--------------

**New contributors should**:

1. **Read this guide** completely
2. **Browse existing issues** with labels:
   
   - `good first issue`: Perfect for newcomers
   - `help wanted`: Community contributions needed
   - `documentation`: Documentation improvements
   - `bug`: Bug fixes

3. **Join our community**:
   
   - GitHub Discussions for questions
   - Issue comments for specific topics
   - Developer meetings (monthly)

4. **Start small**: Begin with documentation or simple bug fixes

Development Environment Setup
=============================

Prerequisites
-------------

**Required software**:

- **Python 3.8+**: Latest Python version recommended
- **Git**: For version control
- **GitHub account**: For submitting contributions

**Optional but recommended**:

- **VS Code**: With Python and Git extensions
- **Docker**: For consistent development environment

Cloning the Repository
----------------------

.. code-block:: bash

   # Clone your fork
   git clone https://github.com/YOUR_USERNAME/TyxonQ.git
   cd TyxonQ
   
   # Add upstream remote
   git remote add upstream https://github.com/TyxonQ/TyxonQ.git
   
   # Verify remotes
   git remote -v

Setting Up Development Environment
----------------------------------

**Option 1: Virtual Environment (Recommended)**

.. code-block:: bash

   # Create virtual environment
   python -m venv tyxonq-dev
   
   # Activate (Linux/Mac)
   source tyxonq-dev/bin/activate
   
   # Activate (Windows)
   tyxonq-dev\Scripts\activate
   
   # Install in development mode
   pip install -e .
   
   # Install development dependencies
   pip install -r requirements-dev.txt

**Option 2: Conda Environment**

.. code-block:: bash

   # Create conda environment
   conda create -n tyxonq-dev python=3.9
   conda activate tyxonq-dev
   
   # Install dependencies
   pip install -e .
   pip install -r requirements-dev.txt

**Option 3: Docker Development**

.. code-block:: bash

   # Build development container
   docker build -t tyxonq-dev -f Dockerfile.dev .
   
   # Run development container
   docker run -it -v $(pwd):/workspace tyxonq-dev

Verifying Installation
----------------------

.. code-block:: bash

   # Run tests to verify setup
   pytest tests/unit/
   
   # Run simple import test
   python -c "import tyxonq; print('TyxonQ imported successfully!')"
   
   # Check development tools
   black --version
   flake8 --version
   mypy --version

Development Workflow
====================

Fork and Branch Strategy
------------------------

**Our branching model**:

- `main`: Stable release branch
- `develop`: Development integration branch
- `feature/*`: Feature development branches
- `bugfix/*`: Bug fix branches
- `hotfix/*`: Critical fixes for releases

**Creating a feature branch**:

.. code-block:: bash

   # Update your fork
   git fetch upstream
   git checkout main
   git merge upstream/main
   
   # Create feature branch
   git checkout -b feature/your-feature-name
   
   # Or for bug fixes
   git checkout -b bugfix/issue-123-description

Making Changes
--------------

**Development best practices**:

1. **Write tests first** (TDD approach)
2. **Make small, focused commits**
3. **Follow code style guidelines**
4. **Update documentation**
5. **Add examples if needed**

**Example development cycle**:

.. code-block:: bash

   # 1. Write failing test
   echo "def test_my_feature(): assert False" >> tests/test_my_feature.py
   pytest tests/test_my_feature.py  # Should fail
   
   # 2. Implement feature
   # Edit src/tyxonq/...
   
   # 3. Make test pass
   pytest tests/test_my_feature.py  # Should pass
   
   # 4. Run all tests
   pytest
   
   # 5. Check code style
   black src/ tests/
   flake8 src/ tests/
   mypy src/

Commit Guidelines
-----------------

**Commit message format**:

.. code-block:: text

   type(scope): brief description
   
   Longer description if needed.
   
   Fixes #123

**Commit types**:

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `perf`: Performance improvements
- `chore`: Build process, dependencies, etc.

**Examples**:

.. code-block:: bash

   git commit -m "feat(core): add RXX gate implementation
   
   Implements the RXX two-qubit gate with parameter support.
   Includes tests and documentation.
   
   Fixes #456"
   
   git commit -m "fix(devices): resolve simulator memory leak
   
   Fixed memory not being properly released after simulation.
   
   Fixes #789"

Code Quality Standards
======================

Code Style
----------

**We use automated code formatting**:

.. code-block:: bash

   # Format all code
   black src/ tests/ examples/
   
   # Check style
   flake8 src/ tests/
   
   # Type checking
   mypy src/

**Style configuration**:

- **Line length**: 100 characters
- **Indentation**: 4 spaces
- **Quotes**: Double quotes preferred
- **Imports**: isort for import sorting

Documentation Standards
-----------------------

**All public APIs must have docstrings**:

.. code-block:: python

   def quantum_gate(qubit: int, angle: float) -> Circuit:
       """Apply a parameterized quantum gate.
       
       Args:
           qubit: Target qubit index (0-based)
           angle: Rotation angle in radians
       
       Returns:
           Updated circuit with gate applied
       
       Raises:
           ValueError: If qubit index is invalid
       
       Examples:
           >>> circuit = Circuit(2)
           >>> circuit = quantum_gate(circuit, 0, np.pi/2)
           >>> print(len(circuit.ops))
           1
       """
       # Implementation here
       pass

**Documentation checklist**:

- ✅ **Args**: All parameters documented
- ✅ **Returns**: Return value described
- ✅ **Raises**: Exceptions documented
- ✅ **Examples**: Working code examples
- ✅ **Type hints**: Full type annotations

Testing Requirements
--------------------

**Test coverage expectations**:

- **New features**: 100% test coverage
- **Bug fixes**: Test reproducing the bug
- **Edge cases**: Boundary conditions tested
- **Error conditions**: Exception handling tested

**Test structure**:

.. code-block:: python

   import pytest
   import numpy as np
   from tyxonq import Circuit
   
   class TestQuantumGate:
       """Test suite for quantum gate functionality."""
       
       def test_valid_input(self):
           """Test gate with valid parameters."""
           circuit = Circuit(2)
           result = quantum_gate(circuit, 0, np.pi/2)
           assert len(result.ops) == 1
       
       def test_invalid_qubit_index(self):
           """Test error handling for invalid qubit."""
           circuit = Circuit(2)
           with pytest.raises(ValueError, match="Invalid qubit index"):
               quantum_gate(circuit, 5, np.pi/2)
       
       @pytest.mark.parametrize("angle", [0, np.pi/4, np.pi/2, np.pi])
       def test_various_angles(self, angle):
           """Test gate with different angles."""
           circuit = Circuit(1)
           result = quantum_gate(circuit, 0, angle)
           # Add specific assertions

**Running tests**:

.. code-block:: bash

   # Run all tests
   pytest
   
   # Run with coverage
   pytest --cov=src/tyxonq --cov-report=html
   
   # Run specific test file
   pytest tests/test_core.py
   
   # Run with verbose output
   pytest -v
   
   # Run only failed tests
   pytest --lf

Submitting Contributions
========================

Pull Request Process
--------------------

**Before submitting**:

1. **Update your branch**:
   
   .. code-block:: bash
   
      git fetch upstream
      git rebase upstream/main

2. **Run full test suite**:
   
   .. code-block:: bash
   
      pytest
      black --check src/ tests/
      flake8 src/ tests/
      mypy src/

3. **Update documentation**:
   
   .. code-block:: bash
   
      cd docs/
      make html
      # Check for warnings

Pull Request Template
---------------------

**Use this template for your PR description**:

.. code-block:: text

   ## Description
   Brief description of changes and motivation.
   
   ## Type of Change
   - [ ] Bug fix (non-breaking change which fixes an issue)
   - [ ] New feature (non-breaking change which adds functionality)
   - [ ] Breaking change (fix or feature that would cause existing functionality to change)
   - [ ] Documentation update
   
   ## Testing
   - [ ] Tests pass locally
   - [ ] Added tests for new functionality
   - [ ] Updated existing tests as needed
   
   ## Documentation
   - [ ] Updated docstrings
   - [ ] Updated user documentation
   - [ ] Added examples if applicable
   
   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Changes are backward compatible (or breaking changes documented)
   - [ ] Related issues referenced
   
   ## Related Issues
   Fixes #123
   Closes #456

Review Process
--------------

**What to expect**:

1. **Automated checks**: CI/CD pipeline runs tests
2. **Code review**: Maintainers review your changes
3. **Feedback incorporation**: Address review comments
4. **Approval**: At least one maintainer approval required
5. **Merge**: Changes integrated into main branch

**Review criteria**:

- ✅ **Functionality**: Code works as intended
- ✅ **Quality**: Follows coding standards
- ✅ **Tests**: Adequate test coverage
- ✅ **Documentation**: Properly documented
- ✅ **Performance**: No significant performance regression
- ✅ **Compatibility**: Maintains backward compatibility

Community Guidelines
====================

Code of Conduct
---------------

**Our community values**:

- **Respect**: Treat everyone with kindness and respect
- **Inclusivity**: Welcome contributors from all backgrounds
- **Constructiveness**: Provide helpful, actionable feedback
- **Patience**: Help newcomers learn and grow
- **Collaboration**: Work together toward shared goals

**Unacceptable behavior**:

- Harassment or discrimination
- Insulting or derogatory comments
- Public or private harassment
- Publishing private information
- Trolling or inflammatory comments

Communication Channels
----------------------

**GitHub Discussions**: General questions and design discussions  
**GitHub Issues**: Bug reports and specific feature requests  
**Pull Request Comments**: Code review and implementation discussion  
**Developer Meetings**: Monthly video calls (open to all)  

**Response expectations**:

- **Maintainers**: Respond within 2-3 business days
- **Community**: Help each other promptly
- **Critical issues**: Same-day response for security/blocking issues

Recognition and Credits
=======================

Contributor Recognition
-----------------------

**How we recognize contributions**:

- **Contributors file**: All contributors listed
- **Release notes**: Major contributions highlighted
- **GitHub profile**: Contribution activity visible
- **Community shoutouts**: Recognition in discussions and meetings
- **Conference opportunities**: Speaking opportunities for major contributors

Maintainer Path
---------------

**Becoming a core maintainer**:

1. **Regular contributions**: Consistent, high-quality contributions
2. **Community involvement**: Active in discussions and reviews
3. **Domain expertise**: Deep knowledge in specific areas
4. **Mentorship**: Help onboard new contributors
5. **Invitation**: Current maintainers invite promising contributors

**Maintainer responsibilities**:

- Code review and approval
- Issue triage and labeling
- Release management
- Community leadership
- Technical decision making

Special Projects
================

Google Summer of Code
---------------------

**TyxonQ participates in GSoC**:

- **Project ideas**: Listed on our wiki
- **Mentor support**: Experienced contributors guide students
- **Timeline**: Follow GSoC schedule
- **Requirements**: University student eligibility

Hackathons and Contests
-----------------------

**Regular community events**:

- **Monthly challenges**: Algorithm implementation contests
- **Documentation sprints**: Focused documentation improvement
- **Bug bounties**: Rewards for finding and fixing critical issues
- **Conference hackathons**: TyxonQ tracks at quantum computing conferences

Research Collaborations
-----------------------

**Academic partnerships**:

- **Paper implementations**: Help researchers implement quantum algorithms
- **Benchmarking**: Comparative studies and performance analysis
- **Grant applications**: Collaborate on research funding
- **Publications**: Co-authorship opportunities for significant contributions

Troubleshooting
===============

Common Issues
-------------

**Development environment problems**:

.. code-block:: bash

   # Issue: Import errors after installation
   # Solution: Reinstall in development mode
   pip uninstall tyxonq
   pip install -e .
   
   # Issue: Tests failing with permission errors
   # Solution: Check file permissions
   chmod +x scripts/run_tests.sh
   
   # Issue: Black formatting conflicts
   # Solution: Use exact same version
   pip install black==22.3.0

**Git workflow issues**:

.. code-block:: bash

   # Issue: Merge conflicts
   # Solution: Rebase your branch
   git fetch upstream
   git rebase upstream/main
   # Resolve conflicts manually, then:
   git rebase --continue
   
   # Issue: Accidentally committed to main
   # Solution: Move commits to feature branch
   git branch feature/my-changes
   git reset --hard upstream/main
   git checkout feature/my-changes

Getting Help
------------

**If you're stuck**:

1. **Check existing issues**: Someone might have faced the same problem
2. **Search discussions**: Look for similar questions
3. **Ask in discussions**: Post your question with context
4. **Join developer meetings**: Get real-time help
5. **Contact maintainers**: For complex technical issues

**When asking for help, include**:

- Operating system and Python version
- TyxonQ version
- Complete error messages
- Steps to reproduce the issue
- What you've already tried

Next Steps
==========

**Ready to contribute?**

1. **Set up your development environment**
2. **Find a good first issue** on GitHub
3. **Read the relevant documentation**:
   
   - :doc:`architecture_overview` - Understand the codebase
   - :doc:`testing_guidelines` - Learn our testing practices
   - :doc:`extending_tyxonq` - For feature development

4. **Start coding and have fun!**

**Thank you for contributing to TyxonQ!** 🚀✨

Your contributions help make quantum computing accessible to everyone.
