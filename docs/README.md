# TyxonQ Documentation

This directory contains the complete documentation system for TyxonQ, a powerful quantum computing framework for research and development.

## Quick Start

### Building Documentation Locally

1. **Install dependencies**:
   ```bash
   pip install -r requirements-doc.txt
   ```

2. **Build HTML documentation**:
   ```bash
   cd docs
   make html
   ```

3. **View documentation**:
   ```bash
   # Open in browser
   open build/html/index.html  # macOS
   xdg-open build/html/index.html  # Linux
   start build/html/index.html  # Windows
   ```

### Live Development Server

For auto-rebuilding documentation during development:

```bash
make livehtml
```

This will start a server at `http://127.0.0.1:8000` with automatic reload on file changes.

## Multi-Language Support

### Building Chinese Documentation

```bash
make html-zh
```

### Building Japanese Documentation

```bash
make html-ja
```

### Updating Translation Files

```bash
# Extract translatable messages
make gettext

# Update .po files
make update-po

# Edit .po files in locale/ directory
# Then rebuild with language-specific make command
```

## Documentation Structure

```
doc-tyxonq/
├── source/                      # Documentation source files
│   ├── getting_started/         # Installation and quickstart guides
│   ├── user_guide/              # Comprehensive user documentation
│   ├── quantum_chemistry/       # Quantum chemistry applications
│   ├── libraries/               # Library and component documentation
│   ├── cloud_services/          # Cloud integration guides
│   ├── tutorials/               # Step-by-step tutorials
│   ├── examples/                # Code examples
│   ├── api/                     # Auto-generated API reference
│   ├── developer_guide/         # Contributing and development
│   ├── technical_references/    # Technical papers and comparisons
│   ├── _static/                 # Static files (CSS, images, JS)
│   ├── _templates/              # Custom Sphinx templates
│   ├── conf.py                  # Sphinx configuration
│   └── index.rst                # Main documentation entry
├── locale/                      # Translation files
│   ├── zh_CN/                   # Simplified Chinese
│   ├── ja_JP/                   # Japanese
│   └── pot/                     # Translation templates
├── build/                       # Generated documentation (gitignored)
├── requirements.txt             # Python dependencies
├── Makefile                     # Build automation (Unix)
└── make.bat                     # Build automation (Windows)
```

## Documentation Standards

### Writing Style

- Use clear, concise language
- Write in present tense
- Use second person ("you") for user-facing docs
- Include code examples for all features
- Add cross-references using `:doc:`, `:ref:`, `:class:`, etc.

### Code Examples

All code examples must be:
- **Executable**: Can run without modification
- **Complete**: Include all necessary imports
- **Annotated**: Have explanatory comments
- **Tested**: Verified to work with current version

Example format:

```python
"""
Example: Creating a Bell State

This example demonstrates quantum entanglement.
"""
import tyxonq as tq

# Create circuit
circuit = tq.Circuit(2)
circuit.h(0)
circuit.cnot(0, 1)

# Execute
result = circuit.compile().device('statevector').run(shots=1000)
print(result.counts)  # Output: {'00': ~500, '11': ~500}
```

### API Documentation

All public APIs must have docstrings following NumPy style:

```python
def my_function(param1, param2=None):
    """
    Short description (one line).

    Detailed description with multiple paragraphs explaining
    the function's purpose and behavior.

    Parameters
    ----------
    param1 : type
        Description of param1
    param2 : type, optional
        Description of param2 (default: None)

    Returns
    -------
    return_type
        Description of return value

    Examples
    --------
    >>> result = my_function(arg1, arg2)
    >>> print(result)
    expected output
    """
    pass
```

## Build Targets

| Command | Description |
|---------|-------------|
| `make html` | Build HTML documentation (English) |
| `make html-zh` | Build Chinese documentation |
| `make html-ja` | Build Japanese documentation |
| `make livehtml` | Start development server with auto-reload |
| `make clean` | Remove build directory |
| `make linkcheck` | Check for broken links |
| `make doctest` | Run doctest on code examples |
| `make latexpdf` | Build PDF documentation |
| `make epub` | Build EPUB documentation |
| `make gettext` | Extract translatable messages |
| `make update-po` | Update translation files |

## Sphinx Extensions Used

- **sphinx.ext.autodoc**: Auto-generate API docs from docstrings
- **sphinx.ext.napoleon**: Support NumPy/Google docstring styles
- **sphinx.ext.intersphinx**: Cross-reference other projects
- **sphinx.ext.mathjax**: Render mathematical equations
- **sphinx_design**: Modern UI components (cards, grids, tabs)
- **sphinx_copybutton**: Add copy buttons to code blocks
- **sphinxcontrib.mermaid**: Render Mermaid diagrams
- **nbsphinx**: Include Jupyter notebooks
- **myst_parser**: Markdown support

## Theme Customization

The documentation uses **PyData Sphinx Theme**. Customization is in:

- `source/conf.py`: Theme configuration
- `source/_static/css/custom.css`: Custom styles
- `source/_templates/`: Custom HTML templates

## Contributing to Documentation

1. **Find or create an issue** describing the documentation improvement
2. **Create a branch** from main
3. **Make changes** following the style guide
4. **Build locally** to verify changes: `make html`
5. **Check links**: `make linkcheck`
6. **Submit a pull request**

### Documentation Priorities

Current priorities (see design document for details):

1. ✅ **Phase 1**: Infrastructure setup (COMPLETE)
2. 🚧 **Phase 2**: Core content (IN PROGRESS)
   - Getting Started (✅ COMPLETE)
   - User Guide (scaffolding complete, content needed)
   - API Reference (scaffolding complete, docstrings needed)
3. ⏳ **Phase 3**: Quantum Chemistry module
4. ⏳ **Phase 4**: Tutorials and examples
5. ⏳ **Phase 5**: Additional sections
6. ⏳ **Phase 6**: Internationalization and QA

## Continuous Integration

Documentation is automatically built and deployed via:

- **ReadTheDocs**: Triggered on every commit to main branch
- **GitHub Actions**: Validates documentation builds on PRs

Configuration:
- `.readthedocs.yaml`: ReadTheDocs configuration
- `.github/workflows/docs.yml`: CI workflow (to be created)

## Troubleshooting

### Build Errors

**"WARNING: document isn't included in any toctree"**
- Add the document to a `.. toctree::` directive in a parent document

**"Extension error: Could not import extension"**
- Ensure all dependencies are installed: `pip install -r requirements.txt`

**"WARNING: undefined label"**
- Check that the reference target exists and is spelled correctly

### Rendering Issues

**Math equations not rendering**
- Verify MathJax is loaded in `conf.py`
- Use proper LaTeX syntax with double backslashes

**Mermaid diagrams not showing**
- Check mermaid syntax
- Verify sphinxcontrib-mermaid is installed

## Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [reStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
- [PyData Sphinx Theme](https://pydata-sphinx-theme.readthedocs.io/)
- [NumPy Docstring Guide](https://numpydoc.readthedocs.io/)

## License

Documentation is released under the same license as TyxonQ (Apache 2.0).

---

For questions or issues related to documentation, please:
- Open an issue on [GitHub](https://github.com/QureGenAI-Biotech/TyxonQ/issues)
- Refer to the :doc:`developer_guide/contributing` guide
