# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Build HTML documentation
html:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

# Build HTML documentation (English - alias)
html-en: html
	@echo "English documentation built (same as 'make html')."

# Build HTML documentation with live reload
livehtml:
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

# Build documentation in Chinese
html-zh:
	@$(SPHINXBUILD) -b html -D language=zh_CN "$(SOURCEDIR)" "$(BUILDDIR)/html-zh" $(SPHINXOPTS) $(O)
	@echo
	@echo "Build finished. The Chinese HTML pages are in $(BUILDDIR)/html-zh."

# Build documentation in Japanese
html-ja:
	@$(SPHINXBUILD) -b html -D language=ja_JP "$(SOURCEDIR)" "$(BUILDDIR)/html-ja" $(SPHINXOPTS) $(O)
	@echo
	@echo "Build finished. The Japanese HTML pages are in $(BUILDDIR)/html-ja."

# Extract translatable messages
gettext:
	@$(SPHINXBUILD) -b gettext "$(SOURCEDIR)" "$(BUILDDIR)/locale" $(SPHINXOPTS) $(O)
	@echo
	@echo "Build finished. The message catalogs are in $(BUILDDIR)/locale."

# Update translation files
update-po:
	sphinx-intl update -p $(BUILDDIR)/locale -l zh_CN -l ja_JP

# Update Chinese translation files only
update-po-zh:
	sphinx-intl update -p $(BUILDDIR)/locale -l zh_CN

# Update Japanese translation files only
update-po-ja:
	sphinx-intl update -p $(BUILDDIR)/locale -l ja_JP

# Build all language versions
html-all: html html-zh html-ja
	@echo
	@echo "All language versions built successfully!"

# Clean build directory
clean:
	rm -rf "$(BUILDDIR)"/*
	@echo "Build directory cleaned."

# Clean all builds including language versions
clean-all:
	rm -rf "$(BUILDDIR)"/html "$(BUILDDIR)"/html-zh "$(BUILDDIR)"/html-ja "$(BUILDDIR)"/locale
	@echo "All build directories cleaned."

# Check links
linkcheck:
	@$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS) $(O)
	@echo
	@echo "Link check complete; look for any errors in the above output " \
	      "or in $(BUILDDIR)/linkcheck/output.txt."

# Check documentation quality
doctest:
	@$(SPHINXBUILD) -b doctest "$(SOURCEDIR)" "$(BUILDDIR)/doctest" $(SPHINXOPTS) $(O)
	@echo "Testing of doctests in the sources finished, look at the " \
	      "results in $(BUILDDIR)/doctest/output.txt."

# Build PDF documentation
latexpdf:
	@$(SPHINXBUILD) -b latex "$(SOURCEDIR)" "$(BUILDDIR)/latex" $(SPHINXOPTS) $(O)
	@echo "Running LaTeX files through pdflatex..."
	$(MAKE) -C "$(BUILDDIR)/latex" all-pdf
	@echo "pdflatex finished; the PDF files are in $(BUILDDIR)/latex."

# Build EPUB documentation
epub:
	@$(SPHINXBUILD) -b epub "$(SOURCEDIR)" "$(BUILDDIR)/epub" $(SPHINXOPTS) $(O)
	@echo
	@echo "Build finished. The EPUB file is in $(BUILDDIR)/epub."

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
