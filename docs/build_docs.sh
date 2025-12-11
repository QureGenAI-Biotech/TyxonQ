#!/bin/bash
# Multi-language Documentation Build Script for TyxonQ
# Usage: ./build_docs.sh [en|zh|ja|all|gettext|update-po]

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Build English documentation (default)
build_english() {
    print_header "Building English Documentation"
    make html
    print_success "English documentation built successfully!"
    echo -e "Output: ${GREEN}build/html/${NC}"
}

# Build Chinese documentation
build_chinese() {
    print_header "Building Chinese Documentation"
    make html-zh
    print_success "Chinese documentation built successfully!"
    echo -e "Output: ${GREEN}build/html-zh/${NC}"
}

# Build Japanese documentation
build_japanese() {
    print_header "Building Japanese Documentation"
    make html-ja
    print_success "Japanese documentation built successfully!"
    echo -e "Output: ${GREEN}build/html-ja/${NC}"
}

# Extract translatable messages
extract_gettext() {
    print_header "Extracting Translatable Messages"
    make gettext
    print_success "Translation templates extracted!"
    echo -e "Output: ${GREEN}build/locale/${NC}"
    
    # Copy to locale/pot directory for version control
    print_header "Copying POT files to locale/pot/"
    mkdir -p locale/pot
    cp -r build/locale/*.pot locale/pot/ 2>/dev/null || true
    print_success "POT files copied to locale/pot/"
}

# Update PO files
update_po_files() {
    print_header "Updating PO Translation Files"
    make update-po
    print_success "PO files updated for Chinese and Japanese!"
}

# Build all languages
build_all() {
    print_header "Building All Language Versions"
    build_english
    echo ""
    build_chinese
    echo ""
    build_japanese
    echo ""
    print_success "All language versions built successfully!"
}

# Clean builds
clean_builds() {
    print_header "Cleaning Build Directories"
    make clean-all
    print_success "All build directories cleaned!"
}

# Show usage
show_usage() {
    echo "TyxonQ Documentation Build Script"
    echo ""
    echo "Usage: ./build_docs.sh [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  en          Build English documentation (default)"
    echo "  zh          Build Chinese documentation"
    echo "  ja          Build Japanese documentation"
    echo "  all         Build all language versions"
    echo "  gettext     Extract translatable messages to POT files"
    echo "  update-po   Update PO files with latest translations"
    echo "  workflow    Full workflow: gettext -> update-po -> build all"
    echo "  clean       Clean all build directories"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./build_docs.sh en          # Build English only"
    echo "  ./build_docs.sh all         # Build all languages"
    echo "  ./build_docs.sh workflow    # Complete translation workflow"
}

# Full translation workflow
full_workflow() {
    print_header "Starting Full Translation Workflow"
    echo ""
    
    extract_gettext
    echo ""
    
    update_po_files
    echo ""
    
    build_all
    echo ""
    
    print_success "Complete translation workflow finished!"
}

# Main script logic
case "${1:-en}" in
    en|english)
        build_english
        ;;
    zh|chinese|zh_CN)
        build_chinese
        ;;
    ja|japanese|ja_JP)
        build_japanese
        ;;
    all)
        build_all
        ;;
    gettext)
        extract_gettext
        ;;
    update-po)
        update_po_files
        ;;
    workflow)
        full_workflow
        ;;
    clean)
        clean_builds
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_usage
        exit 1
        ;;
esac

exit 0
