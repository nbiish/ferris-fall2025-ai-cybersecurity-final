#!/usr/bin/env bash
set -euo pipefail

# Nanoboozhoo Gradio Application Launcher
# Usage: ./run_gradio.sh [--install] [--dev]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_DIR="${SCRIPT_DIR}/python"
VENV_DIR="${SCRIPT_DIR}/.venv"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check Python version
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        log_error "Python not found. Please install Python 3.10+"
        exit 1
    fi
    
    PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    log_info "Using Python ${PYTHON_VERSION}"
}

# Create virtual environment if needed
setup_venv() {
    if [[ ! -d "${VENV_DIR}" ]]; then
        log_info "Creating virtual environment..."
        $PYTHON_CMD -m venv "${VENV_DIR}"
    fi
    
    # Activate venv
    source "${VENV_DIR}/bin/activate"
    log_info "Virtual environment activated"
}

# Install dependencies
install_deps() {
    log_info "Installing dependencies..."
    pip install --upgrade pip
    pip install -r "${PYTHON_DIR}/requirements.txt"
    log_info "Dependencies installed"
}

# Run the application
run_app() {
    cd "${SCRIPT_DIR}"
    
    log_info "Starting Nanoboozhoo Gradio Application..."
    log_info "Open http://localhost:7860 in your browser"
    echo ""
    
    # Run the Gradio app
    PYTHONPATH="${SCRIPT_DIR}" $PYTHON_CMD -m python.app
}

# Parse arguments
INSTALL_DEPS=false
DEV_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --install|-i)
            INSTALL_DEPS=true
            shift
            ;;
        --dev|-d)
            DEV_MODE=true
            shift
            ;;
        --help|-h)
            echo "Usage: ./run_gradio.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --install, -i    Install/update dependencies"
            echo "  --dev, -d        Run in development mode (auto-reload)"
            echo "  --help, -h       Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Main execution
main() {
    check_python
    setup_venv
    
    if [[ "${INSTALL_DEPS}" == true ]]; then
        install_deps
    fi
    
    # Check if gradio is installed
    if ! $PYTHON_CMD -c "import gradio" &> /dev/null; then
        log_warn "Gradio not found. Installing dependencies..."
        install_deps
    fi
    
    if [[ "${DEV_MODE}" == true ]]; then
        log_info "Running in development mode with auto-reload"
        GRADIO_WATCH=1 run_app
    else
        run_app
    fi
}

main
