#!/usr/bin/env bash
# start.sh — Pre-flight checks and Streamlit launcher for VibeVoice ASR.
#
# This script:
#   1. Checks whether a Python virtual environment is active; if not, activates it.
#   2. If the virtual environment does not exist, runs setup.sh to create it.
#   3. Verifies that all required Python packages are installed.
#   4. Performs AMD and NVIDIA GPU pre-flight checks.
#   5. Launches the Streamlit web application.
#
# Usage:
#   bash start.sh

set -e

# ── Colour helpers (disabled when output is not a terminal) ───────────────
if [ -t 1 ]; then
  RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'
else
  RED=''; GREEN=''; YELLOW=''; NC=''
fi

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── Resolve script directory ──────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"

info "Project directory: ${SCRIPT_DIR}"

# ── 1. Check if a virtual environment is already active ───────────────────
if [ -n "${VIRTUAL_ENV:-}" ]; then
  info "Virtual environment is already active: ${VIRTUAL_ENV}"
else
  # ── 2. Activate (or create) the virtual environment ─────────────────────
  if [ -f "${VENV_DIR}/bin/activate" ]; then
    info "Activating virtual environment at ${VENV_DIR}…"
    # shellcheck disable=SC1091
    . "${VENV_DIR}/bin/activate"
  else
    warn "Virtual environment not found at ${VENV_DIR}."
    warn "Running setup.sh to create it…"

    if [ ! -f "${SCRIPT_DIR}/setup.sh" ]; then
      error "setup.sh not found in ${SCRIPT_DIR}. Cannot create the virtual environment."
      exit 1
    fi

    # shellcheck disable=SC1091
    . "${SCRIPT_DIR}/setup.sh"

    # After setup.sh runs, the venv should be activated. Verify.
    if [ -z "${VIRTUAL_ENV:-}" ]; then
      error "setup.sh finished but the virtual environment is not active."
      exit 1
    fi
  fi

  # ── Verify activation ───────────────────────────────────────────────────
  if [ -z "${VIRTUAL_ENV:-}" ]; then
    error "Virtual environment activation failed. Aborting."
    exit 1
  fi
  info "Active venv: ${VIRTUAL_ENV}"
  info "Python: $(python --version 2>&1)"
fi

# ── 3. Check required Python packages ─────────────────────────────────────
info "Checking required Python packages…"

REQUIRED_PACKAGES="torch transformers accelerate librosa streamlit"
MISSING_PACKAGES=()

for pkg in ${REQUIRED_PACKAGES}; do
  if python -c "import ${pkg}" &>/dev/null; then
    info "  ✓ ${pkg}"
  else
    warn "  ✗ ${pkg} (missing or broken)"
    MISSING_PACKAGES+=("${pkg}")
  fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
  warn "Missing packages: ${MISSING_PACKAGES[*]}"
  warn "Installing missing packages from requirements.txt…"

  if [ -f "${SCRIPT_DIR}/requirements.txt" ]; then
    pip install -r "${SCRIPT_DIR}/requirements.txt" --quiet || {
      error "Failed to install required packages. Aborting."
      exit 1
    }
    info "Packages installed. Re-verifying…"

    # Re-check the previously missing packages
    STILL_MISSING=()
    for pkg in "${MISSING_PACKAGES[@]}"; do
      if python -c "import ${pkg}" &>/dev/null; then
        info "  ✓ ${pkg}"
      else
        warn "  ✗ ${pkg} (still missing after install)"
        STILL_MISSING+=("${pkg}")
      fi
    done

    if [ ${#STILL_MISSING[@]} -gt 0 ]; then
      error "The following packages could not be installed: ${STILL_MISSING[*]}"
      error "Please install them manually and try again."
      exit 1
    fi
  else
    error "requirements.txt not found in ${SCRIPT_DIR}."
    exit 1
  fi
else
  info "All required packages are available."
fi

# ── 4. GPU pre-flight check (AMD / NVIDIA) ──────────────────────────────
info "Checking GPU availability…"

# ── AMD GPU check ──────────────────────────────────────────────────────
AMD_GPU_FOUND=false
if command -v lspci &>/dev/null; then
  AMD_GPU_LINE=$(lspci 2>/dev/null | grep -iE 'vga|3d|display' | grep -i -E 'amd|ati|radeon' || echo "")
  if [ -n "$AMD_GPU_LINE" ]; then
    AMD_GPU_NAME=$(echo "$AMD_GPU_LINE" | sed 's/.*: *//' || echo "unknown")
    info "AMD GPU: ${AMD_GPU_NAME}"
    AMD_GPU_FOUND=true

    # Check ROCm runtime
    if command -v rocminfo &>/dev/null && rocminfo &>/dev/null; then
      ROCM_VERSION=$(rocminfo 2>/dev/null | grep "ROCk module" | head -1 || echo "unknown")
      info "ROCm runtime: ${ROCM_VERSION}"
    elif [ -f /opt/rocm/bin/rocminfo ]; then
      info "ROCm runtime installed at /opt/rocm"
    else
      warn "ROCm runtime not found — GPU acceleration may not be available."
    fi
  fi
fi

# ── NVIDIA GPU check ───────────────────────────────────────────────────
if ! $AMD_GPU_FOUND; then
  if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
    info "NVIDIA GPU: ${GPU_NAME}"
    # Verify nvidia-smi can communicate with the GPU driver
    nvidia-smi --query-gpu=temperature.gpu,memory.used --format=csv,noheader 2>/dev/null &>/dev/null || {
      warn "nvidia-smi reported an error — GPU may not be fully functional."
    }
  else
    warn "nvidia-smi not found — NVIDIA GPU not detected or driver not installed."
  fi
fi

# ── PyTorch GPU availability check ─────────────────────────────────────
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  # Check if we're on ROCm (HIP) or CUDA
  if python -c "import torch; print(torch.version.hip or '')" 2>/dev/null | grep -q '^[0-9]'; then
    HIP_VERSION=$(python -c "import torch; print(torch.version.hip)" 2>/dev/null || echo "unknown")
    GPU_DEVICE=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "unknown")
    info "ROCm/HIP ${HIP_VERSION} available — GPU: ${GPU_DEVICE}"
  else
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "unknown")
    GPU_DEVICE=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "unknown")
    info "CUDA ${CUDA_VERSION} available — GPU: ${GPU_DEVICE}"
  fi
else
  warn "torch.cuda.is_available() returned False."
  warn "PyTorch will run on CPU. For GPU acceleration, install the appropriate PyTorch build."
fi

# ── 5. Launch the Streamlit web application ───────────────────────────────
echo ""
info "Launching Streamlit web application…"
echo ""

streamlit run "${SCRIPT_DIR}/webui/app.py"

read -p "Press return (ENTER) to continue and exit this script... " CONTINUE_KEY
