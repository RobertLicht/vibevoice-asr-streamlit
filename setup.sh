#!/usr/bin/env bash
# setup.sh — One-shot setup for the VibeVoice ASR project.
#
# This script installs system dependencies, creates a virtual environment,
# installs all Python packages, and optionally runs the example scripts.
#
# IMPORTANT:
#   • The virtual environment is created inside the same directory as this
#     script (venv/).
#   • No packages are installed outside the virtual environment.
#   • Running this script requires root/sudo for system package installation.
#   • Requires Ubuntu (apt) or Fedora (dnf) with an NVIDIA GPU (CUDA support),
#     an AMD GPU (ROCm support), or a CPU-only fallback.

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

# ── Determine script directory ────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"

info "Script directory: ${SCRIPT_DIR}"
info "Virtual environment will be created at: ${VENV_DIR}"

# ── 0. Detect distribution type (deb = Ubuntu/Debian, rpm = Fedora/RHEL) ─
DISTRO_TYPE="unknown"
if [ -f /etc/os-release ]; then
  # shellcheck disable=SC1091
  . /etc/os-release
  case "${ID}" in
    ubuntu|debian)
      DISTRO_TYPE="deb"
      ;;
    fedora|rhel|centos|almalinux|rocky)
      DISTRO_TYPE="rpm"
      ;;
  esac
fi
if [ "$DISTRO_TYPE" = "unknown" ]; then
  warn "Unknown distribution. Defaulting to apt-based (Debian/Ubuntu) commands."
  DISTRO_TYPE="deb"
fi
info "Detected package manager: ${DISTRO_TYPE==deb && echo 'apt' || echo 'dnf'}"

# ── 1. Install system dependency for venv support ─────────────────────────
info "Checking for python3-virtualenv…"
echo -e "\nChecking if the necessary venv module is available...\n"
if python3 -c "import venv" &>/dev/null; then
    info "Python venv module is available.\n"
else
    warn "Python venv module not available.\n"
    echo -e "Installing the missing package...\n"
    case "$DISTRO_TYPE" in
      deb)
        sudo apt update -y
        sudo apt install -y python3-virtualenv || {
          warn "Failed to install python3-virtualenv — continuing anyway."
        }
        ;;
      rpm)
        sudo dnf check-update -y 2>/dev/null
        sudo dnf install -y python3-devel pciutils 2>/dev/null || {
          warn "Failed to install python3-devel on Fedora — continuing anyway."
        }
        ;;
    esac
fi

# ── 2. Verify python3.12 is available ─────────────────────────────────────
if command -v python3.12 &>/dev/null; then
  PYTHON_CMD="python3.12"
else
  # Try python3 as a fallback (Ubuntu may have 3.12 as default python3)
  PYTHON3_VERSION=$(python3 --version 2>&1 | grep -oP 'Python 3\.(1[0-9]|2[0-9])' || echo "")
  if [ -n "$PYTHON3_VERSION" ]; then
    warn "python3.12 not found, but found Python 3.x — using python3 instead."
    PYTHON_CMD="python3"
  else
    error "python3.12 not found on PATH."
    error "Please install Python 3.12 (e.g. sudo apt install python3.12) and try again."
    read -p "Press return (ENTER) to continue and exit this script... " CONTINUE_KEY
    exit 1
  fi
fi
info "Using Python: $($PYTHON_CMD --version 2>&1)"

# ── 3. Install build tools, Python dev headers, and pciutils ─────────────
info "Checking for build tools, python3-dev, and pciutils…"
case "$DISTRO_TYPE" in
  deb)
    if ! dpkg -s build-essential &>/dev/null || ! dpkg -s pciutils &>/dev/null; then
      sudo apt update -y
      sudo apt install -y build-essential python3-dev pciutils || {
        warn "Failed to install build-essential/pciutils — continuing anyway."
      }
    fi
    ;;
  rpm)
    NEEDS_DEVTOLS=false
    NEEDS_PACKAGES=false

    # Check if Development Tools group is needed (proxy: gcc available?)
    if ! command -v gcc &>/dev/null; then
      NEEDS_DEVTOLS=true
    fi

    # Check for python3-devel and pciutils via rpm
    if ! rpm -q python3-devel &>/dev/null || ! rpm -q pciutils &>/dev/null; then
      NEEDS_PACKAGES=true
    fi

    if [ "$NEEDS_DEVTOLS" = true ] || [ "$NEEDS_PACKAGES" = true ]; then
      sudo dnf check-update -y 2>/dev/null

      # Install Development Tools group (try DNF5 syntax first, fall back to legacy)
      if [ "$NEEDS_DEVTOLS" = true ]; then
        sudo dnf group install -y development-tools 2>/dev/null || \
          sudo dnf groupinstall -y "Development Tools" 2>/dev/null || {
            warn "Failed to install Development Tools group — continuing anyway."
          }
      fi

      # Install individual packages if needed
      if [ "$NEEDS_PACKAGES" = true ]; then
        sudo dnf install -y python3-devel pciutils 2>/dev/null || {
          warn "Failed to install python3-devel/pciutils — continuing anyway."
        }
      fi
    fi
    ;;
esac
info "Build tools, Python dev headers, and pciutils OK."

# ── 4. Detect AMD GPU (ROCm) ────────────────────────────────────────────
GPU_BACKEND="none"  # will be "rocm", "cuda", or "cpu"
ROCm_WHEEL="rocm7.2"

if command -v lspci &>/dev/null; then
  AMD_GPU_LINE=$(lspci 2>/dev/null | grep -iE 'vga|3d|display' | grep -i -E 'amd|ati|radeon' || echo "")
  if [ -n "$AMD_GPU_LINE" ]; then
    AMD_GPU_NAME=$(echo "$AMD_GPU_LINE" | sed 's/.*: *//' || echo "unknown")
    info "AMD GPU detected: ${AMD_GPU_NAME}"

    # Check if ROCm is already installed
    if command -v rocminfo &>/dev/null; then
      ROCM_VERSION=$(rocminfo 2>/dev/null | grep "ROCk module" | head -1 || echo "unknown")
      info "ROCm already installed: ${ROCM_VERSION}"
    elif [ -f /opt/rocm/bin/rocminfo ]; then
      info "ROCm installed at /opt/rocm (found rocminfo)."
    else
      warn "ROCm runtime not found on this system."
      warn "For AMD GPU acceleration, install ROCm using amdgpu-install."
      warn "Without ROCm, PyTorch will fall back to CPU."
    fi

    # Decide: if ROCm is present and functional, use it
    if command -v rocminfo &>/dev/null && rocminfo &>/dev/null; then
      info "ROCm is functional. Setting GPU backend to ROCm (PyTorch wheel: ${ROCm_WHEEL})."
      GPU_BACKEND="rocm"
    else
      warn "ROCm not functional — will check for NVIDIA GPU next."
    fi
  fi
fi

# ── 5. Detect NVIDIA GPU (CUDA) ─────────────────────────────────────────
if [ "$GPU_BACKEND" = "none" ]; then
  NVIDIA_GPU_NAME=""
  CUDA_WHEEL=""
  if command -v nvidia-smi &>/dev/null; then
    NVIDIA_GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "")
    CUDA_DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "0")
    info "NVIDIA GPU detected: ${NVIDIA_GPU_NAME:-unknown}"
    info "NVIDIA driver version: ${CUDA_DRIVER_VERSION}"

    # Auto-select the best CUDA wheel based on driver support.
    # CUDA 12.8 requires driver >= 550.54
    # CUDA 12.6 requires driver >= 525.60
    # CUDA 11.8 requires driver >= 450.80
    CUDA_MAJOR=$(echo "$CUDA_DRIVER_VERSION" | grep -oP '^[0-9]+' || echo "0")
    if [ "$CUDA_MAJOR" -ge 550 ]; then
      CUDA_WHEEL="cu128"
    elif [ "$CUDA_MAJOR" -ge 525 ]; then
      CUDA_WHEEL="cu126"
    elif [ "$CUDA_MAJOR" -ge 450 ]; then
      CUDA_WHEEL="cu118"
    else
      warn "NVIDIA driver is too old for CUDA 11.8+. Falling back to CPU PyTorch."
      CUDA_WHEEL="cpu"
    fi
    info "Selected CUDA wheel: ${CUDA_WHEEL}"
    GPU_BACKEND="cuda"
  else
    warn "nvidia-smi not found — no NVIDIA GPU detected."
    GPU_BACKEND="cpu"
  fi
fi

# ── 6. Final GPU backend summary ────────────────────────────────────────
if [ "$GPU_BACKEND" = "rocm" ]; then
  info "GPU backend: AMD ROCm (${ROCm_WHEEL})"
elif [ "$GPU_BACKEND" = "cuda" ]; then
  info "GPU backend: NVIDIA CUDA (${CUDA_WHEEL})"
else
  warn "No GPU backend detected. Falling back to CPU-only PyTorch."
fi

# ── 7. Create the virtual environment ────────────────────────────────────
if [ -d "${VENV_DIR}" ]; then
  warn "Virtual environment already exists at ${VENV_DIR} — skipping creation."
else
  info "Creating virtual environment at ${VENV_DIR}…"
  $PYTHON_CMD -m venv "${VENV_DIR}" || {
    error "Failed to create the virtual environment. Aborting."
    read -p "Press return (ENTER) to continue and exit this script... " CONTINUE_KEY
    exit 1
  }
  info "Virtual environment created."
fi

# ── 8. Activate the virtual environment ───────────────────────────────────
info "Activating virtual environment…"
# shellcheck disable=SC1091
. "${VENV_DIR}/bin/activate"

# Verify we are inside the venv
if [ "${VIRTUAL_ENV:-}" = "" ]; then
  error "Virtual environment activation failed. Aborting."
  read -p "Press return (ENTER) to continue and exit this script... " CONTINUE_KEY
  exit 1
fi
info "Active venv: ${VIRTUAL_ENV}"
info "Python: $(python --version 2>&1)"

# ── 9. Upgrade pip ───────────────────────────────────────────────────────
info "Upgrading pip…"
pip install --upgrade pip || {
  error "Failed to upgrade pip. Aborting."
  read -p "Press return (ENTER) to continue and exit this script... " CONTINUE_KEY
  exit 1
}
info "pip upgraded to $(pip --version 2>&1)."

# ── 10. Install PyTorch (ROCm, CUDA, or CPU fallback) ──────────────────
if [ "$GPU_BACKEND" = "rocm" ]; then
  info "Installing PyTorch with ROCm ${ROCm_WHEEL} support…"
  pip install torch torchvision torchaudio \
    --index-url "https://download.pytorch.org/whl/${ROCm_WHEEL}" || {
    error "Failed to install PyTorch with ROCm. Aborting."
    deactivate 2>/dev/null
    exit 1
  }
elif [ "$GPU_BACKEND" = "cuda" ]; then
  info "Installing PyTorch with CUDA ${CUDA_WHEEL} support…"
  pip install torch torchvision torchaudio \
    --index-url "https://download.pytorch.org/whl/${CUDA_WHEEL}" || {
    error "Failed to install PyTorch with CUDA. Aborting."
    deactivate 2>/dev/null
    exit 1
  }
else
  info "Installing PyTorch CPU-only build…"
  pip install torch torchvision torchaudio || {
    error "Failed to install PyTorch. Aborting."
    deactivate 2>/dev/null
    exit 1
  }
fi
info "PyTorch installed."

# ── 11. Install remaining Python packages ────────────────────────────────
info "Installing transformers, accelerate, librosa…"
pip install transformers accelerate librosa streamlit Spire.Doc.Free torchcodec || {
  error "Failed to install required packages. Aborting."
  deactivate 2>/dev/null
  read -p "Press return (ENTER) to continue and exit this script... " CONTINUE_KEY
  exit 1
}
info "All Python packages installed."

# ── 12. Verification ─────────────────────────────────────────────────────
echo ""
info "===== Environment Verification ====="
python --version
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import torch; print(f'CUDA/ROCm available: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  if python -c "import torch; print(torch.version.hip or '')" 2>/dev/null | grep -q '^\d'; then
    HIP_VERSION=$(python -c "import torch; print(torch.version.hip)" 2>/dev/null || echo "unknown")
    GPU_DEVICE=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "unknown")
    info "ROCm/HIP ${HIP_VERSION} available — GPU: ${GPU_DEVICE}"
  else
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "unknown")
    GPU_DEVICE=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "unknown")
    info "CUDA ${CUDA_VERSION} available — GPU: ${GPU_DEVICE}"
  fi
else
  warn "CUDA/ROCm not available — running CPU-only."
fi
python -c "import transformers; print(f'transformers {transformers.__version__}')"
python -c "import accelerate; print(f'accelerate {accelerate.__version__}')"
python -c "import librosa; print(f'librosa {librosa.__version__}')"
echo "====================================="
echo ""

# ── 13. Run example scripts (non-fatal — they may require audio input) ───
info "Setup complete. The virtual environment is still active."
info "When you are ready, you can run the example scripts:"
info "  python transcribe.py"
info "  python vibevoice.py"
info ""
info "Or deactivate the environment with:  deactivate"
read -p "Press return (ENTER) to continue and exit this script... " CONTINUE_KEY
