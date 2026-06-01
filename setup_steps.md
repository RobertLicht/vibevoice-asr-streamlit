# Function to check if the relevant package is available on the system
fnc_chk_pkg_python_virtualenv() {
  PACKAGE_NAME="python-virtualenv"
  echo -e "\\nChecking if the necessary package '$PACKAGE_NAME' is installed...\\n"
  if which "$PACKAGE_NAME" &>/dev/null; then
      echo -e "🟢 $PACKAGE_NAME is installed.\\n"
  else
      echo -e "🟡 Warning: $PACKAGE_NAME is not installed.\\n"
      echo -e "Installing the missing package...\\n"
      sudo dnf -y update
      sudo dnf -y install $PACKAGE_NAME
  fi
}

# Create virtual python environment with a specific version of python
python -m virtualenv -p python3.12 venv

# Activate the virtual python environment
source venv/bin/activate

# Check if the virtual python environment is active
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Virtual environment is active: $VIRTUAL_ENV"
else
    echo "No virtual environment is active."
fi   

# Upgrade pip
pip install --upgrade pip

# Check if and which GPU is available on the system
lspci | grep -i vga

# Install the version of PyTorch which fits to the available hardware on the system
#    AMD
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2
#    NVIDIA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu132
#    CPU (fallback)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install additional necessary python packages
pip install transformers accelerate librosa

# Check if the GPU can be utilized
python torch_cuda_avail.py

# Execute a basic script which uses microsoft/VibeVoice-ASR-HF
python vibevoice_asr.py
