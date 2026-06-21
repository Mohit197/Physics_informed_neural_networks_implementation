# Physics-Informed Heat Sink Simulation Using NVIDIA Modulus-Sym

This repository contains a physics-informed machine learning / PINN heat-sink example using the older NVIDIA **Modulus-Sym** API.

The main script is:

```bash
heat_sink_main.py
```

and the configuration file is:

```bash
config.yaml
```

> Important: this code uses the older import style:
>
> ```python
> import modulus.sym
> from modulus.sym.solver import Solver
> from modulus.sym.node import Node
> ```
>
> Therefore, the correct package stack is **NVIDIA Modulus-Sym**, not the newer `physicsnemo.sym` API.

---

## 1. Recommended System

This setup was tested using:

```text
Windows + WSL2
Ubuntu 22.04
Python 3.10
NVIDIA GPU
PyTorch 2.5.1+cu121
CUDA available: True
GPU example: NVIDIA GeForce RTX 4050 Laptop GPU
```

Using Ubuntu 22.04 is strongly recommended because newer Ubuntu releases may default to Python 3.14, which is too new for many PyTorch / Modulus-Sym dependencies.

---

## 2. Open Ubuntu 22.04 WSL

From Windows PowerShell:

```powershell
wsl -d Ubuntu-22.04
```

Check Ubuntu and GPU access:

```bash
lsb_release -a
python3 --version
nvidia-smi
```

Expected:

```text
Ubuntu 22.04
Python 3.10.x
NVIDIA GPU visible through nvidia-smi
```

---

## 3. Project Folder Layout

Recommended folder structure:

```text
~/projects/physicsnemo_heat_sink/
├── heat_sink_main.py
├── config.yaml
├── outputs/
├── checkpoints/
└── requirements_physicsnemo_modulus.txt
```

Create the project directory:

```bash
mkdir -p ~/projects/physicsnemo_heat_sink
cd ~/projects/physicsnemo_heat_sink
```

If copying files from Windows, the WSL folder can be opened in Windows File Explorer using:

```text
\\wsl$\Ubuntu-22.04\home\mohit\projects\physicsnemo_heat_sink
```

Alternatively, from the Ubuntu terminal:

```bash
cd ~/projects/physicsnemo_heat_sink
explorer.exe .
```

Copy the following files into the project folder:

```text
heat_sink_main.py
config.yaml
```

Because the script uses:

```python
@modulus.sym.main(config_path="./", config_name="config.yaml")
```

the `config.yaml` file must be in the same directory as `heat_sink_main.py`.

---

## 4. Create and Activate Python Environment

Create a reusable virtual environment:

```bash
python3.10 -m venv ~/venvs/physicsnemo
source ~/venvs/physicsnemo/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Confirm Python version:

```bash
python --version
```

Expected:

```text
Python 3.10.x
```

---

## 5. Install PyTorch with CUDA

Install PyTorch CUDA 12.1 wheels:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Test CUDA:

```bash
python - << 'EOF'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
EOF
```

Expected output:

```text
CUDA available: True
GPU: <your NVIDIA GPU>
```

---

## 6. Install CUDA Toolkit Inside WSL

PyTorch can use CUDA runtime libraries from its wheel, but Modulus-Sym needs CUDA development tools such as `nvcc` during installation.

Install CUDA Toolkit 12.1 inside WSL:

```bash
sudo apt update
sudo apt install -y wget gnupg

wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

sudo apt install -y cuda-toolkit-12-1
```

Set CUDA environment variables:

```bash
echo 'export CUDA_HOME=/usr/local/cuda-12.1' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Check:

```bash
which nvcc
nvcc --version
echo $CUDA_HOME
```

Expected:

```text
/usr/local/cuda-12.1/bin/nvcc
/usr/local/cuda-12.1
```

---

## 7. Install NVIDIA Modulus-Sym

Activate the environment:

```bash
source ~/venvs/physicsnemo/bin/activate
cd ~/projects/physicsnemo_heat_sink
```

Install build and compatibility dependencies:

```bash
pip install "Cython==0.29.28"
pip install "numpy==1.23.5"
pip install "pint==0.19.2"
```

Install common Modulus-Sym dependencies:

```bash
pip install \
  "h5py==3.8.0" \
  "symengine==0.9.2" \
  "scipy==1.10.1" \
  "matplotlib==3.7.1" \
  "pandas==1.5.3" \
  "tqdm" \
  "omegaconf" \
  "hydra-core" \
  "tensorboard" \
  "vtk" \
  "trimesh" \
  "shapely" \
  "scikit-learn" \
  "sympy" \
  "termcolor" \
  "chaospy" \
  "numpy-stl" \
  "opencv-python" \
  "psutil"
```

Install Modulus core and Modulus-Sym without triggering problematic optional dependencies such as old DALI packages:

```bash
pip install nvidia-modulus==0.2.1 --no-deps
pip install nvidia-modulus.sym --no-build-isolation --no-deps
```

Re-pin NumPy and Pint after installation, because newer dependency installs may upgrade NumPy to 2.x:

```bash
pip install --force-reinstall "numpy==1.23.5" "pint==0.19.2"
```

---

## 8. Test Installation

Run:

```bash
python - << 'EOF'
import numpy as np
import pint
import h5py
import symengine
import torch
import modulus
import modulus.sym
from modulus.sym.solver import Solver

print("NumPy:", np.__version__)
print("Pint:", pint.__version__)
print("h5py:", h5py.__version__)
print("symengine:", symengine.__version__)
print("Modulus imported successfully")
print("Modulus Sym + Solver OK")
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
EOF
```

Expected output should include:

```text
Modulus imported successfully
Modulus Sym + Solver OK
CUDA: True
```

---

## 9. Run the Heat-Sink Model

From the project folder:

```bash
cd ~/projects/physicsnemo_heat_sink
source ~/venvs/physicsnemo/bin/activate
python heat_sink_main.py
```

---

## 10. Run a Short Test Instead of Full Training

The default `config.yaml` may contain a very large training step count, for example:

```yaml
training:
  rec_validation_freq: 1000
  rec_inference_freq: 1000
  rec_monitor_freq: 1000
  rec_constraint_freq: 2000
  max_steps: 5000000
```

For a quick test, override the step count from the command line:

```bash
python heat_sink_main.py training.max_steps=100
```

For a more meaningful intermediate test:

```bash
python heat_sink_main.py \
  training.max_steps=20000 \
  training.rec_validation_freq=1000 \
  training.rec_inference_freq=1000 \
  training.rec_monitor_freq=1000 \
  training.rec_constraint_freq=2000
```

This runs for 20,000 steps while keeping logging / validation / monitor outputs active.

To interrupt training at any time:

```text
Ctrl + C
```

---

## 11. Monitor GPU Usage

Open a second Ubuntu terminal and run:

```bash
watch -n 1 nvidia-smi
```

You should see the Python process using GPU memory during training.

---

## 12. Save the Working Environment

After the environment is confirmed working, save all package versions:

```bash
cd ~/projects/physicsnemo_heat_sink
source ~/venvs/physicsnemo/bin/activate
pip freeze > requirements_physicsnemo_modulus.txt
```

This file can be committed to the repository for reproducibility.

To recreate the environment later:

```bash
python3.10 -m venv ~/venvs/physicsnemo_new
source ~/venvs/physicsnemo_new/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements_physicsnemo_modulus.txt
```

Note: system-level CUDA Toolkit installation and `CUDA_HOME` setup may still need to be done separately on a new machine.

---

## 13. Optional: Add a Convenience Alias

To make future activation easier:

```bash
echo "alias activate_physicsnemo='source ~/venvs/physicsnemo/bin/activate'" >> ~/.bashrc
source ~/.bashrc
```

Then activate the environment using:

```bash
activate_physicsnemo
```

---

## 14. Common Troubleshooting

### Error: `ModuleNotFoundError: No module named 'modulus'`

Modulus-Sym is not installed in the active environment.

Fix:

```bash
source ~/venvs/physicsnemo/bin/activate
pip install nvidia-modulus==0.2.1 --no-deps
pip install nvidia-modulus.sym --no-build-isolation --no-deps
```

---

### Error: `CUDA_HOME environment variable is not set`

CUDA Toolkit is missing or `CUDA_HOME` is not configured.

Fix:

```bash
echo 'export CUDA_HOME=/usr/local/cuda-12.1' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Then check:

```bash
which nvcc
echo $CUDA_HOME
```

---

### Error: `module 'numpy' has no attribute 'cumproduct'`

This usually means NumPy 2.x was installed, which is incompatible with the older Pint / Modulus-Sym stack.

Fix:

```bash
pip uninstall -y numpy pint
pip install "numpy==1.23.5" "pint==0.19.2"
```

---

### Error: `ModuleNotFoundError: No module named 'h5py'`

Fix:

```bash
pip install "h5py==3.8.0"
```

---

### Error: `ModuleNotFoundError: No module named 'symengine'`

Fix:

```bash
pip install "symengine==0.9.2"
```

---

### Error involving `nvidia-dali-cuda110` or `wheel_stub.buildapi`

Avoid full automatic dependency resolution for old Modulus-Sym.

Use:

```bash
pip install nvidia-modulus==0.2.1 --no-deps
pip install nvidia-modulus.sym --no-build-isolation --no-deps
```

and install only the required dependencies manually.

---

## 15. Notes on PhysicsNeMo vs Modulus

NVIDIA Modulus has evolved into PhysicsNeMo, but this project uses the older Modulus-Sym API.

This project is compatible with scripts that import:

```python
import modulus.sym
from modulus.sym.solver import Solver
```

If updating the code to newer PhysicsNeMo-Sym in the future, imports and some APIs may need to be changed from:

```python
modulus.sym
```

to:

```python
physicsnemo.sym
```

---

## 16. Minimal Run Checklist

For a user who already has WSL, CUDA Toolkit, and the environment configured:

```bash
wsl -d Ubuntu-22.04

cd ~/projects/physicsnemo_heat_sink
source ~/venvs/physicsnemo/bin/activate

python - << 'EOF'
import torch
import modulus.sym
print("CUDA:", torch.cuda.is_available())
print("Modulus-Sym OK")
EOF

python heat_sink_main.py training.max_steps=20000
```

