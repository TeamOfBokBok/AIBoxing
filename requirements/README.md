# Dependency profiles

Install the common packages and one hardware profile from the project root:

```powershell
python -m pip install -r requirements/base.txt
python -m pip install -r requirements/cpu.txt
```

Use `cuda11.txt` or `cuda12.txt` for NVIDIA GPUs. Use `amd.txt` for ONNX
inference through DirectML on Windows; training with an AMD or Intel GPU also
requires the compatible `torch-directml` package for the selected Python
version. If no accelerator package is installed, the CPU profile is used.

The PyTorch package is intentionally kept in each hardware profile because
CUDA, ROCm, DirectML, and CPU wheels are not interchangeable.