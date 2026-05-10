# libtensorflow v2.13 – Haswell-optimized build (WSL Debian)

This repository provides a precompiled **TensorFlow v2.13** C API shared library (`libtensorflow.so`) plus C++ headers (optional) built specifically for **Intel Haswell** microarchitecture.  
The build was performed on **WSL2 (Windows Subsystem for Linux)** running **Debian** (Bookworm), with optimizations for AVX2, FMA, and MKL.

> 💡 **Intended use**: Lightweight inference in C/C++ applications without the full Python runtime.  

## 🖥️ Build environment

| Component | Detail |
|-----------|--------|
| Host OS | Windows 11 + WSL2 |
| WSL distro | Debian 12 (Bookworm) |
| Architecture | x86_64 (Haswell target) |
| Compiler | GCC 12.2 |
| Bazel version | 6.1.0 |
| CPU optimization | `-march=haswell -mtune=haswell -O3` |
| BLAS / math | Intel MKL (oneDNN with AVX2) |
| Distributed / GPU | OFF (CPU‑only) |
