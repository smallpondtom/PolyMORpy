# PolyMORpy: Benchmark PDEs for Model Reduction in Python

A research repository implementing high-performance solvers for benchmark partial differential equations (PDEs) using JAX, with a focus on model reduction, reduced order models (ROMs), and surrogate modeling. All code uses [Pixi](https://pixi.sh) for reproducible dependency management.

## 📋 Table of Contents

- [Overview](#overview)
- [Getting Started with Pixi](#getting-started-with-pixi)
- [System Requirements](#system-requirements)

## Overview

PolyMORpy contains efficient implementations of benchmark PDEs commonly used in model reduction research:

| PDE | Location | Status | Features |
|-----|----------|--------|----------|
| **Heat** | `Heat1D/` | ✅ Complete | Classic parabolic PDE |
| **Viscous Burgers** | `ViscousBurgers1D/` | ✅ Complete | Nonlinear advection-diffusion |
| **Allen-Cahn** | `AllenCahn1D/` | ✅ Complete | Phase field dynamics, cubic nonlinearity, SICN/CNAB schemes |
| **Kawahara** | `Kawahara/` | ✅ Complete | Dispersive dynamics, multiple conservation formulations, parameter updates |

All solvers are:

- ⚡ **GPU-accelerated** via JAX with JIT compilation
- 📊 **Efficient** using unique Kronecker products for nonlinear terms
- 🔍 **Research-grade** with proper numerical schemes (CNAB, SICN, etc.)
- 📝 **Well-documented** with mathematical background

---

## Getting Started with Pixi

### What is Pixi?

[Pixi](https://pixi.sh) is a fast, reliable package manager for Python and other languages that uses conda-compatible packages. It provides:

- **Lock files** for reproducible environments across machines
- **Fast dependency resolution** and installation
- **Cross-platform support** (Linux, macOS, Windows)
- **Environment isolation** without needing separate virtual environments
- **Task management** for common workflows

### Prerequisites

Before using this repository, ensure you have [Pixi installed](https://pixi.sh/latest/#installation). On macOS:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

For other platforms, see the [Pixi installation guide](https://pixi.sh/latest/#installation).

Verify installation:

```bash
pixi --version
```

### Initial Setup

1. **Clone and navigate to the repository:**

   ```bash
   git clone https://github.com/smallpondtom/PolyMORpy.git
   cd PolyMORpy
   ```

2. **Verify the project configuration:**

   ```bash
   pixi info
   ```

   This displays your workspace setup, channels, and platforms.

3. **Initialize the default environment:**

   ```bash
   pixi install
   ```

   This command:
   - Reads `pyproject.toml` and `pixi.lock`
   - Downloads all dependencies
   - Creates an isolated environment in `.pixi/envs/default/`
   - Takes ~5-10 minutes on first run

### Running Python with Pixi

Once the environment is set up, run Python code using:

```bash
# Execute a Python script
pixi run python AllenCahn1D/allen_cahn_jax.py

# Launch interactive Python shell
pixi run python

# Run Jupyter notebooks
pixi run jupyter lab

# Execute a specific module or script
pixi run python -c "import jax; print(jax.devices())"
```

### Project Dependencies

The project uses these core packages (see `pyproject.toml`):

```toml
jax = ">=0.7.2,<0.8"           # Numerical computing framework
jaxlib = ">=0.7.2,<0.8"        # JAX compiled extensions
jupyter = ">=1.1.1,<2"         # Interactive notebooks
matplotlib = ">=3.10.7,<4"     # Plotting
```

Channels used:

- `conda-forge` - Primary package source
- `nvidia` - GPU/CUDA support (optional)

Supported platforms:

- `linux-64` - Linux x86-64
- `osx-arm64` - macOS Apple Silicon (M1, M2, M3, etc.)

### Environment Management

**List available environments:**

```bash
pixi env list
```

**Create a custom environment (e.g., for CUDA):**

```bash
pixi install --environment cuda
```

This uses the `cuda` environment configuration defined in `pyproject.toml`.

**View environment details:**

```bash
pixi env info
```

**Remove an environment:**

```bash
pixi env remove <env-name>
```

---

## System Requirements

### Minimum Requirements

- **CPU:** Any modern processor (x86-64 or ARM64)
- **RAM:** 4 GB (8 GB recommended)
- **Disk:** 2-3 GB for environment and dependencies
- **OS:** macOS 10.13+, Ubuntu 18.04+, Windows 10+

### Recommended Setup for Development

- **CPU:** Multi-core processor (4+ cores)
- **RAM:** 16 GB
- **Disk:** 5 GB available
- **GPU:** (Optional) NVIDIA GPU with CUDA 12 support for acceleration

### macOS Apple Silicon Note

This project is optimized for Apple Silicon (M1, M2, M3) through the `osx-arm64` platform. Pixi will automatically select the correct binaries.

### Adding New Code

1. Create a new folder (e.g., `Heat1D/`)
2. Add your solver implementation
3. Write a `README.md` documenting your solver
4. Test with `pixi run python your_solver.py`

---

### Getting Help

- Check solver-specific README files in each subdirectory
- Review JAX documentation: [https://jax.readthedocs.io/](https://jax.readthedocs.io/)
- Pixi docs: [https://pixi.sh](https://pixi.sh)

---

## License

This project is intended for research and educational purposes.

---

## Contributing

When contributing:

1. Use `pixi add <package>` to add dependencies (updates `pyproject.toml`)
2. Test with `pixi run python`
3. Ensure code runs on both `linux-64` and `osx-arm64`
4. Update relevant README files

---

**Last Updated:** October 2025
