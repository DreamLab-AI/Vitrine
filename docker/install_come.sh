#!/bin/bash
# install_come.sh — Install CoMe (Confidence-based Mesh Extraction) inside the
# come sidecar container.  Invoked by docker/Dockerfile.come when INSTALL_COME=1.
#
# LICENSING (verified 2026-05-26 against github.com/r4dl/CoMe): CoMe ships
# LICENSE.md = the Inria/MPII "Gaussian-Splatting License" — NON-COMMERCIAL
# research use only (plus NOTICE.md covering SOF and StopThePop). Do NOT use
# its outputs in commercial products/distribution without a separate agreement
# with Inria. Build is gated behind INSTALL_COME=0 by default. See ADR-004.
#
# Installs the upstream-tested conda env (environment.yml: python 3.10,
# pytorch>=2.1, pytorch-cuda=12.1, cgal, gmp, dacite) and builds the four
# vendored CUDA submodules. The come env's python is placed on PATH by the
# Dockerfile so `docker exec come python3 ...` resolves to it.
set -e

echo "=== Installing CoMe (Confidence-based Mesh Extraction) ==="
export PATH="/opt/miniconda/bin:$PATH"

# ---------------------------------------------------------------------------
# Miniconda (if absent). TOS accept required since conda 25.x.
# ---------------------------------------------------------------------------
if ! command -v conda &> /dev/null; then
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /opt/miniconda
    rm /tmp/miniconda.sh
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# Clone CoMe. Submodules are vendored in-tree (no .gitmodules); --recursive is
# harmless. Force HTTPS in case any nested ref uses git@.
# ---------------------------------------------------------------------------
git config --global url."https://github.com/".insteadOf "git@github.com:"
if [ ! -d /opt/come ]; then
    echo "Cloning CoMe..."
    git clone --recursive https://github.com/r4dl/CoMe.git /opt/come
fi
cd /opt/come

# ---------------------------------------------------------------------------
# Create the upstream-tested conda env. environment.yml pins python=3.10,
# pytorch>=2.1, pytorch-cuda=12.1, cgal, gmp, dacite, open3d, trimesh, etc.
# ---------------------------------------------------------------------------
echo "Creating conda env 'come' from environment.yml..."
conda env create -f environment.yml || conda env update -n come -f environment.yml

RUN_COME="conda run -n come"
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="8.9"   # RTX 6000 Ada / RTX 4090 = sm_89

# ---------------------------------------------------------------------------
# Build the four vendored CUDA submodules into the come env.
# Verified set (README setup): simple-knn, diff-gaussian-rasterization,
# decoupled-fused-ssim. tetra-triangulation (Delaunay, CGAL+GMP from conda) is
# required by extract_mesh_tets.py (real-world marching tetrahedra).
# ---------------------------------------------------------------------------
echo "Building simple-knn..."
$RUN_COME pip install --no-build-isolation ./submodules/simple-knn

echo "Building diff-gaussian-rasterization..."
$RUN_COME pip install --no-build-isolation ./submodules/diff-gaussian-rasterization

echo "Building decoupled-fused-ssim..."
$RUN_COME pip install --no-build-isolation ./submodules/decoupled-fused-ssim

# tetra-triangulation: cmake build using conda-provided CGAL/GMP. Best-effort —
# on failure extract_mesh_tets.py is unavailable but extract_mesh_tsdf.py and
# training still work.
echo "Building tetra-triangulation (non-fatal on failure)..."
CONDA_ENV_PREFIX="$(conda env list | awk '/come/{print $NF}' | head -1)"
( cd submodules/tetra-triangulation \
    && $RUN_COME cmake . -DCMAKE_PREFIX_PATH="${CONDA_ENV_PREFIX}" \
    && $RUN_COME make -j"$(nproc)" \
    && $RUN_COME pip install -e . ) \
    || echo "WARNING: tetra-triangulation build failed — extract_mesh_tets.py (real-world tets) unavailable; TSDF path still works"

# ---------------------------------------------------------------------------
# Verify core extensions in the come env.
# ---------------------------------------------------------------------------
echo "Verifying CoMe installation..."
$RUN_COME python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
$RUN_COME python -c "import diff_gaussian_rasterization, simple_knn; print('rasterizer + simple-knn OK')"
$RUN_COME python -c "import dacite, plyfile, trimesh, open3d; print('config + IO deps OK')"

echo "=== CoMe installation complete (conda env: come, NON-COMMERCIAL licence) ==="
echo "Train:   docker exec come python3 /opt/come/train.py --splatting_config configs/hierarchical.json -s <DATASET> -m <OUT>"
echo "Extract: docker exec come python3 /opt/come/extract_mesh_tets.py -m <OUT>   # real-world"
