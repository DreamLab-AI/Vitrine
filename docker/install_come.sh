#!/bin/bash
# install_come.sh — Install CoMe (Confidence-based Mesh Extraction) inside the
# come sidecar container.  Invoked by docker/Dockerfile.come when INSTALL_COME=1.
#
# LICENSING NOTE: CoMe has no LICENSE file as of 2026-05-26 (SPDX: NOASSERTION).
# Do not include in commercial images until a permissive licence is reviewed.
# See ADR-004 for the build-arg gate that prevents accidental inclusion.
set -e

echo "=== Installing CoMe (Confidence-based Mesh Extraction) ==="

# ---------------------------------------------------------------------------
# Optional conda setup (mirrors install_milo.sh pattern)
# CoMe targets system Python 3.10; conda is installed only if it is absent,
# and the TOS is accepted to satisfy conda 25.x requirements.
# ---------------------------------------------------------------------------
if ! command -v conda &> /dev/null; then
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /opt/miniconda
    rm /tmp/miniconda.sh
    export PATH="/opt/miniconda/bin:$PATH"
    # Accept TOS (required since conda 25.x)
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
fi

export PATH="/opt/miniconda/bin:$PATH"

# ---------------------------------------------------------------------------
# Clone CoMe — force HTTPS for submodules (SSH not available in Docker build)
# ---------------------------------------------------------------------------
echo "Configuring HTTPS for git submodules..."
git config --global url."https://github.com/".insteadOf "git@github.com:"

echo "Cloning CoMe..."
if [ ! -d "/opt/come" ]; then
    git clone --recursive https://github.com/r4dl/CoMe.git /opt/come
fi

# ---------------------------------------------------------------------------
# Build CUDA submodules
# CoMe uses the same diff-gaussian-rasterization and simple-knn extensions as
# standard 3DGS, plus tetra-triangulation for marching tetrahedra extraction.
# ---------------------------------------------------------------------------
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="8.9"

echo "Building diff-gaussian-rasterization..."
cd /opt/come
pip3 install --no-build-isolation submodules/diff-gaussian-rasterization/

echo "Building simple-knn..."
pip3 install --no-build-isolation submodules/simple-knn/

# tetra_triangulation (Delaunay, requires CGAL + pybind11) — non-fatal
# Build failure falls back gracefully; mesh extraction uses a fallback path.
echo "Building tetra_triangulation (non-fatal on failure)..."
pip3 install pybind11
cd /opt/come/submodules/tetra_triangulation \
    && cmake . \
        -DPYTHON_EXECUTABLE=/usr/bin/python3 \
        -DCMAKE_CXX_FLAGS="-I/usr/local/cuda/include" \
        -Dpybind11_DIR="$(python3 -m pybind11 --cmakedir)" \
    && make -j"$(nproc)" \
    && pip3 install -e . \
    || echo "WARNING: tetra_triangulation build failed — marching-tetrahedra extraction unavailable"

cd /opt/come

# ---------------------------------------------------------------------------
# Python requirements
# ---------------------------------------------------------------------------
echo "Installing CoMe Python requirements..."
if [ -f requirements.txt ]; then
    pip3 install -r requirements.txt
fi

pip3 install \
    open3d==0.19.0 trimesh scikit-image opencv-python plyfile tqdm einops

# PLY-to-GLB export capability
pip3 install pygltflib

# ---------------------------------------------------------------------------
# Verify core extensions
# ---------------------------------------------------------------------------
echo "Verifying CoMe installation..."
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"
python3 -c "import diff_gaussian_rasterization; print('diff-gaussian-rasterization OK')"
python3 -c "import simple_knn; print('simple-knn OK')"

echo "=== CoMe installation complete ==="
echo "Usage: docker exec come python3 /opt/come/extract.py -s <COLMAP_DATASET> -m <OUTPUT_DIR>"
