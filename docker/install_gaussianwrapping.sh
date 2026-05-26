#!/bin/bash
# install_gaussianwrapping.sh — Install GaussianWrapping into the MILo sidecar
# (Ubuntu 22.04, CUDA 11.8, Python 3.9).  Invoked by docker/Dockerfile.milo
# when INSTALL_GAUSSIANWRAPPING=1.
#
# LICENSING NOTE: GaussianWrapping has no LICENSE file as of 2026-05-26.
# This is the same status as CoMe and must be treated identically — do not
# include in commercial distribution images until a permissive licence is
# reviewed. See ADR-005.
#
# FAILURE POLICY: Build failures are non-fatal (|| true at top level).
# The MILo sidecar remains fully functional if GaussianWrapping fails to build.
# is_gaussianwrapping_available() in gaussianwrapping_extractor.py checks for
# /opt/gaussianwrapping before invoking this backend.

set -e  # exit on error within this script; caller wraps with || true

echo "=== Installing GaussianWrapping (thin-structure mesh extraction) ==="

# Guard: skip silently if the arg/env gate was not set (defensive check;
# Dockerfile.milo already gates the COPY+RUN pair behind INSTALL_GAUSSIANWRAPPING).
if [ "${INSTALL_GAUSSIANWRAPPING:-0}" != "1" ]; then
    echo "INSTALL_GAUSSIANWRAPPING not set to 1 — skipping GaussianWrapping install."
    exit 0
fi

# ---------------------------------------------------------------------------
# HTTPS for git submodules (SSH not available in Docker build)
# ---------------------------------------------------------------------------
git config --global url."https://github.com/".insteadOf "git@github.com:"

# ---------------------------------------------------------------------------
# Clone GaussianWrapping
# ---------------------------------------------------------------------------
echo "Cloning GaussianWrapping..."
if [ ! -d "/opt/gaussianwrapping" ]; then
    git clone --recursive https://github.com/diego1401/GaussianWrapping.git /opt/gaussianwrapping
fi

# ---------------------------------------------------------------------------
# Build with CUDA 11.8 (matches the MILo sidecar CUDA version)
# GaussianWrapping ships an install.py that compiles its CUDA extensions and
# installs them into the active Python environment.
# ---------------------------------------------------------------------------
echo "Building GaussianWrapping CUDA extensions (CUDA 11.8)..."

export CUDA_HOME=/usr/local/cuda
# If CUDA 11.8 path exists use it specifically, else fall back to system CUDA
if [ -d "/usr/local/cuda-11.8" ]; then
    export CUDA_HOME=/usr/local/cuda-11.8
fi

cd /opt/gaussianwrapping
python3 install.py --cuda_version 11.8

# ---------------------------------------------------------------------------
# Verify the installation landed at /opt/gaussianwrapping
# (is_gaussianwrapping_available() in the pipeline uses this path as a probe)
# ---------------------------------------------------------------------------
echo "Verifying GaussianWrapping installation..."
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"
test -d /opt/gaussianwrapping && echo "GaussianWrapping directory confirmed at /opt/gaussianwrapping"

echo "=== GaussianWrapping installation complete ==="
echo "Invocation: docker exec milo python3 /opt/gaussianwrapping/extract.py ..."
