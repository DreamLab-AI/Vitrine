#!/usr/bin/env bash
# Launch the canonical "owner" ComfyUI (ADR-014 endpoint).
#
# Runs ~/comfyui-api-data/ComfyUI (with its custom nodes: TRELLIS2, Hunyuan3D-2.1,
# SAM3D) against the host model-staging tree (~/comfyui-models-staging), on GPU0,
# published on host :8200. The gaussian-toolkit image is reused purely as a
# CUDA + torch 2.11 + ComfyUI-deps runtime.
#
# IMPORTANT: --entrypoint override is REQUIRED. The gaussian-toolkit image's
# entrypoint is supervisord (which would start the image's OWN bare /opt/comfyui
# on GPU1 and ignore this command). Overriding it runs the owner's /comfyui/main.py.
#
# Verified: extra search paths load (/staging/{diffusion_models,vae,...}) and
# FLUX.2 (flux2_dev_fp8mixed.safetensors) is visible in UNETLoader.
set -euo pipefail
COMFY_DIR="${COMFYUI_DIR:-$HOME/comfyui-api-data/ComfyUI}"
STAGING="${COMFYUI_STAGING:-$HOME/comfyui-models-staging}"
PORT="${COMFYUI_HOST_PORT:-8200}"
GPU="${COMFYUI_GPU:-0}"
IMAGE="${COMFYUI_IMAGE:-gaussian-toolkit:latest}"

docker rm -f vitrine-comfyui 2>/dev/null || true
docker run -d --name vitrine-comfyui --runtime nvidia --user 0:0 \
  -e CUDA_VISIBLE_DEVICES="$GPU" -p "${PORT}:8188" \
  -v "$COMFY_DIR":/comfyui -v "$STAGING":/staging \
  -w /comfyui --entrypoint /usr/bin/python3.12 \
  "$IMAGE" \
  main.py --listen 0.0.0.0 --port 8188 --cuda-device "$GPU" \
    --extra-model-paths-config /comfyui/extra_model_paths.yaml --preview-method auto

# Join the v2g-net so the gaussian-toolkit pipeline can reach it by name as
# http://vitrine-comfyui:8188 (set V2G_COMFYUI_URL accordingly).
docker network create v2g-net >/dev/null 2>&1 || true
docker network connect v2g-net vitrine-comfyui >/dev/null 2>&1 || true

echo "vitrine-comfyui launched on host :${PORT} (GPU${GPU}); owner ComfyUI + /staging model tree."
echo "Pipeline endpoint: V2G_COMFYUI_URL=http://vitrine-comfyui:8188 (over v2g-net)"
