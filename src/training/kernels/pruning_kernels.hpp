/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <cstdint>

namespace lfs::training::pruning {

    /**
     * Fused dead mask computation.
     *
     * Dead Gaussians are those with:
     *   - opacity <= min_opacity OR
     *   - ||rotation||^2 < 1e-8 (near-zero rotation magnitude)
     *
     * @param opacities [N] - Opacity values
     * @param rotations [N, 4] - Quaternion rotations
     * @param dead_mask [N] - Output: boolean mask (uint8_t)
     * @param N - Number of Gaussians
     * @param min_opacity - Minimum valid opacity threshold
     * @param stream - CUDA stream
     */
    void launch_compute_dead_mask(
        const float* opacities,
        const float* rotations,
        uint8_t* dead_mask,
        size_t N,
        float min_opacity,
        void* stream = nullptr);

    /**
     * Compute boolean mask for near-zero quaternion magnitude.
     *
     * Marks entries where ||rotation||^2 < 1e-8.
     *
     * @param rotations [N, 4] - Quaternion rotations
     * @param mask [N] - Output: boolean mask (uint8_t)
     * @param N - Number of Gaussians
     * @param stream - CUDA stream
     */
    void launch_compute_near_zero_rotation_mask(
        const float* rotations,
        uint8_t* mask,
        size_t N,
        void* stream = nullptr);

} // namespace lfs::training::pruning
