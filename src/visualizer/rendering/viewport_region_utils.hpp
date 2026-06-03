/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "internal/viewport.hpp"
#include "rendering_types.hpp"

#include <algorithm>
#include <cmath>
#include <glm/glm.hpp>

namespace lfs::vis {

    struct ResolvedFramebufferViewportRegion {
        glm::ivec2 gl_pos{0, 0};
        glm::ivec2 size{0, 0};

        [[nodiscard]] bool valid() const {
            return size.x > 0 && size.y > 0;
        }
    };

    [[nodiscard]] inline ResolvedFramebufferViewportRegion resolveFramebufferViewportRegion(
        const Viewport& viewport,
        const glm::ivec2 logical_screen_size,
        const ViewportRegion* region) {
        ResolvedFramebufferViewportRegion resolved;
        const glm::ivec2 framebuffer_size = viewport.frameBufferSize;
        if (framebuffer_size.x <= 0 || framebuffer_size.y <= 0) {
            return resolved;
        }

        if (!region) {
            resolved.size = framebuffer_size;
            return resolved;
        }

        const glm::ivec2 screen_size =
            logical_screen_size.x > 0 && logical_screen_size.y > 0
                ? logical_screen_size
                : viewport.windowSize;
        const float scale_x =
            screen_size.x > 0
                ? static_cast<float>(framebuffer_size.x) / static_cast<float>(screen_size.x)
                : 1.0f;
        const float scale_y =
            screen_size.y > 0
                ? static_cast<float>(framebuffer_size.y) / static_cast<float>(screen_size.y)
                : 1.0f;

        const int left = std::clamp(
            static_cast<int>(std::floor(region->x * scale_x)),
            0,
            framebuffer_size.x);
        const int right = std::clamp(
            static_cast<int>(std::ceil((region->x + region->width) * scale_x)),
            0,
            framebuffer_size.x);
        const int top = std::clamp(
            static_cast<int>(std::floor(region->y * scale_y)),
            0,
            framebuffer_size.y);
        const int bottom = std::clamp(
            static_cast<int>(std::ceil((region->y + region->height) * scale_y)),
            0,
            framebuffer_size.y);

        resolved.gl_pos = {left, framebuffer_size.y - bottom};
        resolved.size = {
            std::max(right - left, 0),
            std::max(bottom - top, 0),
        };
        return resolved;
    }

} // namespace lfs::vis
