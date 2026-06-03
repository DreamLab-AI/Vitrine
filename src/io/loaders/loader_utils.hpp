/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "io/loader.hpp"
#include <algorithm>
#include <memory>
#include <vector>

namespace lfs::io {

    inline bool detect_camera_alpha(const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras,
                                    const CancelCallback& cancel_requested = nullptr) {
        bool images_have_alpha = false;
        size_t alpha_count = 0;
        for (size_t i = 0; i < cameras.size(); ++i) {
            if (cancel_requested && (i % 64) == 0 && cancel_requested()) {
                throw LoadCancelledError("Image alpha probe cancelled");
            }

            const auto& cam = cameras[i];
            auto ext = cam->image_path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".jpg" || ext == ".jpeg")
                continue;
            try {
                auto [w, h, c] = lfs::core::get_image_info(cam->image_path());
                if (c == 4) {
                    cam->set_has_alpha(true);
                    images_have_alpha = true;
                    ++alpha_count;
                }
            } catch (const std::exception& e) {
                LOG_DEBUG("Failed to probe alpha for '{}': {}", cam->image_name(), e.what());
            }
        }
        if (alpha_count > 0) {
            LOG_INFO("Alpha channel detected in {}/{} images", alpha_count, cameras.size());
        }
        return images_have_alpha;
    }

} // namespace lfs::io
