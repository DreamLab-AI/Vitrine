/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gl_resources.hpp"
#include "rendering/frame_contract.hpp"
#include "shader_manager.hpp"
#include <filesystem>

namespace lfs::rendering {

    class EnvironmentRenderer {
    public:
        Result<void> initialize();
        [[nodiscard]] bool isInitialized() const { return initialized_; }

        Result<void> render(const FrameView& frame_view,
                            const std::filesystem::path& environment_path,
                            float exposure,
                            float rotation_degrees,
                            bool equirectangular_view);

    private:
        Result<void> ensureTextureLoaded(const std::filesystem::path& environment_path);

        ManagedShader shader_;
        VAO vao_;
        VBO vbo_;
        Texture environment_texture_;
        std::filesystem::path loaded_environment_path_;
        std::string last_load_error_;
        bool texture_ready_ = false;
        bool initialized_ = false;
    };

} // namespace lfs::rendering
