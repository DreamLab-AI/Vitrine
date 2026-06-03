/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "../render_pass.hpp"
#include <filesystem>
#include <rendering/environment_renderer.hpp>
#include <string>

namespace lfs::vis {

    class EnvironmentPass final : public RenderPass {
    public:
        [[nodiscard]] const char* name() const override { return "EnvironmentPass"; }
        [[nodiscard]] DirtyMask sensitivity() const override {
            return DirtyFlag::BACKGROUND | DirtyFlag::CAMERA | DirtyFlag::VIEWPORT | DirtyFlag::SPLIT_VIEW;
        }

        [[nodiscard]] bool shouldExecute(DirtyMask frame_dirty, const FrameContext& ctx) const override;

        void execute(lfs::rendering::RenderingEngine& engine,
                     const FrameContext& ctx,
                     FrameResources& res) override;

    private:
        [[nodiscard]] std::filesystem::path resolveEnvironmentPathCached(const std::string& path_value);

        lfs::rendering::EnvironmentRenderer renderer_;
        std::string cached_environment_path_value_;
        std::filesystem::path cached_environment_resolved_path_;
        std::string last_environment_error_;
    };

} // namespace lfs::vis
