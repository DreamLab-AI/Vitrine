/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "tools/align_tool.hpp"
#include "core/services.hpp"
#include "gui/gui_focus_state.hpp"
#include "internal/viewport.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/rendering_manager.hpp"
#include "theme/theme.hpp"
#include <algorithm>
#include <optional>
#include <imgui.h>

namespace lfs::vis::tools {

    AlignTool::AlignTool() = default;

    bool AlignTool::initialize(const ToolContext& ctx) {
        tool_context_ = &ctx;
        return true;
    }

    void AlignTool::shutdown() {
        tool_context_ = nullptr;
        services().clearAlignPickedPoints();
    }

    void AlignTool::update([[maybe_unused]] const ToolContext& ctx) {}

    namespace {
        struct PanelProjection {
            lfs::vis::RenderingManager::ViewerPanelInfo info{};
            Viewport viewport;
            float focal_length_mm = lfs::rendering::DEFAULT_FOCAL_LENGTH_MM;
            float screen_scale_x = 1.0f;
            float screen_scale_y = 1.0f;
        };

        [[nodiscard]] std::optional<PanelProjection> resolvePanelProjection(const ToolContext& ctx,
                                                                            const glm::vec2& screen_point,
                                                                            const float fallback_focal_length_mm) {
            auto* const rm = ctx.getRenderingManager();
            if (!rm) {
                return std::nullopt;
            }

            const auto& bounds = ctx.getViewportBounds();
            const glm::vec2 viewport_pos(bounds.x, bounds.y);
            const glm::vec2 viewport_size(bounds.width, bounds.height);
            const auto panel_info = rm->resolveViewerPanel(
                ctx.getViewport(),
                viewport_pos,
                viewport_size,
                screen_point);
            if (!panel_info || !panel_info->valid()) {
                return std::nullopt;
            }

            PanelProjection proj{};
            proj.info = *panel_info;
            proj.focal_length_mm = rm->getFocalLengthMm();
            if (proj.focal_length_mm <= 0.0f) {
                proj.focal_length_mm = fallback_focal_length_mm;
            }

            proj.viewport = *panel_info->viewport;
            proj.viewport.windowSize = {panel_info->render_width, panel_info->render_height};
            proj.screen_scale_x = panel_info->width / static_cast<float>(std::max(panel_info->render_width, 1));
            proj.screen_scale_y = panel_info->height / static_cast<float>(std::max(panel_info->render_height, 1));
            return proj;
        }

        [[nodiscard]] glm::vec2 screenToRender(const PanelProjection& proj, const glm::vec2& screen_point) {
            const float scale_x =
                static_cast<float>(proj.info.render_width) / std::max(proj.info.width, 1.0f);
            const float scale_y =
                static_cast<float>(proj.info.render_height) / std::max(proj.info.height, 1.0f);
            return {(screen_point.x - proj.info.x) * scale_x,
                    (screen_point.y - proj.info.y) * scale_y};
        }

        [[nodiscard]] ImVec2 renderToScreen(const PanelProjection& proj, const glm::vec2& render_point) {
            return ImVec2(
                proj.info.x + render_point.x * proj.screen_scale_x,
                proj.info.y + render_point.y * proj.screen_scale_y);
        }

        [[nodiscard]] ImVec2 projectToScreen(const PanelProjection& proj, const glm::vec3& world_pos) {
            const auto projected = lfs::rendering::projectWorldPoint(
                proj.viewport.camera.R,
                proj.viewport.camera.t,
                proj.viewport.windowSize,
                world_pos,
                proj.focal_length_mm);
            if (!projected) {
                return ImVec2(-1000, -1000);
            }

            return renderToScreen(proj, glm::vec2(projected->x, projected->y));
        }
    } // namespace

    static float calculateScreenRadius(const glm::vec3& world_pos,
                                       const float world_radius,
                                       const Viewport& viewport,
                                       const float focal_length_mm) {
        const glm::mat4 view = viewport.getViewMatrix();
        const glm::mat4 proj = viewport.getProjectionMatrix(focal_length_mm);
        const glm::vec4 view_pos = view * glm::vec4(world_pos, 1.0f);
        const float depth = -view_pos.z;

        if (depth <= 0.0f)
            return 0.0f;

        const float screen_radius = (world_radius * proj[1][1] * viewport.windowSize.y) / (2.0f * depth);
        return screen_radius;
    }

    void AlignTool::renderUI([[maybe_unused]] const lfs::vis::gui::UIContext& ui_ctx,
                             [[maybe_unused]] bool* p_open) {
        if (!isEnabled() || !tool_context_)
            return;

        ImDrawList* const draw_list = ImGui::GetForegroundDrawList();
        const ImVec2 mouse_pos = ImGui::GetMousePos();
        auto* const rendering_manager = tool_context_->getRenderingManager();
        const float fallback_focal_length_mm = rendering_manager
                                                   ? rendering_manager->getFocalLengthMm()
                                                   : lfs::rendering::DEFAULT_FOCAL_LENGTH_MM;
        const bool over_gui = gui::guiFocusState().want_capture_mouse;

        const auto& bounds = tool_context_->getViewportBounds();

        // Pick an active render panel based on cursor position (handles split view and HiDPI scaling).
        const auto panel_proj_opt = resolvePanelProjection(
            *tool_context_,
            {mouse_pos.x, mouse_pos.y},
            fallback_focal_length_mm);

        const glm::ivec2 rendered_size = rendering_manager
                                             ? rendering_manager->getRenderedSize()
                                             : glm::ivec2(0, 0);
        const int fallback_render_width =
            rendered_size.x > 0 ? rendered_size.x : std::max(tool_context_->getViewport().windowSize.x, 1);
        const int fallback_render_height =
            rendered_size.y > 0 ? rendered_size.y : std::max(tool_context_->getViewport().windowSize.y, 1);

        PanelProjection panel_proj_fallback{};
        panel_proj_fallback.info.panel = SplitViewPanelId::Left;
        panel_proj_fallback.info.viewport = &tool_context_->getViewport();
        panel_proj_fallback.info.x = bounds.x;
        panel_proj_fallback.info.y = bounds.y;
        panel_proj_fallback.info.width = bounds.width;
        panel_proj_fallback.info.height = bounds.height;
        panel_proj_fallback.info.render_width = fallback_render_width;
        panel_proj_fallback.info.render_height = fallback_render_height;
        panel_proj_fallback.focal_length_mm = fallback_focal_length_mm;
        panel_proj_fallback.viewport = tool_context_->getViewport();
        panel_proj_fallback.viewport.windowSize = {fallback_render_width, fallback_render_height};
        panel_proj_fallback.screen_scale_x = bounds.width / static_cast<float>(fallback_render_width);
        panel_proj_fallback.screen_scale_y = bounds.height / static_cast<float>(fallback_render_height);

        const PanelProjection& panel_proj = panel_proj_opt ? *panel_proj_opt : panel_proj_fallback;

        constexpr float SPHERE_RADIUS = 0.05f;
        const auto& t = theme();
        const ImU32 SPHERE_COLOR = t.error_u32();
        const ImU32 SPHERE_OUTLINE = t.overlay_text_u32();
        const ImU32 PREVIEW_COLOR = toU32WithAlpha(t.palette.error, 0.6f);
        const ImU32 CROSSHAIR_COLOR = toU32WithAlpha(t.palette.error, 0.8f);

        // Get picked points from services
        const auto& picked_points = services().getAlignPickedPoints();

        // Draw picked points
        for (size_t i = 0; i < picked_points.size(); ++i) {
            const ImVec2 screen_pos = projectToScreen(panel_proj, picked_points[i]);
            const float radius_render = calculateScreenRadius(
                picked_points[i], SPHERE_RADIUS, panel_proj.viewport, panel_proj.focal_length_mm);
            const float screen_radius =
                glm::clamp(radius_render * glm::min(panel_proj.screen_scale_x, panel_proj.screen_scale_y), 5.0f, 50.0f);

            draw_list->AddCircleFilled(screen_pos, screen_radius, SPHERE_COLOR, 32);
            draw_list->AddCircle(screen_pos, screen_radius, SPHERE_OUTLINE, 32, 1.5f);

            const char label = '1' + static_cast<char>(i);
            draw_list->AddText(ImVec2(screen_pos.x - 4, screen_pos.y - 6), t.overlay_text_u32(), &label, &label + 1);
        }

        if (over_gui)
            return;

        draw_list->AddCircle(mouse_pos, 5.0f, CROSSHAIR_COLOR, 16, 2.0f);

        // Live preview at mouse position
        if (picked_points.size() < 3 && rendering_manager) {
            const auto render_point = screenToRender(panel_proj, {mouse_pos.x, mouse_pos.y});
            const float depth = rendering_manager->getDepthAtPixel(
                static_cast<int>(render_point.x),
                static_cast<int>(render_point.y),
                panel_proj_opt ? std::optional<SplitViewPanelId>(panel_proj.info.panel) : std::nullopt);

            if (depth > 0.0f && depth < 1e9f) {
                const glm::vec3 preview_point = panel_proj.viewport.unprojectPixel(
                    render_point.x, render_point.y, depth, panel_proj.focal_length_mm);
                if (Viewport::isValidWorldPosition(preview_point)) {
                    const ImVec2 screen_pos = projectToScreen(panel_proj, preview_point);
                    const float radius_render = calculateScreenRadius(
                        preview_point, SPHERE_RADIUS, panel_proj.viewport, panel_proj.focal_length_mm);
                    const float screen_radius = glm::clamp(
                        radius_render * glm::min(panel_proj.screen_scale_x, panel_proj.screen_scale_y), 5.0f, 50.0f);

                    draw_list->AddCircleFilled(screen_pos, screen_radius, PREVIEW_COLOR, 32);
                    draw_list->AddCircle(screen_pos, screen_radius, toU32WithAlpha(t.palette.text, 0.6f), 32, 1.5f);

                    const char label = '1' + static_cast<char>(picked_points.size());
                    draw_list->AddText(ImVec2(screen_pos.x - 4, screen_pos.y - 6), toU32WithAlpha(t.palette.text, 0.7f), &label, &label + 1);
                }
            }
        }

        // Normal preview when 2 points picked
        if (picked_points.size() == 2 && rendering_manager) {
            const auto render_point = screenToRender(panel_proj, {mouse_pos.x, mouse_pos.y});
            const float depth = rendering_manager->getDepthAtPixel(
                static_cast<int>(render_point.x),
                static_cast<int>(render_point.y),
                panel_proj_opt ? std::optional<SplitViewPanelId>(panel_proj.info.panel) : std::nullopt);

            if (depth > 0.0f && depth < 1e9f) {
                const glm::vec3 p2 = panel_proj.viewport.unprojectPixel(
                    render_point.x, render_point.y, depth, panel_proj.focal_length_mm);
                if (Viewport::isValidWorldPosition(p2)) {
                    const glm::vec3& p0 = picked_points[0];
                    const glm::vec3& p1 = picked_points[1];

                    const glm::vec3 v01 = p1 - p0;
                    const glm::vec3 v02 = p2 - p0;
                    glm::vec3 normal = glm::normalize(glm::cross(v01, v02));
                    constexpr glm::vec3 kTargetUp(0.0f, 1.0f, 0.0f);
                    if (glm::dot(normal, kTargetUp) < 0.0f)
                        normal = -normal;

                    const glm::vec3 center = (p0 + p1 + p2) / 3.0f;
                    const float line_length = glm::max(glm::length(v01) * 0.5f, 0.1f);
                    const glm::vec3 normal_end = center + normal * line_length;

                    const ImVec2 center_screen = projectToScreen(panel_proj, center);
                    const ImVec2 normal_screen = projectToScreen(panel_proj, normal_end);

                    draw_list->AddLine(center_screen, normal_screen, IM_COL32(255, 255, 0, 255), 4.0f);
                    draw_list->AddCircleFilled(normal_screen, 10.0f, IM_COL32(255, 255, 0, 255));
                    draw_list->AddText(ImVec2(normal_screen.x + 12, normal_screen.y - 8), IM_COL32(255, 255, 0, 255), "UP");

                    const ImVec2 p0_screen = projectToScreen(panel_proj, p0);
                    const ImVec2 p1_screen = projectToScreen(panel_proj, p1);
                    const ImVec2 p2_screen = projectToScreen(panel_proj, p2);
                    draw_list->AddLine(p0_screen, p1_screen, IM_COL32(255, 0, 0, 200), 2.0f);
                    draw_list->AddLine(p1_screen, p2_screen, IM_COL32(0, 255, 0, 200), 2.0f);
                    draw_list->AddLine(p2_screen, p0_screen, IM_COL32(0, 0, 255, 200), 2.0f);
                }
            }
        }

        // Instructions
        const char* instruction = nullptr;
        switch (picked_points.size()) {
        case 0: instruction = "Click 1st point"; break;
        case 1: instruction = "Click 2nd point"; break;
        case 2: instruction = "Click 3rd point"; break;
        default: break;
        }
        if (instruction) {
            draw_list->AddText(ImVec2(mouse_pos.x + 15, mouse_pos.y - 10), CROSSHAIR_COLOR, instruction);
        }

        char count_text[16];
        snprintf(count_text, sizeof(count_text), "Points: %zu/3", picked_points.size());
        draw_list->AddText(ImVec2(10, 50), t.overlay_text_u32(), count_text);
    }

    void AlignTool::onEnabledChanged(bool enabled) {
        if (!enabled) {
            services().clearAlignPickedPoints();
        }
        if (tool_context_) {
            tool_context_->requestRender();
        }
    }

} // namespace lfs::vis::tools
