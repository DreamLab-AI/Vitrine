/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "visualizer/ipc/render_settings_convert.hpp"
#include "visualizer/ipc/view_context.hpp"
#include "visualizer/rendering/rendering_types.hpp"

#include <glm/gtc/quaternion.hpp>
#include <gtest/gtest.h>

TEST(RenderSettingsDefaults, CameraFrustumsAreDisabledByDefault) {
    const lfs::vis::RenderSettings render_settings;
    const lfs::vis::RenderSettingsProxy proxy_settings;

    EXPECT_FALSE(render_settings.show_camera_frustums);
    EXPECT_FALSE(proxy_settings.show_camera_frustums);
    EXPECT_FLOAT_EQ(render_settings.camera_frustum_scale, 0.25f);
    EXPECT_FLOAT_EQ(proxy_settings.camera_frustum_scale, 0.25f);
}

TEST(RenderSettingsProxy, DepthFilterTransformRoundTrips) {
    lfs::vis::RenderSettings render_settings;
    const glm::quat rotation = glm::normalize(glm::quat(0.9f, 0.1f, -0.2f, 0.3f));
    const glm::vec3 translation{1.25f, -2.5f, 3.75f};

    render_settings.depth_filter_enabled = true;
    render_settings.depth_filter_transform = lfs::geometry::EuclideanTransform(rotation, translation);
    render_settings.depth_filter_min = {-4.0f, -5.0f, -6.0f};
    render_settings.depth_filter_max = {4.0f, 5.0f, 6.0f};

    const auto proxy = lfs::vis::to_proxy(render_settings);

    lfs::vis::RenderSettings roundtrip;
    lfs::vis::apply_proxy(roundtrip, proxy);

    const glm::quat roundtrip_rotation = roundtrip.depth_filter_transform.getRotation();
    const glm::vec3 roundtrip_translation = roundtrip.depth_filter_transform.getTranslation();

    EXPECT_TRUE(roundtrip.depth_filter_enabled);
    EXPECT_EQ(roundtrip.depth_filter_min, render_settings.depth_filter_min);
    EXPECT_EQ(roundtrip.depth_filter_max, render_settings.depth_filter_max);
    EXPECT_FLOAT_EQ(roundtrip_rotation.w, rotation.w);
    EXPECT_FLOAT_EQ(roundtrip_rotation.x, rotation.x);
    EXPECT_FLOAT_EQ(roundtrip_rotation.y, rotation.y);
    EXPECT_FLOAT_EQ(roundtrip_rotation.z, rotation.z);
    EXPECT_FLOAT_EQ(roundtrip_translation.x, translation.x);
    EXPECT_FLOAT_EQ(roundtrip_translation.y, translation.y);
    EXPECT_FLOAT_EQ(roundtrip_translation.z, translation.z);
}
