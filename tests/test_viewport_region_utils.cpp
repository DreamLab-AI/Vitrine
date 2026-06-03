/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "internal/viewport.hpp"
#include "rendering/rendering_types.hpp"
#include "rendering/viewport_region_utils.hpp"

#include <gtest/gtest.h>

TEST(ViewportRegionUtilsTest, FullFramebufferRegionUsesFramebufferSize) {
    Viewport viewport(800, 600);
    viewport.windowSize = {800, 600};
    viewport.frameBufferSize = {1600, 1200};

    const auto resolved =
        lfs::vis::resolveFramebufferViewportRegion(viewport, viewport.windowSize, nullptr);

    EXPECT_EQ(resolved.gl_pos.x, 0);
    EXPECT_EQ(resolved.gl_pos.y, 0);
    EXPECT_EQ(resolved.size.x, 1600);
    EXPECT_EQ(resolved.size.y, 1200);
    EXPECT_TRUE(resolved.valid());
}

TEST(ViewportRegionUtilsTest, LogicalRegionScalesIntoFramebufferSpace) {
    Viewport viewport(800, 600);
    viewport.windowSize = {800, 600};
    viewport.frameBufferSize = {1600, 1200};

    const lfs::vis::ViewportRegion region{
        .x = 100.0f,
        .y = 50.0f,
        .width = 500.0f,
        .height = 400.0f,
    };

    const auto resolved =
        lfs::vis::resolveFramebufferViewportRegion(viewport, viewport.windowSize, &region);

    EXPECT_EQ(resolved.gl_pos.x, 200);
    EXPECT_EQ(resolved.gl_pos.y, 300);
    EXPECT_EQ(resolved.size.x, 1000);
    EXPECT_EQ(resolved.size.y, 800);
    EXPECT_TRUE(resolved.valid());
}

TEST(ViewportRegionUtilsTest, RegionIsClampedToFramebufferBounds) {
    Viewport viewport(800, 600);
    viewport.windowSize = {800, 600};
    viewport.frameBufferSize = {1600, 1200};

    const lfs::vis::ViewportRegion region{
        .x = -50.0f,
        .y = -20.0f,
        .width = 900.0f,
        .height = 700.0f,
    };

    const auto resolved =
        lfs::vis::resolveFramebufferViewportRegion(viewport, viewport.windowSize, &region);

    EXPECT_EQ(resolved.gl_pos.x, 0);
    EXPECT_EQ(resolved.gl_pos.y, 0);
    EXPECT_EQ(resolved.size.x, 1600);
    EXPECT_EQ(resolved.size.y, 1200);
    EXPECT_TRUE(resolved.valid());
}

TEST(ViewportRegionUtilsTest, RegionUsesLogicalScreenSizeInsteadOfViewportSize) {
    Viewport viewport(940, 653);
    viewport.windowSize = {940, 653};
    viewport.frameBufferSize = {2560, 1440};

    const glm::ivec2 logical_screen_size{1280, 720};
    const lfs::vis::ViewportRegion region{
        .x = 0.0f,
        .y = 24.0f,
        .width = 1095.0f,
        .height = 696.0f,
    };

    const auto resolved =
        lfs::vis::resolveFramebufferViewportRegion(viewport, logical_screen_size, &region);

    EXPECT_EQ(resolved.gl_pos.x, 0);
    EXPECT_EQ(resolved.gl_pos.y, 0);
    EXPECT_EQ(resolved.size.x, 2190);
    EXPECT_EQ(resolved.size.y, 1392);
    EXPECT_TRUE(resolved.valid());
}
