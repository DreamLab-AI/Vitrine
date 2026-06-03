/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/splat_data.hpp"
#include <expected>
#include <filesystem>
#include <string>

namespace lfs::io {

    using lfs::core::SplatData;

    // Cheap archive sniff used to route .usdz files to the correct loader.
    std::expected<bool, std::string> is_nurec_usdz(const std::filesystem::path& filepath);

    // Load NuRec payloads packaged inside USDZ archives.
    std::expected<SplatData, std::string> load_nurec_usdz(const std::filesystem::path& filepath);
    std::expected<void, std::string> validate_nurec_usdz(const std::filesystem::path& filepath);

} // namespace lfs::io
