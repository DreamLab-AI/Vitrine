/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>

namespace lfs::vis::gui {

    enum class SequencerViewportEditMode : uint8_t {
        None = 0,
        Translate,
        Rotate,
    };

} // namespace lfs::vis::gui
