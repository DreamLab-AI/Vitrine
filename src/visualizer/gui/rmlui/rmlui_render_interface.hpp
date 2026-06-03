/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/rmlui/rmlui_gl3_backend.hpp"

#include <memory>

namespace lfs::vis {
    class SceneManager;
}

namespace lfs::vis::gui {

    struct PreviewTextureCache;

    class RmlRenderInterface final : public RenderInterface_GL3 {
    public:
        RmlRenderInterface();
        ~RmlRenderInterface() override;

        void RenderGeometry(Rml::CompiledGeometryHandle handle, Rml::Vector2f translation,
                            Rml::TextureHandle texture) override;
        Rml::TextureHandle LoadTexture(Rml::Vector2i& dimensions, const Rml::String& source) override;
        void ReleaseTexture(Rml::TextureHandle texture_handle) override;

        void set_scene_manager(lfs::vis::SceneManager* scene_manager);
        void process_pending_preview_uploads();
        void clear_pending_preview_loads();
        void register_external_texture(const Rml::String& source, unsigned int texture_id,
                                       int width, int height, bool flip_y = false);
        void unregister_external_texture(const Rml::String& source);

    private:
        bool is_external_texture(Rml::TextureHandle texture_handle) const;
        Rml::TextureHandle load_external_texture(Rml::Vector2i& dimensions, const Rml::String& source);
        Rml::TextureHandle load_preview_texture(Rml::Vector2i& dimensions, const Rml::String& source);

        lfs::vis::SceneManager* scene_manager_ = nullptr;
        std::unique_ptr<PreviewTextureCache> preview_cache_;
    };

} // namespace lfs::vis::gui
